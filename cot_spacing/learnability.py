"""How readily gpt-oss would learn to glue words, or to shorten them.

``gradients``: a first-order estimate of what one fine-tuning step does. The
step is taken on four excerpts' rewritten reasoning (gluing, shortening or
both); what is measured is how much it raises the fifth, held-out excerpt's
log-probability of its rewritten reasoning *over* its spaced reasoning. That
margin cancels whatever the step teaches about the content, leaving what it
teaches about the rewrite. The step has a fixed length in parameter space, so
the three rewrites are compared at equal cost. Gradients are taken with
respect to every attention weight; experts, embeddings and norms are held
fixed.

``sites``: where an RL run would start from. At every point where the base
model, reading the spaced reasoning, could drop the next space or write the
next word's short form, how probable is that edit compared with what the
spaced reading has? A policy gradient raises an action's probability at a rate
proportional to how often it is already sampled times the reward it earns, so
these probabilities, weighted by characters saved, are the starting pull a
character penalty would exert on each habit.

Usage:
    python -m cot_spacing.learnability gradients --out <results>/gradients.json
    python -m cot_spacing.learnability sites --out <results>/sites.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from cot_spacing.analysis import glued_only, segmentations
from cot_spacing.data import load_excerpts
from cot_spacing.steering import DIRECTIONS, EXP_DIR, load_examples
from cot_spacing.variants import VARIANTS, shorten

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from cot_spacing.segment import Segmentation
    from cot_spacing.variants import Example

EXAMPLES = EXP_DIR / "results" / "variants.json"


def attention_params(model: PreTrainedModel) -> list[torch.nn.Parameter]:
    from cot_spacing.gpt_oss import layers

    return [
        p
        for layer in layers(model)
        for p in layer.get_submodule("self_attn").parameters()
    ]


def gradients(out: Path) -> None:
    from cot_spacing.gpt_oss import load, with_reasoning

    tok, model = load()
    params = attention_params(model)
    for p in params:
        p.requires_grad_(True)
    examples = load_examples(EXAMPLES)
    keys = [(e.id, v) for e in examples for v in VARIANTS]
    logp: dict[str, float] = {}
    grads: list[torch.Tensor] = []
    for e in examples:
        for v in VARIANTS:
            ids, span = with_reasoning(tok, e.user, e.texts[v])
            x = torch.tensor([ids], device="cuda")
            logits = model(x).logits[0, span.start - 1 : span.stop - 1].float()
            lp = torch.log_softmax(logits, -1)
            target = x[0, span].to(lp.device)
            total = lp.gather(-1, target[:, None]).sum()
            model.zero_grad(set_to_none=True)
            (-total).backward()
            grads.append(
                torch.cat([p.grad.flatten().to("cpu", torch.bfloat16) for p in params])  # type: ignore[union-attr]
            )
            logp[f"{e.id}/{v}"] = total.item()
            print(f"{e.id} {v}: log p = {total.item():.1f}")
    model.zero_grad(set_to_none=True)

    n = len(grads)
    gram = torch.zeros(n, n, dtype=torch.float64)
    chunk = 1 << 25
    for start in range(0, grads[0].numel(), chunk):
        block = torch.stack([g[start : start + chunk] for g in grads]).to(
            "cuda", torch.float32
        )
        gram += (block @ block.T).double().cpu()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "keys": [f"{i}/{v}" for i, v in keys],
                "n_params": grads[0].numel(),
                "logp": logp,
                "gram": gram.tolist(),
            },
            indent=1,
        )
        + "\n"
    )


def edits(e: Example, direction: str) -> int:
    return {
        "glued": e.glued_boundaries,
        "short": e.shortened_words,
        "both": e.glued_boundaries + e.shortened_words,
    }[direction]


def step_gains(gradients_path: Path) -> list[dict[str, object]]:
    """Gain in an excerpt's rewrite-over-spaced margin per unit training step.

    With g_i the gradient of excerpt i's loss on its rewrite and s_j that on
    its spaced reading, a unit step down the summed loss of excerpts T moves
    excerpt j's margin log p(rewrite_j) - log p(spaced_j) by
    <sum_T g_i, g_j - s_j> / |sum_T g_i|. ``held_out`` trains on the other
    excerpts; ``same`` trains on excerpt j alone, the upper bound for a rewrite
    that does not carry over from one excerpt's words to another's.
    """
    saved = json.loads(gradients_path.read_text())
    examples = load_examples(EXAMPLES)
    at = {k: i for i, k in enumerate(saved["keys"])}
    gram = torch.tensor(saved["gram"], dtype=torch.float64)
    logp = saved["logp"]

    def gain(train: list[int], held: str, d: str) -> float:
        g_norm = math.sqrt(gram[train][:, train].sum().item())
        toward = gram[train, at[f"{held}/{d}"]] - gram[train, at[f"{held}/spaced"]]
        return toward.sum().item() / g_norm

    rows = []
    for d in DIRECTIONS:
        for held in examples:
            others = [at[f"{e.id}/{d}"] for e in examples if e.id != held.id]
            held_out = gain(others, held.id, d)
            same = gain([at[f"{held.id}/{d}"]], held.id, d)
            rows.append(
                {
                    "direction": d,
                    "excerpt": held.id,
                    "margin": logp[f"{held.id}/{d}"] - logp[f"{held.id}/spaced"],
                    "edits": edits(held, d),
                    "chars_saved": held.chars_saved[d],
                    "held_out_per_edit": held_out / edits(held, d),
                    "held_out_per_char": held_out / held.chars_saved[d],
                    "same_per_edit": same / edits(held, d),
                    "same_per_char": same / held.chars_saved[d],
                }
            )
    return rows


Site = tuple[dict[str, object], list[int], list[int], list[int]]


def site_sequences(tok: PreTrainedTokenizerBase) -> list[Site]:
    """Every glue and shorten opportunity in the spaced reasoning.

    Each is (site, prefix ids, original continuation, edited continuation).
    A continuation runs through the first token of the following word, so it
    also scores that the edited word ended where it did.
    """
    from cot_spacing.gpt_oss import prompt

    examples = {e.id: e for e in load_examples(EXAMPLES)}
    out = []
    for excerpt, seg in glued_only(segmentations(load_excerpts())):
        if excerpt.id in examples:
            head = prompt(tok, examples[excerpt.id].user)
            out += excerpt_sites(
                tok, head + "<|channel|>analysis<|message|>", excerpt.id, seg
            )
    return out


def excerpt_sites(
    tok: PreTrainedTokenizerBase, head: str, excerpt_id: str, seg: Segmentation
) -> list[Site]:
    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    words, gaps = seg.words, seg.spaced_gaps()
    out: list[Site] = []
    for k, word in enumerate(words):
        lead = gaps[k - 1] if k else ""
        before = "".join(w + g for w, g in zip(words[:k], gaps[:k], strict=True))
        prefix = encode(head) + encode(before[: len(before) - len(lead)])
        following = gaps[k] + words[k + 1] if k + 1 < len(words) else "<|end|>"
        tail = encode(following)[:1]
        site = {"excerpt": excerpt_id, "index": k, "word": word}
        if lead == " ":
            glue = {
                "kind": "glue",
                "chars_saved": 1,
                "gpt6_glued": seg.gaps[k - 1] == "",
            }
            out.append(
                (
                    {**site, **glue},
                    prefix,
                    encode(" " + word) + tail,
                    encode(word) + tail,
                )
            )
        short = shorten(word)
        if short != word:
            abbrev = {
                "kind": "short",
                "short": short,
                "chars_saved": len(word) - len(short),
            }
            out.append(
                (
                    {**site, **abbrev},
                    prefix,
                    encode(lead + word) + tail,
                    encode(lead + short) + tail,
                )
            )
    return out


@torch.no_grad()
def continuation_logp(
    model: PreTrainedModel, batch: list[tuple[list[int], list[int]]]
) -> list[float]:
    longest = max(len(p) + len(c) for p, c in batch)
    ids = torch.zeros(len(batch), longest, dtype=torch.long)
    for row, (p, c) in enumerate(batch):
        ids[row, : len(p) + len(c)] = torch.tensor(p + c)
    hidden = model.model(ids.to("cuda")).last_hidden_state  # type: ignore[operator]
    out = []
    for row, (p, c) in enumerate(batch):
        positions = len(p) - 1 + torch.arange(len(c))
        logits = model.lm_head(hidden[row, positions]).float()  # type: ignore[operator]
        lp = torch.log_softmax(logits, -1).cpu()
        out.append(lp[torch.arange(len(c)), torch.tensor(c)].sum().item())
    return out


def sites(out: Path, batch_size: int = 16) -> None:
    from cot_spacing.gpt_oss import load

    tok, model = load()
    seqs = site_sequences(tok)
    pairs = [(p, c) for _, p, orig, edit in seqs for c in (orig, edit)]
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        scores += continuation_logp(model, pairs[start : start + batch_size])
    rows = [
        {**site, "log_ratio": scores[2 * i + 1] - scores[2 * i]}
        for i, (site, *_rest) in enumerate(seqs)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sites": rows}, indent=1) + "\n")
    print(json.dumps(site_summary(out), indent=1))


def site_summary(sites_path: Path) -> dict[str, dict[str, float]]:
    """Per kind of edit: how probable it is, and what it would save.

    The expected saving is a mean over sites and so is carried by the few
    sites where the edit is nearly as likely as the original; the median
    ratio is the typical site.
    """
    rows = json.loads(sites_path.read_text())["sites"]
    summary = {}
    for kind in ("glue", "short"):
        rs = [r for r in rows if r["kind"] == kind]
        log10 = [r["log_ratio"] / math.log(10) for r in rs]
        summary[kind] = {
            "opportunities": len(rs),
            "median_log10_ratio": statistics.median(log10),
            "share_within_100x": statistics.mean(x >= -2 for x in log10),
            "chars_per_opportunity": statistics.mean(r["chars_saved"] for r in rs),
            "expected_chars_per_opportunity": statistics.mean(
                10**x * r["chars_saved"] for x, r in zip(log10, rs, strict=True)
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("gradients", "sites"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "gradients":
        gradients(args.out)
    else:
        sites(args.out)


if __name__ == "__main__":
    main()
