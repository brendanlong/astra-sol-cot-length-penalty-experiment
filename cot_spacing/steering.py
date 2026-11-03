"""Steering gpt-oss toward gluing words, shortening them, or both.

Directions: ``glued``, ``short`` and ``both`` (see :mod:`cot_spacing.variants`);
``nospace``, which also drops the spaces GPT-6 kept after sentences and
clauses; and ``glued_wordwise``, GPT-6's glued text tokenized a word at a time
rather than canonically.

``vectors``: each direction is a difference of means of the residual stream
over the reasoning tokens, a variant minus the spaced reading of the same
excerpts (see :mod:`cot_spacing.variants`). Held-out separation is checked by
leaving each excerpt out in turn.

``generate``: adds one direction at one layer while gpt-oss answers held-out
prompts, at one strength. Every direction is scaled to the same length,
a fraction of the typical residual norm at that layer, so a direction that
needs less push to produce its behaviour is the easier one to steer.

``score``: how much each generation glues and shortens, which spaces it
keeps, whether its tokens are the canonical tokenization of its text, whether
it loops or breaks into garbage, and whether it still gets the arithmetic
right.

Usage:
    python -m cot_spacing.steering vectors --out results/steering/vectors.pt
    python -m cot_spacing.steering generate --vectors ... --direction glued \
        --layer 12 --scale 0.25 --out results/steering/gen/glued-12-0.25.jsonl
    python -m cot_spacing.steering score results/steering/gen/*.jsonl --out ...
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path

import torch

from cot_spacing.variants import SHORT_FORMS, Example

EXP_DIR = Path(__file__).parent
DIRECTIONS = ("glued", "short", "both", "nospace", "glued_wordwise")
PROMPT_FILES = (
    EXP_DIR / "corpus" / "gsm8k_test_first32.json",
    EXP_DIR / "corpus" / "agentic_prompts.json",
)
MAX_NEW_TOKENS = 1024


def load_examples(path: Path) -> list[Example]:
    return [Example.model_validate(d) for d in json.loads(path.read_text())]


@torch.no_grad()
def reasoning_states(
    tok: object, model: object, user: str, cot: list[int]
) -> torch.Tensor:
    """Each decoder block's output at each reasoning token: [layer, tok, d].

    Read with hooks on the blocks rather than from ``output_hidden_states``,
    whose last entry has the final norm applied: :func:`gpt_oss.steer` adds to
    the block output, so that is what has to be measured.
    """
    from cot_spacing.gpt_oss import layers, with_reasoning

    ids, span = with_reasoning(tok, user, cot)  # type: ignore[arg-type]
    captured: list[torch.Tensor] = []
    handles = [
        block.register_forward_hook(
            lambda _m, _i, output: captured.append(output[0, span].float().cpu())
        )
        for block in layers(model)  # type: ignore[arg-type]
    ]
    try:
        model(torch.tensor([ids], device="cuda"))  # type: ignore[operator]
    finally:
        for h in handles:
            h.remove()
    return torch.stack(captured)


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Probability a random ``pos`` scores above a random ``neg``."""
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean()).item()


def vectors(examples_path: Path, out: Path) -> None:
    from cot_spacing.gpt_oss import load

    tok, model = load()
    examples = load_examples(examples_path)
    states = {
        (e.id, v): reasoning_states(tok, model, e.user, ids)
        for e in examples
        for v, ids in e.ids.items()
    }

    def difference(variant: str, ids: list[str]) -> torch.Tensor:
        """Mean over excerpts of each excerpt's own variant-minus-spaced mean.

        Every excerpt weighs the same on both sides. Pooling tokens instead
        would weigh an excerpt by its token count, which differs between
        renderings, and mix a content difference into the direction.
        """
        return torch.stack(
            [
                states[i, variant].mean(dim=1) - states[i, "spaced"].mean(dim=1)
                for i in ids
            ]
        ).mean(dim=0)

    ids = [e.id for e in examples]
    directions = {d: difference(d, ids) for d in DIRECTIONS}
    resid_norm = torch.cat([states[i, "spaced"] for i in ids], dim=1).norm(dim=-1)
    n_layers = resid_norm.shape[0]

    held_out: dict[str, list[list[float]]] = {d: [] for d in DIRECTIONS}
    for d in DIRECTIONS:
        for held in ids:
            rest = [i for i in ids if i != held]
            v = difference(d, rest)
            held_out[d].append(
                [
                    auc(
                        states[held, d][layer] @ v[layer],
                        states[held, "spaced"][layer] @ v[layer],
                    )
                    for layer in range(n_layers)
                ]
            )

    def cos(a: torch.Tensor, b: torch.Tensor) -> list[float]:
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"directions": directions, "resid_norm": resid_norm.median(dim=1).values}, out
    )
    stats = {
        "excerpts": ids,
        "tokens": {
            v: sum(states[i, v].shape[1] for i in ids) for v in ("spaced", *DIRECTIONS)
        },
        "resid_norm_median": resid_norm.median(dim=1).values.tolist(),
        "direction_norm": {d: v.norm(dim=-1).tolist() for d, v in directions.items()},
        "held_out_auc": {
            d: [statistics.mean(r[layer] for r in rows) for layer in range(n_layers)]
            for d, rows in held_out.items()
        },
        "held_out_auc_min": {
            d: [min(r[layer] for r in rows) for layer in range(n_layers)]
            for d, rows in held_out.items()
        },
        "cosine": {
            "glued_short": cos(directions["glued"], directions["short"]),
            "glued_both": cos(directions["glued"], directions["both"]),
            "short_both": cos(directions["short"], directions["both"]),
            "both_vs_sum": cos(
                directions["both"], directions["glued"] + directions["short"]
            ),
        },
    }
    out.with_suffix(".json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({k: stats[k] for k in ("tokens",)}, indent=1))


def load_prompts() -> list[dict[str, str]]:
    return [p for f in PROMPT_FILES for p in json.loads(f.read_text())]


@torch.no_grad()
def generate(
    vectors_path: Path, direction: str, layer: int, scale: float, out: Path
) -> None:
    from cot_spacing.gpt_oss import load, prompt, steer

    tok, model = load()
    saved = torch.load(vectors_path)
    unit = saved["directions"][direction][layer]
    unit = unit / unit.norm()
    resid = saved["resid_norm"][layer].item()
    prompts = load_prompts()
    tok.padding_side = "left"
    batch = tok(
        [prompt(tok, p["question"]) for p in prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to("cuda")

    with steer(model, layer, unit * scale * resid):
        gen = model.generate(  # type: ignore[operator]
            **batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    with tmp.open("w") as f:
        for p, row in zip(prompts, gen[:, batch["input_ids"].shape[1] :], strict=True):
            ids = [int(i) for i in row if i != tok.pad_token_id]
            text = str(tok.decode(ids, skip_special_tokens=False))
            record = {
                "direction": direction,
                "layer": layer,
                "scale": scale,
                "norm": scale * resid,
                "id": p["id"],
                "text": text,
                "ids": ids,
            }
            f.write(json.dumps(record) + "\n")
    tmp.rename(out)


ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)
FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(<\|return\|>|<\|end\|>|$)", re.S
)
LETTERS = re.compile(r"[A-Za-z]")
WORD = re.compile(r"(?<![\w/])[A-Za-z]+(?:/o|/)?(?![\w])")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
#: Code is excluded from short-form counts: `git diff` or `<pkg>` there is
#: syntax, not a choice of word.
CODE = re.compile(r"```.*?(?:```|$)|`[^`\n]*`", re.S)


#: Thousands separators gpt-oss writes between digit groups.
SEPARATOR = re.compile(r"(?<=\d)(?:,|\{,\}|\\[,;!]|\u202f|\u2009)(?=\d{3}(?!\d))")


def stated_numbers(answer: str) -> set[float]:
    """Numbers the answer presents as its result.

    The last number, and every number in a boxed span (else in a bold one).
    Taking all of a marked span's numbers rather than guessing which is the
    result ("**2 + 1 = 3 bolts**", "**160 minutes** = **2 h 40 min**") is
    lenient, but the same for every run.
    """
    text = SEPARATOR.sub("", answer)
    numbers = NUMBER.findall(text)
    stated = {numbers[-1]} if numbers else set()
    for pattern in (BOXED, BOLD):
        marked = [n for m in pattern.findall(text) for n in NUMBER.findall(m)]
        if marked:
            stated |= set(marked)
            break
    return {float(n.replace(",", "")) for n in stated}


LONG = set(SHORT_FORMS)
SHORT = set(SHORT_FORMS.values()) - LONG

#: How far a run may fall below unsteered gpt-oss and still count as coherent.
COHERENT_DROP = 0.1
#: Below this many words from the short-form list, a share is too noisy to report.
MIN_LIST_WORDS = 20


def letters_per_space(text: str) -> float | None:
    chunks = [c for c in text.split() if LETTERS.search(c)]
    return sum(map(str.isalpha, text)) / len(chunks) if chunks else None


#: A space directly after the end of a sentence, or of a clause, followed by
#: more words: ``done. Next``, ``a, b``. Decimals and line ends are not counted.
AFTER_SENTENCE = re.compile(r"(?<=[A-Za-z0-9)\"'])[.?!]( ?)(?=[A-Za-z])")
AFTER_CLAUSE = re.compile(r"(?<=[A-Za-z0-9)\"'])[,:;]( ?)(?=[A-Za-z])")
BETWEEN_WORDS = re.compile(r"(?<=[^\s.?!,:;]) (?=\S)")


def degenerate(text: str) -> bool:
    """Stuck in a loop, or broken into undecodable bytes.

    A loop shows as many repeated 30-character windows: a third of them
    repeated is far beyond anything unsteered gpt-oss writes.
    """
    windows = [text[i : i + 30] for i in range(0, max(0, len(text) - 30), 10)]
    repeated = 1 - len(set(windows)) / len(windows) if windows else 0.0
    return repeated > 0.3 or text.count("\ufffd") >= 3


def spacing(reasoning: str) -> dict[str, int]:
    """Which spaces the reasoning keeps: after sentences, clauses, other words."""
    text = CODE.sub(" ", reasoning)
    sentence = AFTER_SENTENCE.findall(text)
    clause = AFTER_CLAUSE.findall(text)
    return {
        "sentence_spaces": sum(1 for g in sentence if g),
        "sentence_ends": len(sentence),
        "clause_spaces": sum(1 for g in clause if g),
        "clause_ends": len(clause),
        "word_spaces": len(BETWEEN_WORDS.findall(text)),
    }


def noncanonical(ids: list[int], tok: object) -> int:
    """Generated tokens that re-tokenizing the decoded text would not give."""
    text = tok.decode(ids, skip_special_tokens=False)  # type: ignore[attr-defined]
    canonical = tok.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
    matcher = difflib.SequenceMatcher(a=canonical, b=ids, autojunk=False)
    return len(ids) - sum(block.size for block in matcher.get_matching_blocks())


def measure(text: str, answer: str | None) -> dict[str, object]:
    """Gluing and shortening in the reasoning and in the answer, separately.

    Steering acts on both. The reasoning is what the question is about, but at
    reasoning effort "low" it is short, so the answer holds most of the words
    a short form could replace.
    """
    analysis_m = ANALYSIS.search(text)
    final_m = FINAL.search(text)
    parts = {
        "reasoning": analysis_m.group(1) if analysis_m else "",
        "answer": final_m.group(1) if final_m else "",
    }
    #: Truncated at the token limit if the final turn never closed.
    finished = final_m is not None and final_m.group(2) != ""
    out: dict[str, object] = {"finished": finished, "degenerate": degenerate(text)}
    for name, part in parts.items():
        words = [w.lower() for w in WORD.findall(CODE.sub(" ", part))]
        out[f"{name}_letters"] = sum(map(str.isalpha, part))
        out[f"{name}_letters_per_space"] = letters_per_space(part)
        out[f"{name}_short"] = sum(w in SHORT for w in words)
        out[f"{name}_long"] = sum(w in LONG for w in words)
    out |= spacing(parts["reasoning"])
    if answer is not None:
        out["correct"] = finished and float(answer) in stated_numbers(parts["answer"])
    return out


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """One steering run: medians over answers, counts pooled."""
    out: dict[str, object] = {}
    for part in ("reasoning", "answer"):
        lps = [r[f"{part}_letters_per_space"] for r in rows]
        lps = [x for x in lps if x is not None]
        n_short = sum(r[f"{part}_short"] for r in rows)  # type: ignore[misc]
        n_long = sum(r[f"{part}_long"] for r in rows)  # type: ignore[misc]
        out[f"{part}_letters_per_space"] = statistics.median(lps) if lps else None  # type: ignore[arg-type]
        out[f"{part}_letters"] = statistics.median(r[f"{part}_letters"] for r in rows)  # type: ignore[misc]
        out[f"{part}_short_long"] = [n_short, n_long]
        out[f"{part}_short_share"] = (
            n_short / (n_short + n_long) if n_short + n_long >= MIN_LIST_WORDS else None
        )
    for key in (
        "sentence_spaces",
        "sentence_ends",
        "clause_spaces",
        "clause_ends",
        "word_spaces",
        "reasoning_letters",
        "degenerate",
    ):
        out[f"total_{key}"] = sum(int(r[key]) for r in rows)  # type: ignore[call-overload]
    if all("noncanonical" in r for r in rows):
        out["noncanonical_tokens"] = sum(int(r["noncanonical"]) for r in rows)  # type: ignore[call-overload]
        out["tokens"] = sum(int(r["tokens"]) for r in rows)  # type: ignore[call-overload]
        out["noncanonical_answers"] = sum(bool(r["noncanonical"]) for r in rows)
    math = [r for r in rows if "correct" in r]
    out["finished"] = statistics.mean(bool(r["finished"]) for r in rows)
    out["accuracy"] = statistics.mean(bool(r["correct"]) for r in math)
    return out


def coherent(run: dict[str, object], base: dict[str, object]) -> bool:
    """Still doing the task: no answer looping or broken (unless unsteered
    gpt-oss does too), and neither GSM8K accuracy nor the share of answers
    finished within the token limit more than 10 points below unsteered.
    Unsteered gpt-oss itself runs out of tokens on 2 of 40 answers, both long
    tables.
    """
    return (
        float(run["accuracy"]) >= float(base["accuracy"]) - COHERENT_DROP  # type: ignore[arg-type]
        and float(run["finished"]) >= float(base["finished"]) - COHERENT_DROP  # type: ignore[arg-type]
        and int(run["total_degenerate"]) <= int(base["total_degenerate"])  # type: ignore[call-overload]
    )


def score(paths: list[Path], out: Path) -> None:
    from transformers import AutoTokenizer

    from cot_spacing.gpt_oss import MODEL

    tok = AutoTokenizer.from_pretrained(MODEL)
    answers = {p["id"]: p.get("answer") for p in load_prompts()}
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["scale"] == 0:
                r |= {"direction": "none", "layer": 0}
            keys = ("direction", "layer", "scale", "norm", "id")
            row = {**{k: r[k] for k in keys}, **measure(r["text"], answers[r["id"]])}
            if "ids" in r:
                row |= {
                    "noncanonical": noncanonical(r["ids"], tok),
                    "tokens": len(r["ids"]),
                }
            rows.append(row)
    groups: dict[tuple[str, int, float], list[dict[str, object]]] = {}
    for r in rows:
        groups.setdefault((r["direction"], r["layer"], r["scale"]), []).append(r)  # type: ignore[arg-type]
    summary = [
        {"direction": d, "layer": layer, "scale": scale, **summarize(rs)}
        for (d, layer, scale), rs in sorted(groups.items())
    ]
    base = next(s for s in summary if s["direction"] == "none")
    for s in summary:
        s["coherent"] = coherent(s, base)
    out.write_text(json.dumps({"summary": summary}, indent=1) + "\n")
    for s in summary:
        print(s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vectors")
    v.add_argument(
        "--examples", type=Path, default=EXP_DIR / "results" / "variants.json"
    )
    v.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--vectors", type=Path, required=True)
    g.add_argument("--direction", choices=DIRECTIONS, required=True)
    g.add_argument("--layer", type=int, required=True)
    g.add_argument("--scale", type=float, required=True)
    g.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("paths", type=Path, nargs="+")
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "vectors":
        vectors(args.examples, args.out)
    elif args.cmd == "generate":
        generate(args.vectors, args.direction, args.layer, args.scale, args.out)
    else:
        score(args.paths, args.out)


if __name__ == "__main__":
    main()
