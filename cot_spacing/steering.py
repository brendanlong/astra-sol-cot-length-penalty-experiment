"""Steering gpt-oss toward gluing words, shortening them, or both.

``vectors``: each direction is a difference of means of the residual stream
over the reasoning tokens, a variant minus the spaced reading of the same
excerpts (see :mod:`cot_spacing.variants`). Held-out separation is checked by
leaving each excerpt out in turn.

``generate``: adds one direction at one layer while gpt-oss answers held-out
prompts, at one strength. Every direction is scaled to the same length,
a fraction of the typical residual norm at that layer, so a direction that
needs less push to produce its behaviour is the easier one to steer.

``score``: how much each generation glues and shortens, and whether it still
gets the arithmetic right.

Usage:
    python -m cot_spacing.steering vectors --out results/steering/vectors.pt
    python -m cot_spacing.steering generate --vectors ... --direction glued \
        --layer 12 --scale 0.25 --out results/steering/gen/glued-12-0.25.jsonl
    python -m cot_spacing.steering score results/steering/gen/*.jsonl --out ...
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import tiktoken
import torch

from cot_spacing.variants import SHORT_FORMS, Example

EXP_DIR = Path(__file__).parent
DIRECTIONS = ("glued", "short", "both")
PROMPT_FILES = (
    EXP_DIR / "corpus" / "gsm8k_test_first32.json",
    EXP_DIR / "corpus" / "agentic_prompts.json",
)
MAX_NEW_TOKENS = 1024


def load_examples(path: Path) -> list[Example]:
    return [Example.model_validate(d) for d in json.loads(path.read_text())]


@torch.no_grad()
def reasoning_states(tok: object, model: object, user: str, cot: str) -> torch.Tensor:
    """Residual stream after every layer at each reasoning token: [layer, tok, d]."""
    from cot_spacing.gpt_oss import with_reasoning

    ids, span = with_reasoning(tok, user, cot)  # type: ignore[arg-type]
    out = model(  # type: ignore[operator]
        torch.tensor([ids], device="cuda"), output_hidden_states=True
    )
    return torch.stack([h[0, span] for h in out.hidden_states[1:]]).float().cpu()


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Probability a random ``pos`` scores above a random ``neg``."""
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean()).item()


def vectors(examples_path: Path, out: Path) -> None:
    from cot_spacing.gpt_oss import load

    tok, model = load()
    examples = load_examples(examples_path)
    states = {
        (e.id, v): reasoning_states(tok, model, e.user, text)
        for e in examples
        for v, text in e.texts.items()
    }

    def mean(variant: str, ids: list[str]) -> torch.Tensor:
        return torch.cat([states[i, variant] for i in ids], dim=1).mean(dim=1)

    ids = [e.id for e in examples]
    directions = {d: mean(d, ids) - mean("spaced", ids) for d in DIRECTIONS}
    resid_norm = torch.cat([states[i, "spaced"] for i in ids], dim=1).norm(dim=-1)
    n_layers = resid_norm.shape[0]

    held_out: dict[str, list[list[float]]] = {d: [] for d in DIRECTIONS}
    for d in DIRECTIONS:
        for held in ids:
            rest = [i for i in ids if i != held]
            v = mean(d, rest) - mean("spaced", rest)
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
            text = str(tok.decode(row, skip_special_tokens=False))
            record = {
                "direction": direction,
                "layer": layer,
                "scale": scale,
                "norm": scale * resid,
                "id": p["id"],
                "text": text.replace(str(tok.pad_token), ""),
            }
            f.write(json.dumps(record) + "\n")
    tmp.rename(out)


ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)
FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S
)
LETTERS = re.compile(r"[A-Za-z]")
WORD = re.compile(r"(?<![\w/])[A-Za-z]+(?:/o|/)?(?![\w])")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
LONG = set(SHORT_FORMS)
SHORT = set(SHORT_FORMS.values()) - LONG


def measure(text: str, answer: str | None, enc: tiktoken.Encoding) -> dict[str, object]:
    analysis_m = ANALYSIS.search(text)
    final_m = FINAL.search(text)
    analysis = analysis_m.group(1) if analysis_m else ""
    final = final_m.group(1) if final_m else None
    chunks = [c for c in analysis.split() if LETTERS.search(c)]
    words = [w.lower() for w in WORD.findall(analysis)]
    n_long = sum(w in LONG for w in words)
    n_short = sum(w in SHORT for w in words)
    correct = None
    if answer is not None:
        numbers = NUMBER.findall(final or "")
        correct = bool(numbers) and float(numbers[-1].replace(",", "")) == float(answer)
    return {
        "letters": sum(map(str.isalpha, analysis)),
        "letters_per_space": sum(map(str.isalpha, analysis)) / max(1, len(chunks)),
        "tokens_per_chunk": (
            statistics.mean(len(enc.encode(" " + c)) for c in chunks)
            if chunks
            else None
        ),
        "n_long": n_long,
        "n_short": n_short,
        "finished": final is not None,
        "correct": correct,
    }


def score(paths: list[Path], out: Path) -> None:
    enc = tiktoken.get_encoding("o200k_base")
    answers = {p["id"]: p.get("answer") for p in load_prompts()}
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["scale"] == 0:
                r |= {"direction": "none", "layer": 0}
            rows.append(
                {
                    **{k: r[k] for k in ("direction", "layer", "scale", "norm", "id")},
                    **measure(r["text"], answers[r["id"]], enc),
                }
            )
    groups: dict[tuple[str, int, float], list[dict[str, object]]] = {}
    for r in rows:
        groups.setdefault((r["direction"], r["layer"], r["scale"]), []).append(r)  # type: ignore[arg-type]
    summary = []
    for (direction, layer, scale), rs in sorted(groups.items()):
        math = [r for r in rs if r["correct"] is not None]
        letters = sum(r["letters"] for r in rs)  # type: ignore[misc]
        chunks = sum(r["letters"] / r["letters_per_space"] for r in rs if r["letters"])  # type: ignore[operator]
        n_long = sum(r["n_long"] for r in rs)  # type: ignore[misc]
        n_short = sum(r["n_short"] for r in rs)  # type: ignore[misc]
        tpc = [r["tokens_per_chunk"] for r in rs if r["tokens_per_chunk"] is not None]
        summary.append(
            {
                "direction": direction,
                "layer": layer,
                "scale": scale,
                "letters_per_space": letters / chunks if chunks else None,
                "tokens_per_chunk": statistics.mean(tpc) if tpc else None,  # type: ignore[arg-type]
                "short_share": n_short / (n_long + n_short)
                if n_long + n_short
                else None,
                "n_short_long": [n_short, n_long],
                "analysis_letters": letters / len(rs),
                "finished": statistics.mean(bool(r["finished"]) for r in rs),
                "accuracy": statistics.mean(bool(r["correct"]) for r in math),
            }
        )
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
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
