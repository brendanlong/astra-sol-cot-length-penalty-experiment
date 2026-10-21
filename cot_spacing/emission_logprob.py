"""Which tokenization of the glued text would a BPE language model emit?

The length analysis in ``analysis.py`` says what each emission model would
cost. It cannot say which one GPT-6 used, because its token IDs are not
observable. What is testable is the general claim the hypothesis rests on: for
text with the spaces removed, is the word-at-a-time token sequence more
probable than the canonical one?

Both sequences decode to exactly the same characters, so comparing their total
log-probability is a well-posed question about the same string. Spaced text is
scored as a control, where the canonical sequence should win because that is
what the model was trained on.

Runs on CPU in a couple of minutes per model; these are a few hundred tokens.

Usage:
    uv run python -m cot_spacing.emission_logprob \
        --models Qwen/Qwen2.5-1.5B-Instruct HuggingFaceTB/SmolLM2-1.7B
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from cot_spacing.data import load_excerpts, model_outputs
from cot_spacing.reference import reference_spaced
from cot_spacing.segment import segment

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from cot_spacing.segment import Segmentation

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B",
)


class Divergence(BaseModel):
    """One point where the two emission models choose a different next token.

    Both candidates follow the same prefix, so their one-step probabilities are
    directly comparable: this is the decision the model would actually face at
    decode time, without the downstream effects that a whole-sequence
    comparison folds in.
    """

    excerpt_id: str
    model: str
    text_kind: str
    prefix_tail: str
    canonical_token: str
    wordwise_token: str
    logprob_canonical: float
    logprob_wordwise: float
    advantage_wordwise: float
    #: Rank of each candidate in the next-token distribution, 1-indexed, so
    #: rank 1 is the model's first choice and the value reads as an ordinal.
    rank_canonical: int
    rank_wordwise: int


class EmissionScore(BaseModel):
    excerpt_id: str
    model: str
    text_kind: str  # "glued" or "spaced"
    n_tokens_canonical: int
    n_tokens_wordwise: int
    logprob_canonical: float
    logprob_wordwise: float
    #: Positive means word-at-a-time is the more probable way to emit this text.
    advantage_wordwise: float
    identical: bool


class Tokenization(BaseModel):
    """Token ids plus the character offset each token ends at.

    Offsets make the two tokenizations comparable: both are partitions of the
    same string, so a shared offset is a point where they agree about where one
    token stops and the next starts.
    """

    ids: list[int]
    ends: list[int]


def _decode(tok: PreTrainedTokenizerBase, ids: list[int]) -> str:
    """``decode`` is typed as possibly returning a list; here it never does."""
    out = tok.decode(ids)
    assert isinstance(out, str)
    return out


def canonical_ids(text: str, tok: PreTrainedTokenizerBase) -> Tokenization:
    ids: list[int] = []
    ends: list[int] = []
    offset = 0
    for piece in tok(text, add_special_tokens=False)["input_ids"]:  # type: ignore[union-attr]
        ids.append(int(piece))
        offset += len(_decode(tok, [piece]))
        ends.append(offset)
    if ends and ends[-1] != len(text):
        raise ValueError(f"offsets do not cover {text!r}")
    return Tokenization(ids=ids, ends=ends)


def wordwise_ids(
    seg: Segmentation, tok: PreTrainedTokenizerBase, *, spaced: bool
) -> Tokenization:
    """Each word encoded on its own, carrying whatever gap preceded it."""
    gaps = [(g if g == "\n" else " ") if spaced else g for g in seg.gaps]
    ids: list[int] = []
    ends: list[int] = []
    offset = 0
    for prefix, word in zip(["", *gaps], seg.words, strict=True):
        piece = canonical_ids(prefix + word, tok)
        ids.extend(piece.ids)
        ends.extend(e + offset for e in piece.ends)
        offset += len(prefix + word)
    return Tokenization(ids=ids, ends=ends)


@torch.no_grad()
def sequence_logprob(ids: list[int], model: PreTrainedModel, bos: int | None) -> float:
    """Total log P of the sequence, teacher-forced, conditioned on BOS only.

    The first token is scored too when the tokenizer has a BOS, so the two
    candidate sequences are compared over the whole string rather than letting
    one of them skip its opening token.
    """
    prefix = [] if bos is None else [bos]
    full = torch.tensor([prefix + ids])
    logits = model(full).logits.float()
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = full[0, 1:]
    scored = logprobs.gather(-1, targets[:, None])[:, 0]
    return float(scored.sum())


def divergences(a: Tokenization, b: Tokenization) -> list[tuple[int, int, int]]:
    """Points where two tokenizations of the same text choose a different token.

    Both are partitions of the same string. Between two offsets where they
    agree, either they pick the same tokens or they diverge; where they
    diverge, the first token of the segment is the choice being made. Returns
    (shared prefix length in tokens of ``a``, next token of ``a``, next token
    of ``b``) per divergence.
    """
    agreed = sorted({0, *a.ends} & {0, *b.ends})
    out: list[tuple[int, int, int]] = []
    for start, stop in itertools.pairwise(agreed):
        lo_a = 0 if start == 0 else a.ends.index(start) + 1
        lo_b = 0 if start == 0 else b.ends.index(start) + 1
        hi_a = a.ends.index(stop) + 1
        hi_b = b.ends.index(stop) + 1
        if a.ids[lo_a:hi_a] == b.ids[lo_b:hi_b]:
            continue
        out.append((lo_a, a.ids[lo_a], b.ids[lo_b]))
    return out


@torch.no_grad()
def score_divergences(
    canonical: Tokenization,
    wordwise: Tokenization,
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    bos: int | None,
    meta: tuple[str, str, str],
) -> list[Divergence]:
    points = divergences(canonical, wordwise)
    if not points:
        return []
    excerpt_id, model_name, kind = meta
    prefix0 = [] if bos is None else [bos]
    full = torch.tensor([prefix0 + canonical.ids])
    logits = model(full).logits.float()
    logprobs = torch.log_softmax(logits[0], dim=-1)
    out: list[Divergence] = []
    for n_prefix, tok_c, tok_w in points:
        # Next-token distribution after the shared prefix.
        row = logprobs[len(prefix0) + n_prefix - 1]
        order = torch.argsort(row, descending=True)
        ranks = torch.empty_like(order)
        # 1-indexed: these are reported as ordinals ("ranked 35th"), and a
        # 0-indexed rank printed that way is off by one.
        ranks[order] = torch.arange(1, len(order) + 1)
        out.append(
            Divergence(
                excerpt_id=excerpt_id,
                model=model_name,
                text_kind=kind,
                prefix_tail=_decode(
                    tok, canonical.ids[max(0, n_prefix - 4) : n_prefix]
                ),
                canonical_token=_decode(tok, [tok_c]),
                wordwise_token=_decode(tok, [tok_w]),
                logprob_canonical=float(row[tok_c]),
                logprob_wordwise=float(row[tok_w]),
                advantage_wordwise=float(row[tok_w] - row[tok_c]),
                rank_canonical=int(ranks[tok_c]),
                rank_wordwise=int(ranks[tok_w]),
            )
        )
    return out


def score_model(
    name: str, dtype: torch.dtype = torch.float32
) -> tuple[list[EmissionScore], list[Divergence]]:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    out: list[EmissionScore] = []
    points: list[Divergence] = []
    for e in model_outputs(load_excerpts()):
        spaced = reference_spaced(e)
        if spaced is None:
            continue
        seg = segment(e.text(), spaced)
        if not seg.dropped:
            continue
        for kind, text, use_spaced in (
            ("glued", seg.original(), False),
            ("spaced", seg.spaced(), True),
        ):
            canon = canonical_ids(text, tok)
            wordwise = wordwise_ids(seg, tok, spaced=use_spaced)
            assert _decode(tok, canon.ids) == _decode(tok, wordwise.ids) == text, e.id
            lp_canon = sequence_logprob(canon.ids, model, bos)
            lp_word = sequence_logprob(wordwise.ids, model, bos)
            points.extend(
                score_divergences(canon, wordwise, tok, model, bos, (e.id, name, kind))
            )
            out.append(
                EmissionScore(
                    excerpt_id=e.id,
                    model=name,
                    text_kind=kind,
                    n_tokens_canonical=len(canon.ids),
                    n_tokens_wordwise=len(wordwise.ids),
                    logprob_canonical=lp_canon,
                    logprob_wordwise=lp_word,
                    advantage_wordwise=lp_word - lp_canon,
                    identical=canon.ids == wordwise.ids,
                )
            )
            print(
                f"  {e.id:<44} {kind:<6} canonical {len(canon.ids):>4} tok "
                f"{lp_canon:>10.1f}   word-at-a-time {len(wordwise.ids):>4} tok "
                f"{lp_word:>10.1f}   advantage {lp_word - lp_canon:>+8.1f}"
            )
    return out, points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores: list[EmissionScore] = []
    points: list[Divergence] = []
    for name in args.models:
        print(f"\n{name}")
        model_scores, model_points = score_model(name)
        scores.extend(model_scores)
        points.extend(model_points)

    (args.out_dir / "emission_logprob.json").write_text(
        json.dumps(
            {
                "sequences": [s.model_dump() for s in scores],
                "divergences": [p.model_dump() for p in points],
            },
            indent=1,
        )
        + "\n"
    )

    print(
        "\nmean advantage of word-at-a-time emission (nats, positive = more probable):"
    )
    for name in args.models:
        for kind in ("glued", "spaced"):
            rows = [s for s in scores if s.model == name and s.text_kind == kind]
            if not rows:
                continue
            wins = sum(s.advantage_wordwise > 0 for s in rows)
            print(
                f"  {name:<32} {kind:<6} "
                f"{statistics.mean(s.advantage_wordwise for s in rows):>+8.1f}  "
                f"({wins}/{len(rows)} excerpts favour word-at-a-time)"
            )

    print("\nper-divergence, one-step (same prefix, two candidate next tokens):")
    for name in args.models:
        for kind in ("glued", "spaced"):
            rows = [p for p in points if p.model == name and p.text_kind == kind]
            if not rows:
                continue
            wins = sum(p.advantage_wordwise > 0 for p in rows)
            print(
                f"  {name:<32} {kind:<6} n={len(rows):<4} mean advantage "
                f"{statistics.mean(p.advantage_wordwise for p in rows):>+6.2f}  "
                f"word-at-a-time more likely at {wins}/{len(rows)}; median rank "
                f"canonical {statistics.median(p.rank_canonical for p in rows):.0f} "
                f"vs word-at-a-time "
                f"{statistics.median(p.rank_wordwise for p in rows):.0f}"
            )


if __name__ == "__main__":
    main()
