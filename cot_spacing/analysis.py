"""Length and complexity analysis of the glued CoT excerpts.

Usage:
    uv run python -m cot_spacing.analysis
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tiktoken
from scipy.stats import fisher_exact

from cot_spacing.chunking import (
    chunk_pattern,
    integer_chunk_patterns,
    invisible_channel_capacity,
    neighbour_disagreement,
    words_in_context,
)
from cot_spacing.data import EXP_DIR, Excerpt, load_excerpts, model_outputs
from cot_spacing.reference import reference_spaced
from cot_spacing.segment import Segmentation, segment
from cot_spacing.tokenize_cost import (
    ENCODINGS,
    PRIMARY_ENCODING,
    boundaries,
    count,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cot_spacing.tokenize_cost import Boundary

RESULTS_DIR = EXP_DIR / "results"


Segmented = list[tuple[Excerpt, Segmentation]]


def segmentations(excerpts: list[Excerpt]) -> Segmented:
    out: Segmented = []
    for e in model_outputs(excerpts):
        spaced = reference_spaced(e)
        if spaced is None:
            continue
        out.append((e, segment(e.text(), spaced)))
    return out


def glued_only(segs: Segmented) -> Segmented:
    return [(e, s) for e, s in segs if s.dropped]


def length_table(
    segs: list[tuple[Excerpt, Segmentation]], encoding: str
) -> dict[str, Any]:
    enc = tiktoken.get_encoding(encoding)
    rows: list[dict[str, Any]] = []
    for e, seg in segs:
        c = count(seg, enc)
        rows.append(
            {
                "id": e.id,
                "model": e.model,
                "speaker": e.speaker,
                "kind": e.kind,
                "words": len(seg.words),
                "boundaries": len(seg.gaps),
                "dropped": len(seg.dropped),
                **c.model_dump(),
            }
        )
    counts: dict[str, int] = {
        k: sum(int(r[k]) for r in rows)
        for k in (
            "words",
            "dropped",
            "chars_glued",
            "chars_spaced",
            "canonical_spaced",
            "canonical_glued",
            "wordwise_glued",
            "wordwise_spaced",
        )
    }
    per_excerpt_pct = [
        100 * (int(r["canonical_glued"]) / int(r["canonical_spaced"]) - 1) for r in rows
    ]
    base = counts["canonical_spaced"]
    totals: dict[str, float] = dict(counts)
    totals["pct_chars_vs_spaced"] = 100 * (
        counts["chars_glued"] / counts["chars_spaced"] - 1
    )
    totals["pct_canonical_glued_vs_spaced"] = 100 * (
        counts["canonical_glued"] / base - 1
    )
    # Against the matching baseline: word-at-a-time emission of the spaced
    # text. Dividing by canonical_spaced instead would fold the ordinary
    # overhead of word-at-a-time emission into the apparent cost of gluing.
    totals["pct_wordwise_glued_vs_wordwise_spaced"] = 100 * (
        counts["wordwise_glued"] / counts["wordwise_spaced"] - 1
    )
    totals["pct_wordwise_spaced_vs_canonical_spaced"] = 100 * (
        counts["wordwise_spaced"] / base - 1
    )
    totals["pct_retokenization_inflation"] = 100 * (
        counts["canonical_glued"] / counts["wordwise_glued"] - 1
    )
    totals["per_excerpt_pct_canonical_min"] = min(per_excerpt_pct, default=0.0)
    totals["per_excerpt_pct_canonical_max"] = max(per_excerpt_pct, default=0.0)
    totals["per_excerpt_pct_canonical_mean"] = (
        statistics.mean(per_excerpt_pct) if per_excerpt_pct else 0.0
    )
    totals["n_excerpts_canonical_positive"] = sum(p > 0 for p in per_excerpt_pct)
    return {"encoding": encoding, "rows": rows, "totals": totals}


def permutation_test_delta(
    segs: Segmented, encoding: str, seed: int = 0, n_perms: int = 20000
) -> dict[str, Any]:
    """Are glued boundaries cheaper in tokens than kept ones, beyond chance?

    The glued/kept labels are shuffled *within each excerpt*, so an excerpt
    that glues a lot and happens to be cheap overall cannot produce the effect
    on its own. Reported for mid-sentence boundaries, since sentence
    boundaries are never glued and are where the savings sit.
    """
    enc = tiktoken.get_encoding(encoding)
    per_excerpt = [
        [b for b in boundaries(e.id, seg, enc) if not b.left_ends_sentence]
        for e, seg in segs
    ]
    per_excerpt = [bs for bs in per_excerpt if any(b.dropped for b in bs)]

    def gap(assignment: list[list[bool]]) -> float:
        glued = [
            b.canonical_delta
            for bs, labels in zip(per_excerpt, assignment, strict=True)
            for b, lab in zip(bs, labels, strict=True)
            if lab
        ]
        kept = [
            b.canonical_delta
            for bs, labels in zip(per_excerpt, assignment, strict=True)
            for b, lab in zip(bs, labels, strict=True)
            if not lab
        ]
        return statistics.mean(glued) - statistics.mean(kept)

    observed_labels = [[b.dropped for b in bs] for bs in per_excerpt]
    observed = gap(observed_labels)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(n_perms):
        shuffled = []
        for labels in observed_labels:
            copy = list(labels)
            rng.shuffle(copy)
            shuffled.append(copy)
        if gap(shuffled) <= observed:
            at_least += 1
    return {
        "statistic": "mean canonical_delta (glued) - mean canonical_delta (kept)",
        "observed": observed,
        "n_permutations": n_perms,
        "p_one_sided_glued_cheaper": at_least / n_perms,
    }


def boundary_analysis(segs: Segmented, encoding: str) -> dict[str, Any]:
    enc = tiktoken.get_encoding(encoding)
    bs = [b for e, seg in segs for b in boundaries(e.id, seg, enc)]
    dropped = [b for b in bs if b.dropped]
    kept = [b for b in bs if not b.dropped]

    # Sentence boundaries are never glued and are also where the tokenizer
    # happens to reward gluing, so every test below is also run on mid-sentence
    # boundaries only.
    mid = [b for b in bs if not b.left_ends_sentence]

    def free_vs_costly(group_: list[Boundary]) -> dict[str, Any]:
        """Drop rate where gluing is token-free against where it costs tokens.

        Restricting "token-aware" to outright *savings* was too narrow:
        declining the joins that cost a token, while taking the ones that cost
        nothing, is the same kind of selectivity and there is far more of it to
        observe.
        """
        free = [b for b in group_ if b.canonical_delta <= 0]
        costly = [b for b in group_ if b.canonical_delta > 0]
        table = [
            [sum(b.dropped for b in free), len(free) - sum(b.dropped for b in free)],
            [
                sum(b.dropped for b in costly),
                len(costly) - sum(b.dropped for b in costly),
            ],
        ]
        fisher = fisher_exact(table)
        return {
            "n_free": len(free),
            "drop_rate_free": sum(b.dropped for b in free) / len(free),
            "n_costly": len(costly),
            "drop_rate_costly": sum(b.dropped for b in costly) / len(costly),
            "fisher_p": float(fisher.pvalue),  # type: ignore[attr-defined]
        }

    def group(bs_: list[Boundary]) -> dict[str, Any]:
        return {
            "n": len(bs_),
            "mean_canonical_delta": statistics.mean(b.canonical_delta for b in bs_)
            if bs_
            else 0.0,
            "mean_wordwise_delta": statistics.mean(b.wordwise_delta for b in bs_)
            if bs_
            else 0.0,
            "canonical_delta_hist": dict(
                sorted(collections.Counter(b.canonical_delta for b in bs_).items())
            ),
        }

    def drop_rate(pred: Callable[[Boundary], bool]) -> dict[str, Any]:
        yes = [b for b in bs if pred(b)]
        no = [b for b in bs if not pred(b)]
        return {
            "n_true": len(yes),
            "drop_rate_true": sum(b.dropped for b in yes) / len(yes) if yes else None,
            "n_false": len(no),
            "drop_rate_false": sum(b.dropped for b in no) / len(no) if no else None,
        }

    return {
        "encoding": encoding,
        "n_boundaries": len(bs),
        "n_dropped": len(dropped),
        "dropped": group(dropped),
        "kept": group(kept),
        "dropped_mid_sentence": group([b for b in mid if b.dropped]),
        "kept_mid_sentence": group([b for b in mid if not b.dropped]),
        "free_vs_costly": free_vs_costly(bs),
        "free_vs_costly_mid_sentence": free_vs_costly(mid),
        "permutation_test_mean_delta": permutation_test_delta(segs, encoding),
        "token_saving_boundaries": {
            "n": sum(b.canonical_delta < 0 for b in bs),
            "n_glued": sum(b.canonical_delta < 0 and b.dropped for b in bs),
            "examples": [
                f"{b.left} | {b.right}" + ("" if b.dropped else " (kept)")
                for b in bs
                if b.canonical_delta < 0
            ],
            "n_mid_sentence": sum(b.canonical_delta < 0 for b in mid),
        },
        "straddled_dropped_boundaries": sum(b.straddled for b in dropped),
        "predictors": {
            "left_ends_sentence": drop_rate(lambda b: b.left_ends_sentence),
            "right_is_identifier": drop_rate(lambda b: b.right_is_identifier),
            "left_len_le_3": drop_rate(lambda b: len(b.left) <= 3),
            "left_len_ge_8": drop_rate(lambda b: len(b.left) >= 8),
            "right_starts_upper": drop_rate(lambda b: b.right[:1].isupper()),
            "right_has_digit": drop_rate(lambda b: any(c.isdigit() for c in b.right)),
        },
    }


def run_structure(segs: Segmented, seed: int = 0, n_sims: int = 2000) -> dict[str, Any]:
    """Are glued runs distributed as independent per-boundary dropping predicts?

    A per-space penalty makes gluing worth the same wherever it happens, so it
    predicts independent drops. Clustering into long runs would point at
    something else.

    The *mean* run length carries no information: it equals
    (words + excerpts) / (kept boundaries + excerpts), so once the null matches
    the observed drop rate it reproduces the observed mean exactly. Only the
    spread is testable, so the statistic here is the variance of run lengths,
    with the null simulated per excerpt so run counts match.
    """
    runs = [len(r) for _, seg in segs for r in seg.runs()]
    gaps = [g for _, seg in segs for g in seg.gaps]
    p = sum(g == "" for g in gaps) / len(gaps)
    rng = random.Random(seed)
    means: list[float] = []
    variances: list[float] = []
    for _ in range(n_sims):
        sim: list[int] = []
        for _, seg in segs:
            cur = 1
            for _ in seg.gaps:
                if rng.random() < p:
                    cur += 1
                else:
                    sim.append(cur)
                    cur = 1
            sim.append(cur)
        means.append(statistics.mean(sim))
        variances.append(statistics.variance(sim))
    variances.sort()
    observed_variance = statistics.variance(runs)
    return {
        "run_length_hist": dict(sorted(collections.Counter(runs).items())),
        "mean_run_length": statistics.mean(runs),
        "drop_rate": p,
        # words / runs, with words = boundaries + excerpts and
        # runs = kept boundaries + excerpts.
        "analytic_null_mean_run": (len(gaps) + len(segs))
        / (sum(1 for g in gaps if g != "") + len(segs)),
        "bernoulli_null_mean_run": statistics.mean(means),
        "variance_run_length": observed_variance,
        "bernoulli_null_variance": statistics.mean(variances),
        "bernoulli_null_variance_ci95": [
            variances[n_sims // 40],
            variances[-n_sims // 40],
        ],
        "p_null_variance_at_least_observed": sum(
            v >= observed_variance for v in variances
        )
        / n_sims,
    }


def sweep_stats(
    rights: list[str],
    lefts: list[str],
    enc: tiktoken.Encoding,
    prefix_pad: str = "",
) -> dict[str, Any]:
    """How much a right item's chunking depends on what is glued before it.

    Reported under both pattern conventions (see
    :func:`~cot_spacing.chunking.chunk_pattern`). ``clipped``
    counts only the bytes of the right item, which is what the integer
    baseline measures and therefore the number to compare across the two.
    """
    out: dict[str, Any] = {"n_right": len(rights), "n_left": len(lefts)}
    for label, clip in (("unclipped", False), ("clipped", True)):
        distinct: list[int] = []
        differs = 0
        for r in rights:
            alone = chunk_pattern(r, enc, clip=clip)
            pats = [
                chunk_pattern(r, enc, prefix=prefix_pad + left, clip=clip)
                for left in lefts
            ]
            distinct.append(len({*pats}))
            differs += sum(p != alone for p in pats)
        out[label] = {
            "mean_distinct_patterns_per_item": statistics.mean(distinct),
            "max_distinct_patterns": max(distinct),
            "distinct_pattern_hist": dict(
                sorted(collections.Counter(distinct).items())
            ),
            "frac_pairs_differing_from_standalone": differs
            / (len(rights) * len(lefts)),
        }
    return out


def chunking_analysis(segs: Segmented, encoding: str) -> dict[str, Any]:
    enc = tiktoken.get_encoding(encoding)
    words = [w for e, seg in segs for w in words_in_context(e.id, seg, enc)]
    glued = [w for w in words if w.glued_left]
    mismatched = [w for w in glued if not w.matches_standalone]

    rights = sorted({w.word for w in glued if w.word.isalpha() and len(w.word) > 2})
    lefts = sorted({w.word for w in words if w.word.isalpha() and len(w.word) > 1})

    gpt2 = tiktoken.get_encoding("gpt2")
    integers = {
        name: integer_chunk_patterns(gpt2, lo, hi)
        for name, lo, hi in (("3_digit", 100, 1000), ("4_digit", 1000, 10000))
    }

    return {
        "encoding": encoding,
        "n_glued_words": len(glued),
        "n_chunked_as_standalone": sum(w.matches_standalone for w in glued),
        "frac_chunked_as_standalone": sum(w.matches_standalone for w in glued)
        / len(glued),
        "mismatches": [
            {"word": w.word, "in_context": w.in_context, "standalone": w.standalone}
            for w in mismatched
        ],
        "context_sweep": sweep_stats(rights, lefts, enc),
        "context_sweep_spaced_prefix": sweep_stats(rights, lefts, enc, prefix_pad=" "),
        "gpt2_integer_context_sweep": {
            name: sweep_stats(
                [str(n) for n in random.Random(0).sample(range(lo, hi), len(rights))],
                [str(n) for n in random.Random(1).sample(range(lo, hi), len(lefts))],
                gpt2,
            )
            for name, lo, hi in (("3_digit", 100, 1000), ("4_digit", 1000, 10000))
        },
        "gpt2_integer_baseline": {
            name: {
                "neighbour_disagreement": neighbour_disagreement(pats),
                "n_distinct_patterns": len({*pats.values()}),
                "n_single_token": sum(1 for v in pats.values() if len(v) == 1),
                "pattern_hist": {
                    "-".join(map(str, k)): v
                    for k, v in collections.Counter(pats.values()).most_common()
                },
            }
            for name, pats in integers.items()
        },
    }


def gluing_by_kind(excerpts: list[Excerpt]) -> dict[str, Any]:
    """Where the glued register shows up: hidden reasoning vs readable text."""
    out: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"excerpts": 0, "glued_excerpts": 0}
    )
    for e in model_outputs(excerpts):
        spaced = reference_spaced(e)
        key = f"{e.model} / {e.kind}"
        out[key]["excerpts"] += 1
        if spaced is not None and spaced != e.text():
            out[key]["glued_excerpts"] += 1
    return dict(out)


def segmentation_sensitivity(segs: Segmented, encoding: str) -> dict[str, Any]:
    """How much the headline moves under the most contestable reading.

    The reference segmentation is OpenAI's own, and its least certain calls are
    the digit labels: it renders "severity4?" as two words while "L2", "P0" and
    "L1" stay single words. Merging every digit-label boundary is the
    worst-case disagreement a reader could reasonably insist on.
    """
    enc = tiktoken.get_encoding(encoding)
    merged: Segmented = []
    n_merged = 0
    for e, seg in segs:
        indices = {
            i
            for i in range(len(seg.gaps))
            if any(c.isdigit() for c in seg.words[i + 1]) and seg.words[i].isalpha()
        }
        n_merged += len(indices)
        merged.append((e, seg.merge(indices) if indices else seg))
    totals = length_table(merged, encoding)["totals"]
    base = length_table(segs, encoding)["totals"]
    del enc
    return {
        "n_boundaries_merged": n_merged,
        "as_reported": {
            k: base[k]
            for k in (
                "pct_chars_vs_spaced",
                "pct_canonical_glued_vs_spaced",
            )
        },
        "digit_labels_merged": {
            k: totals[k]
            for k in (
                "pct_chars_vs_spaced",
                "pct_canonical_glued_vs_spaced",
            )
        },
    }


def gluing_by_kind_boundaries(segs: Segmented) -> dict[str, dict[str, int]]:
    """Glued boundaries by kind, which is a far larger sample than excerpts."""
    out: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"boundaries": 0, "glued": 0}
    )
    for e, seg in segs:
        out[e.kind]["boundaries"] += len(seg.gaps)
        out[e.kind]["glued"] += len(seg.dropped)
    return dict(out)


def channel_comparison(segs: Segmented, encoding: str) -> dict[str, Any]:
    """The visible space channel against the invisible re-segmentation channel.

    Dropping a space is a choice a reader can see. Splitting a token is a
    choice that leaves the text byte-identical. Both carry information; only
    one of them shows up in the report.
    """
    spaced = "\n".join(seg.spaced() for _, seg in segs)
    invisible = invisible_channel_capacity(spaced, encoding)
    n_boundaries = sum(len(seg.gaps) for _, seg in segs)
    return {
        "visible_space_choice": {
            "n_boundaries": n_boundaries,
            "max_bits": float(n_boundaries),
            "bits_per_token": n_boundaries / invisible["n_tokens"],
        },
        "invisible_resegmentation": invisible,
    }


def report(results: dict[str, Any]) -> str:
    """Markdown summary of the numbers quoted in RESULTS.md."""
    lines: list[str] = []
    n_glued, n_seg = results["n_glued"], results["n_segmented"]
    lines.append(
        f"## Length ({n_glued} glued excerpts of {n_seg} with a reference spacing)"
    )
    lines.append("")
    lines.append("| measure | as written (glued) | spaced | change |")
    lines.append("| --- | --- | --- | --- |")
    for subset in ("glued_only", "all_with_reference"):
        t = results["length"][subset][PRIMARY_ENCODING]["totals"]
        tag = "glued excerpts" if subset == "glued_only" else "all excerpts"
        for label, glued_key, spaced_key, pct_key in (
            ("characters", "chars_glued", "chars_spaced", "pct_chars_vs_spaced"),
            (
                "canonical tokens",
                "canonical_glued",
                "canonical_spaced",
                "pct_canonical_glued_vs_spaced",
            ),
            (
                "word-at-a-time tokens",
                "wordwise_glued",
                "wordwise_spaced",
                "pct_wordwise_glued_vs_wordwise_spaced",
            ),
        ):
            lines.append(
                f"| {label} ({tag}) | {t[glued_key]} | {t[spaced_key]} | "
                f"{t[pct_key]:+.1f}% |"
            )
    t = results["length"]["glued_only"][PRIMARY_ENCODING]["totals"]
    lines += [
        "",
        f"Re-tokenizing the stored glued transcript costs "
        f"{t['pct_retokenization_inflation']:+.1f}% versus word-at-a-time emission "
        f"of the same text.",
        "",
        "Other tokenizers (glued excerpts, canonical tokens vs spaced):",
        "",
    ]
    for encoding in ENCODINGS:
        e = results["length"]["glued_only"][encoding]["totals"]
        lines.append(
            f"- `{encoding}`: {e['canonical_glued']} vs {e['canonical_spaced']} "
            f"({e['pct_canonical_glued_vs_spaced']:+.1f}%), word-at-a-time "
            f"{e['wordwise_glued']} vs {e['wordwise_spaced']} "
            f"({e['pct_wordwise_glued_vs_wordwise_spaced']:+.1f}%)"
        )

    sens = results["segmentation_sensitivity"]
    kinds = results["gluing_by_kind_boundaries"]
    lines += [
        "",
        f"Per-excerpt, not pooled: {t['n_excerpts_canonical_positive']} of "
        f"{results['n_glued']} glued excerpts are individually positive in canonical "
        f"tokens, range {t['per_excerpt_pct_canonical_min']:+.1f}% to "
        f"{t['per_excerpt_pct_canonical_max']:+.1f}%, mean "
        f"{t['per_excerpt_pct_canonical_mean']:+.1f}%.",
        "",
        f"Segmentation sensitivity: merging all {sens['n_boundaries_merged']} "
        f"digit-label boundaries (the reference's least certain calls) moves the "
        f"headline from "
        f"{sens['as_reported']['pct_chars_vs_spaced']:+.1f}% / "
        f"{sens['as_reported']['pct_canonical_glued_vs_spaced']:+.1f}% to "
        f"{sens['digit_labels_merged']['pct_chars_vs_spaced']:+.1f}% / "
        f"{sens['digit_labels_merged']['pct_canonical_glued_vs_spaced']:+.1f}%.",
        "",
        "Glued boundaries by kind (a larger sample than the excerpt counts):",
        "",
    ]
    for kind, d in sorted(kinds.items()):
        lines.append(f"- {kind}: {d['glued']}/{d['boundaries']}")

    b = results["boundaries"][PRIMARY_ENCODING]
    lines += [
        "",
        "## Which boundaries get glued",
        "",
        f"{b['n_dropped']} of {b['n_boundaries']} word boundaries in the glued "
        f"excerpts have no space.",
        "",
        f"- mean canonical tokens added per glued boundary: "
        f"{b['dropped']['mean_canonical_delta']:+.3f}",
        f"- mean that each kept boundary would have added, had it been glued: "
        f"{b['kept']['mean_canonical_delta']:+.3f}",
        f"- a canonical token straddles "
        f"{b['straddled_dropped_boundaries']}/{b['n_dropped']} glued boundaries",
        "",
        "| predictor | drop rate when true | n | drop rate when false |",
        "| --- | --- | --- | --- |",
    ]
    for name, pred in b["predictors"].items():
        lines.append(
            f"| {name} | {pred['drop_rate_true']:.0%} | {pred['n_true']} | "
            f"{pred['drop_rate_false']:.0%} |"
        )
    sav = b["token_saving_boundaries"]
    perm = b["permutation_test_mean_delta"]
    lines += [
        "",
        "### Token cost of the boundaries it chose",
        "",
        f"Mid-sentence only, since sentence boundaries are never glued: mean "
        f"canonical cost {b['dropped_mid_sentence']['mean_canonical_delta']:+.3f} for "
        f"the glued boundaries against "
        f"{b['kept_mid_sentence']['mean_canonical_delta']:+.3f} for the kept ones; "
        f"labels shuffled within each excerpt, "
        f"p={perm['p_one_sided_glued_cheaper']:.4f} one-sided "
        f"({perm['n_permutations']} permutations).",
        "",
        "| set | glued where token-free | glued where it costs tokens | Fisher p |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in (
        ("all boundaries", "free_vs_costly"),
        ("mid-sentence only", "free_vs_costly_mid_sentence"),
    ):
        fv = b[key]
        lines.append(
            f"| {label} | {fv['drop_rate_free']:.1%} (n={fv['n_free']}) "
            f"| {fv['drop_rate_costly']:.1%} (n={fv['n_costly']}) "
            f"| {fv['fisher_p']:.5f} |"
        )
    lines += [
        "",
        f"Boundaries where gluing would outright *save* a canonical token: "
        f"{sav['n']}, of which the model glued {sav['n_glued']} and only "
        f"{sav['n_mid_sentence']} are mid-sentence — too few to test, which is why "
        f"the free-versus-costly split above is the usable version.",
    ]

    ch = results["channels"]
    vis, inv = ch["visible_space_choice"], ch["invisible_resegmentation"]
    lines += [
        "",
        "## Channel capacity, visible vs invisible",
        "",
        "Both figures are upper bounds on capacity at a fixed reference text, per "
        "canonical reference token. Neither is a throughput a model could rely on: "
        "spending the invisible capacity emits more tokens than the reference has, "
        "and the visible figure assumes every boundary is free when sentence "
        "boundaries are never glued.",
        "",
        f"- space/no-space at {vis['n_boundaries']} word boundaries: at most "
        f"{vis['max_bits']:.0f} bits, {vis['bits_per_token']:.2f} per reference token, "
        f"and plainly visible in the transcript.",
        f"- re-segmenting the same characters into different vocabulary tokens: "
        f"{inv['n_tokens_with_alternatives']}/{inv['n_tokens']} tokens have at least "
        f"one alternative, {inv['bits']:.0f} bits total "
        f"({inv['bits_per_token']:.2f} per reference token), and invisible in a "
        f"transcript stored as text.",
    ]

    r = results["run_structure"]
    lines += [
        "",
        "## Run structure",
        "",
        f"Mean glued-run length {r['mean_run_length']:.2f} words carries no "
        f"information: it is fixed at {r['analytic_null_mean_run']:.2f} by the drop "
        f"rate {r['drop_rate']:.3f} and the excerpt count, however the drops are "
        f"arranged. The testable statistic is the spread.",
        "",
        f"Run-length variance {r['variance_run_length']:.2f} against "
        f"{r['bernoulli_null_variance']:.2f} for independent per-boundary dropping at "
        f"the same rate, simulated per excerpt (95% "
        f"{r['bernoulli_null_variance_ci95'][0]:.2f}-"
        f"{r['bernoulli_null_variance_ci95'][1]:.2f}, "
        f"p={r['p_null_variance_at_least_observed']:.2f}).",
        f"Run lengths: {r['run_length_hist']}",
    ]

    c = results["chunking"]
    sweep = c["context_sweep"]
    lines += [
        "",
        "## How hard the glued text is to emit canonically",
        "",
        f"- {c['n_chunked_as_standalone']}/{c['n_glued_words']} "
        f"({c['frac_chunked_as_standalone']:.0%}) of glued words are canonically "
        f"chunked exactly as if emitted on their own, so word-at-a-time emission is "
        f"canonical there.",
        f"- the other {len(c['mismatches'])} need chunks that reach across the word "
        f"boundary.",
        "",
        f"Context sweep, {sweep['n_right']} items x {sweep['n_left']} left contexts, "
        f"mean distinct chunk patterns per item. The GPT-2 integer rows run the same "
        f"sweep on random integers, so they are the like-for-like comparison; "
        f"clipped is the convention that matches how integer patterns are counted.",
        "",
        "| sweep | unclipped | clipped | pairs differing from standalone (clipped) |",
        "| --- | --- | --- | --- |",
    ]
    sweeps = [
        ("glued English (o200k)", sweep),
        (
            "glued English, spaced prefix",
            results["chunking"]["context_sweep_spaced_prefix"],
        ),
    ] + [
        (f"GPT-2 {name.replace('_', '-')} integers", d)
        for name, d in c["gpt2_integer_context_sweep"].items()
    ]
    for label, d in sweeps:
        lines.append(
            f"| {label} | {d['unclipped']['mean_distinct_patterns_per_item']:.2f} "
            f"(max {d['unclipped']['max_distinct_patterns']}) | "
            f"{d['clipped']['mean_distinct_patterns_per_item']:.2f} "
            f"(max {d['clipped']['max_distinct_patterns']}) | "
            f"{d['clipped']['frac_pairs_differing_from_standalone']:.1%} |"
        )
    lines.append("")
    for name, d in c["gpt2_integer_baseline"].items():
        lines.append(
            f"- GPT-2 {name.replace('_', '-')} integers, population view "
            f"(not comparable to the sweep): {d['neighbour_disagreement']:.1%} of "
            f"consecutive integers chunk differently, {d['n_distinct_patterns']} "
            f"distinct patterns across all of them, {d['n_single_token']} single-token."
        )
    lines += ["", "Glued words whose in-context chunks differ from their own:", ""]
    for m in c["mismatches"]:
        lines.append(f"- `{m['word']}`: {m['in_context']} vs {m['standalone']} alone")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    excerpts = load_excerpts()
    segs = segmentations(excerpts)
    glued = glued_only(segs)
    results: dict[str, Any] = {
        "n_excerpts": len(excerpts),
        "n_segmented": len(segs),
        "n_glued": len(glued),
        "gluing_by_kind": gluing_by_kind(excerpts),
        "gluing_by_kind_boundaries": gluing_by_kind_boundaries(segs),
        "segmentation_sensitivity": segmentation_sensitivity(glued, PRIMARY_ENCODING),
        "run_structure": run_structure(glued),
        "length": {
            "all_with_reference": {enc: length_table(segs, enc) for enc in ENCODINGS},
            "glued_only": {enc: length_table(glued, enc) for enc in ENCODINGS},
        },
        "boundaries": {enc: boundary_analysis(glued, enc) for enc in ENCODINGS},
        "chunking": chunking_analysis(glued, PRIMARY_ENCODING),
        "channels": channel_comparison(segs, PRIMARY_ENCODING),
    }
    (args.out_dir / "analysis.json").write_text(json.dumps(results, indent=1) + "\n")
    text = report(results)
    (args.out_dir / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
