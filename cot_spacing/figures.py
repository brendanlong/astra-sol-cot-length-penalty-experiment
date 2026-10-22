"""Figures for the cot_spacing experiment.

- ``length``: the headline result — dropping spaces shortens the two things a
  length penalty could count and *lengthens* the thing a token counter counts.
- ``chunking``: the Beren-Millidge analogue — a glued word's chunking moves
  with its left neighbour, shown beside GPT-2 integers under the same sweep.

Regenerate: ``uv run python -m cot_spacing.figures``.

Every number printed on a figure is read from ``results/analysis.json``, the
run recorded in RESULTS.md; none is typed in here. The per-cell grids cannot
be, since the sweeps are 37200 cells each, so they are recomputed with the
same ``chunking``/``analysis`` helpers the run used and then checked cell for
cell against the JSON's summary statistics (:func:`_check_sweep`). If the
dataset, the segmentation or the pattern convention moves under us, this
raises instead of drawing a figure that disagrees with RESULTS.md.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import tiktoken
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from cot_spacing.analysis import (
    RESULTS_DIR,
    glued_only,
    segmentations,
    sweep_stats,
)
from cot_spacing.chunking import chunk_pattern, words_in_context
from cot_spacing.data import load_excerpts
from cot_spacing.tokenize_cost import PRIMARY_ENCODING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

FIG_DIR = Path(__file__).parent / "figures"

# Palette: the documented default instance (the dataviz skill's
# references/palette.md), validated with its scripts/validate_palette.js.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Diverging pair for "which side of zero": blue <-> red, warm/cool poles.
#: validate_palette.js "#2a78d6,#e34948" --mode light --pairs all -> all PASS.
DOWN = "#2a78d6"
UP = "#e34948"
#: Ordinal ramp for "how far from the item's own chunking": blue steps
#: 250/450/600/700. validate_palette.js --ordinal --mode light -> all PASS.
RAMP = ("#86b6ef", "#2a78d6", "#184f95", "#0d366b")
RANK_LABELS = (
    "chunked as if alone",
    "2nd chunking",
    "3rd chunking",
    "4th or rarer",
)

#: Rows drawn from each sweep: every 6th right item, so the subset is a
#: systematic sample rather than a flattering pick. 150 // 6 = 25 rows.
ROW_STRIDE = 6
#: The integer sweep to draw. 4-digit is Beren Millidge's case; RESULTS.md
#: reports the 3-digit sweep too and the caption quotes both, because the two
#: bracket the glued-word numbers from opposite sides.
INT_DIGITS = "4_digit"
INT_RANGE = (1000, 10000)
#: Seeds of the integer sweep, matching ``analysis.chunking_analysis`` exactly
#: — a different draw would be a different sweep from the one in RESULTS.md.
INT_RIGHT_SEED, INT_LEFT_SEED = 0, 1


def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((results_dir / "analysis.json").read_text())
    return out


def _frame(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, length=3)


def _pct(value: float) -> str:
    """Signed percent with a real minus sign, not a hyphen."""
    return f"{value:+.1f}%".replace("-", "−")


def _save(fig: Figure, out_path: Path) -> None:
    """Write the PNG, and an SVG beside it.

    The HTML report inlines the SVG rather than drawing its own copy of the
    same chart, so there is one implementation and the page cannot drift from
    the figure the repo ships. ``svg.fonttype="none"`` keeps the labels as
    text rather than outlines: smaller, selectable, and readable by a screen
    reader.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(
            out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)
    print(f"Wrote {out_path} (+ .svg)")


# --------------------------------------------------------------------------
# Figure 1: what gluing actually saves
# --------------------------------------------------------------------------


def plot_length(out_path: Path, results: dict[str, Any]) -> None:
    """Diverging bars: the sign flip between the length units and the token units."""
    t = results["length"]["glued_only"][PRIMARY_ENCODING]["totals"]
    rows = [
        ("characters", t["pct_chars_vs_spaced"], t["chars_glued"], t["chars_spaced"]),
        (
            "tokens, one word at a time",
            t["pct_wordwise_glued_vs_wordwise_spaced"],
            t["wordwise_glued"],
            t["wordwise_spaced"],
        ),
        (
            "tokens, re-encoded as one string",
            t["pct_canonical_glued_vs_spaced"],
            t["canonical_glued"],
            t["canonical_spaced"],
        ),
    ]

    fig = plt.figure(figsize=(9.0, 4.1))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0.30, 0.275, 0.665, 0.50))
    _frame(ax)
    ax.spines["left"].set_visible(False)
    ys = range(len(rows))
    pcts = [r[1] for r in rows]
    ax.barh(
        list(ys), pcts, height=0.3,
        color=[DOWN if p < 0 else UP for p in pcts], zorder=3,
    )  # fmt: skip
    trans = ax.get_yaxis_transform()
    for y, (name, pct, glued, spaced) in enumerate(rows):
        ax.annotate(
            _pct(pct), xy=(pct, y), xytext=(-7 if pct < 0 else 7, 0),
            textcoords="offset points", ha="right" if pct < 0 else "left",
            va="center", fontsize=12, fontweight="bold", color=INK, zorder=4,
        )  # fmt: skip
        ax.text(
            -0.012, y - 0.1, name, transform=trans, ha="right", va="center",
            fontsize=10.5, color=INK,
        )  # fmt: skip
        ax.text(
            -0.012, y + 0.18, f"{glued:,} vs {spaced:,}", transform=trans,
            ha="right", va="center", fontsize=9, color=MUTED,
        )  # fmt: skip
    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax.set_yticks([])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-12, 12)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel(
        "every glued excerpt, against the same words with the spaces put back (%)",
        fontsize=10, color=SECONDARY,
    )  # fmt: skip
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)

    fig.text(
        0.012, 0.955,
        "Dropping the spaces makes the text shorter in characters\n"
        "and longer in tokens",
        fontsize=13.5, color=INK, ha="left", va="top",
    )  # fmt: skip
    fig.text(
        0.012, 0.015,
        f"{results['n_glued']} chain-of-thought and agent-message excerpts from "
        "GPT-6.1 Sol and GPT-6 Astra, published in OpenAI's misalignment report "
        "of 2026-10-20,\nmeasured against OpenAI's own space-restored reading of "
        "the same text. Tokens are tiktoken o200k_base, the closest public "
        "stand-in for\nGPT-6's unpublished tokenizer. "
        f"{t['n_excerpts_canonical_positive']} of the {results['n_glued']} excerpts "
        "are individually longer in tokens, not only the total.",
        fontsize=8.5, color=MUTED, ha="left", va="bottom",
    )  # fmt: skip
    _save(fig, out_path)


def _sweep_words(encoding: str) -> tuple[list[str], list[str]]:
    """The sweep's right words and left contexts, by ``chunking_analysis``'s rule."""
    enc = tiktoken.get_encoding(encoding)
    segs = glued_only(segmentations(load_excerpts()))
    words = [w for e, seg in segs for w in words_in_context(e.id, seg, enc)]
    rights = sorted(
        {w.word for w in words if w.glued_left and w.word.isalpha() and len(w.word) > 2}
    )
    lefts = sorted({w.word for w in words if w.word.isalpha() and len(w.word) > 1})
    return rights, lefts


def _sweep_integers(n_right: int, n_left: int) -> tuple[list[str], list[str]]:
    """The integer sweep's items, drawn exactly as ``chunking_analysis`` draws them."""
    lo, hi = INT_RANGE
    rights = [
        str(n) for n in random.Random(INT_RIGHT_SEED).sample(range(lo, hi), n_right)
    ]
    lefts = [str(n) for n in random.Random(INT_LEFT_SEED).sample(range(lo, hi), n_left)]
    return rights, lefts


def _check_sweep(
    name: str,
    rights: Sequence[str],
    lefts: Sequence[str],
    enc: tiktoken.Encoding,
    published: dict[str, Any],
) -> None:
    """Fail loudly if the recomputed sweep no longer matches analysis.json."""
    got = sweep_stats(list(rights), list(lefts), enc)
    for stats in got.values():
        # JSON has no integer keys, so the histogram comes back stringified.
        if isinstance(stats, dict):
            hist = stats["distinct_pattern_hist"]
            stats["distinct_pattern_hist"] = {str(k): v for k, v in hist.items()}
    want = {k: published[k] for k in got}
    if got != want:
        msg = (
            f"{name} sweep recomputed as {got!r}, but results/analysis.json says "
            f"{want!r}. Re-run `python -m cot_spacing.analysis`, check "
            "RESULTS.md, then regenerate the figures."
        )
        raise ValueError(msg)


def _rank_grid(
    rights: Sequence[str], lefts: Sequence[str], enc: tiktoken.Encoding
) -> tuple[list[list[int]], list[int]]:
    """Chunking ranks of each right item over the left contexts, and its pattern count.

    Rank 1 is the pattern the item takes on its own; its other patterns take
    ranks 2, 3, ... by how many left contexts produce them. Rank 1 is
    therefore a genuine zero ("gluing changed nothing here"), so the ink in a
    panel is exactly the pairs that chunk differently from standalone.

    ``clip=True`` throughout: that convention counts only the bytes of the
    right item, which is the one the integer baseline uses and so the only one
    under which the two panels can be compared (see ``chunk_pattern``).
    """
    grid: list[list[int]] = []
    distinct: list[int] = []
    for right in rights:
        alone = chunk_pattern(right, enc, clip=True)
        pats = [chunk_pattern(right, enc, prefix=left, clip=True) for left in lefts]
        counts = collections.Counter(pats)
        others = sorted(
            (p for p in counts if p != alone), key=lambda p: (-counts[p], p)
        )
        rank = {alone: 1} | {p: i for i, p in enumerate(others, start=2)}
        grid.append([min(rank[p], len(RAMP)) for p in pats])
        distinct.append(len(counts))
    return grid, distinct


def _sampled_panel(
    rights: Sequence[str], lefts: Sequence[str], enc: tiktoken.Encoding
) -> tuple[list[str], list[int], list[list[int]], float, float]:
    """Rows of a panel: every ROW_STRIDE-th right item, ordered by how much it moves."""
    sampled = list(rights[::ROW_STRIDE])
    grid, distinct = _rank_grid(sampled, lefts, enc)
    differing = sum(r != 1 for row in grid for r in row) / (len(sampled) * len(lefts))
    mean_distinct = sum(distinct) / len(distinct)
    # The subset is fixed by ROW_STRIDE; only the row order is cosmetic, so
    # order the rows by how often gluing moves the chunking.
    order = sorted(
        range(len(sampled)), key=lambda i: (-sum(r != 1 for r in grid[i]), sampled[i])
    )
    return (
        [sampled[i] for i in order],
        [distinct[i] for i in order],
        [grid[i] for i in order],
        mean_distinct,
        differing,
    )


def _draw_cells(ax: Axes, grid: Sequence[Sequence[int]], labels: Sequence[str]) -> None:
    """Paint a rank grid with its row labels.

    No surface gap between cells here: at 248 columns a cell is about 3px
    wide, so the usual 2px separator would consume most of the mark. A raster
    of touching cells is the honest trade at this density.
    """
    ax.pcolormesh(
        grid,
        cmap=ListedColormap(RAMP),
        norm=BoundaryNorm([i + 0.5 for i in range(len(RAMP) + 1)], len(RAMP)),
        edgecolors="none",
        rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.tick_params(axis="y", length=0, labelcolor=SECONDARY)


def _panel_title(ax: Axes, heading: str, stats: dict[str, Any]) -> None:
    ax.set_title(
        f"{heading}\n"
        f"mean {stats['mean_distinct_patterns_per_item']:.2f} chunkings per item "
        f"(max {stats['max_distinct_patterns']}), and "
        f"{stats['frac_pairs_differing_from_standalone']:.1%} of pairs\n"
        "chunk the item differently than it chunks alone",
        fontsize=10.5,
        color=INK,
        loc="left",
        pad=10,
    )


def plot_chunking(out_path: Path, results: dict[str, Any]) -> None:
    """GPT-2 integers beside glued English words, under the same context sweep."""
    c = results["chunking"]
    word_sweep = c["context_sweep"]
    int_sweeps = c["gpt2_integer_context_sweep"]
    int_sweep = int_sweeps[INT_DIGITS]
    population = c["gpt2_integer_baseline"][INT_DIGITS]

    enc = tiktoken.get_encoding(PRIMARY_ENCODING)
    gpt2 = tiktoken.get_encoding("gpt2")
    rights, lefts = _sweep_words(PRIMARY_ENCODING)
    int_rights, int_lefts = _sweep_integers(word_sweep["n_right"], word_sweep["n_left"])
    _check_sweep("glued English", rights, lefts, enc, word_sweep)
    _check_sweep("GPT-2 integer", int_rights, int_lefts, gpt2, int_sweep)

    int_rows, int_distinct, int_grid, int_mean, int_differing = _sampled_panel(
        int_rights, int_lefts, gpt2
    )
    rows, distinct, word_grid, word_mean, word_differing = _sampled_panel(
        rights, lefts, enc
    )

    fig, (ax_int, ax_word) = plt.subplots(1, 2, figsize=(13.4, 6.5))
    fig.patch.set_facecolor("white")

    _draw_cells(
        ax_int,
        int_grid,
        [f"{r}  ({n})" for r, n in zip(int_rows, int_distinct, strict=True)],
    )
    _panel_title(
        ax_int, "GPT-2 on an integer glued after an integer", int_sweep["clipped"]
    )
    ax_int.set_xlabel(
        f"{len(int_lefts)} left-context integers →", fontsize=9.5, color=SECONDARY
    )

    _draw_cells(
        ax_word,
        word_grid,
        [f"{w}  ({n})" for w, n in zip(rows, distinct, strict=True)],
    )
    _panel_title(ax_word, "o200k on a word glued after a word", word_sweep["clipped"])
    ax_word.set_xlabel(
        f"{len(lefts)} left-context word types from the same excerpts →",
        fontsize=9.5,
        color=SECONDARY,
    )

    # Legend and captions go on after the layout pass: they are wider than the
    # panels, and tight_layout would otherwise shrink both grids to make room
    # for text that bbox_inches="tight" is happy to keep outside.
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.91), w_pad=3.5)
    fig.legend(
        handles=[
            Patch(facecolor=col, label=lab)
            for col, lab in zip(RAMP, RANK_LABELS, strict=True)
        ],
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
        handlelength=1.1,
        handleheight=1.1,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.175),
        ncol=4,
    )
    other = "3_digit" if INT_DIGITS == "4_digit" else "4_digit"
    other_clipped = int_sweeps[other]["clipped"]
    fig.text(
        0.008,
        0.015,
        f"{INT_DIGITS.replace('_', '-')} integers, {len(int_rights)} right "
        f"× {len(int_lefts)} left, sampled from {INT_RANGE[0]} to "
        f"{INT_RANGE[1] - 1} with the seeds\nanalysis.py uses. The "
        f"{other.replace('_', '-')} sweep gives "
        f"{other_clipped['mean_distinct_patterns_per_item']:.2f} chunkings and "
        f"{other_clipped['frac_pairs_differing_from_standalone']:.1%} of pairs "
        "differing;\nthe two bracket the glued-word numbers from opposite sides. "
        "Beren Millidge's\npopulation view of the same tokenizer — 2249 = 2|249, "
        "2250 = 22|50, 2251 = 225|1,\nand "
        f"{population['neighbour_disagreement']:.1%} of consecutive integers "
        "chunked unlike their successor — is what\nthis sweep generalises: there, "
        "each integer still has exactly one chunking.",
        fontsize=8.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    fig.text(
        0.5,
        0.015,
        f"Rows of both panels: every {ROW_STRIDE}th right item of the "
        f"{len(rights)} in each sweep — {len(rows)} of them,\nordered here by how "
        "often they move. The samples reproduce the full sweeps:\n"
        f"{word_mean:.2f} chunkings and {word_differing:.1%} of pairs differing for "
        f"the words, {int_mean:.2f} and {int_differing:.1%} for the\nintegers. The "
        "number after each item is its chunkings over all left contexts.\nShade = "
        "how common that chunking is for that item. Chunk patterns are clipped\nto "
        "the right item's own bytes, the convention the integer baseline uses.",
        fontsize=8.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    fig.text(
        0.008,
        0.985,
        "A glued word's chunking moves with its left neighbour — "
        "about as much as a GPT-2 integer's",
        fontsize=13.5,
        color=INK,
        ha="left",
        va="top",
    )
    _save(fig, out_path)


def main() -> None:
    results = load_results()
    plot_length(FIG_DIR / "length.png", results)
    plot_chunking(FIG_DIR / "chunking.png", results)


if __name__ == "__main__":
    main()
