"""Figures for the steering and learnability results.

Usage:
    uv run python -m cot_spacing.steering_figures
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from cot_spacing.figures import FIG_DIR, INK, MUTED, SECONDARY, _frame, _save
from cot_spacing.learnability import REWRITES, step_gains
from cot_spacing.steering import COHERENT_DROP, coherent

RESULTS = Path(__file__).parent / "results"
#: validate_palette.js --mode light --pairs all passes for both figures'
#: palettes: "#2a78d6,#eb6834,#1baf7a" and "#2a78d6,#eda100,#e87ba4". The light
#: ones are under 3:1 against the surface, so every figure has a legend.
COLOR = {
    "glued": "#2a78d6",
    "short": "#eb6834",
    "both": "#1baf7a",
    "nospace": "#eda100",
    "glued_wordwise": "#e87ba4",
}
LABEL = {
    "glued": "glue",
    "short": "short forms",
    "both": "both",
    "nospace": "no spaces at all",
    "glued_wordwise": "glue, word-by-word tokens",
}
GLUE_VARIANTS = ("glued", "nospace", "glued_wordwise")

Metric = tuple[str, Callable[[dict[str, Any]], float | None], bool]


def field(key: str) -> Callable[[dict[str, Any]], float | None]:
    return lambda s: s[key]


def ratio(num: str, den: str) -> Callable[[dict[str, Any]], float | None]:
    return lambda s: s[num] / s[den] if s.get(den) else None


def word_spaces_kept(base: dict[str, Any]) -> Callable[[dict[str, Any]], float | None]:
    """Spaces between words per letter, as a share of unsteered gpt-oss's."""
    per_letter = base["total_word_spaces"] / base["total_reasoning_letters"]

    def kept(s: dict[str, Any]) -> float | None:
        if not s["total_reasoning_letters"]:
            return None
        return s["total_word_spaces"] / s["total_reasoning_letters"] / per_letter

    return kept


REWRITE_METRICS: tuple[Metric, ...] = (
    (
        "letters per space,\nreasoning (median)",
        field("reasoning_letters_per_space"),
        True,
    ),
    ("short-form share,\nreasoning", field("reasoning_short_share"), False),
    ("short-form share,\nanswer", field("answer_short_share"), False),
    ("GSM8K accuracy\n(32 problems)", field("accuracy"), False),
)


def spacing_metrics(base: dict[str, Any]) -> tuple[Metric, ...]:
    return (
        (
            "spaces kept after\nsentence ends",
            ratio("total_sentence_spaces", "total_sentence_ends"),
            False,
        ),
        ("spaces kept between\nwords (vs unsteered)", word_spaces_kept(base), False),
        (
            "non-canonical share\nof generated tokens",
            ratio("noncanonical_tokens", "tokens"),
            False,
        ),
        ("GSM8K accuracy\n(32 problems)", field("accuracy"), False),
    )


def steering_figure(
    summary: list[dict[str, Any]],
    directions: tuple[str, ...],
    metrics: tuple[Metric, ...],
    out: Path,
) -> None:
    base = next(s for s in summary if s["direction"] == "none")
    layers = sorted({int(s["layer"]) for s in summary if s["direction"] != "none"})
    fig, axes = plt.subplots(
        len(metrics),
        len(layers),
        figsize=(3.6 * len(layers), 2.6 * len(metrics)),
        sharex=True,
    )
    for col, layer in enumerate(layers):
        for row, (label, value, log) in enumerate(metrics):
            ax = axes[row][col]
            _frame(ax)
            if (b := value(base)) is not None:
                ax.axhline(b, color=MUTED, lw=1, ls="--", zorder=1)
            for d in directions:
                runs = sorted(
                    (s for s in summary if s["direction"] == d and s["layer"] == layer),
                    key=lambda s: s["scale"],
                )
                runs = [s for s in runs if value(s) is not None]
                xs = [s["scale"] for s in runs]
                ys = [value(s) for s in runs]
                ax.plot(xs, ys, color=COLOR[d], lw=2, zorder=3, label=LABEL[d])
                ax.scatter(
                    xs,
                    ys,
                    s=30,
                    zorder=4,
                    linewidths=1.5,
                    edgecolors=COLOR[d],
                    facecolors=[
                        COLOR[d] if coherent(s, base) else "white" for s in runs
                    ],
                )
            ax.set_xscale("log", base=2)
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
            if row == 0:
                ax.set_title(f"steered at layer {layer}", fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=SECONDARY)
            if row == len(metrics) - 1:
                ax.set_xlabel(
                    "strength (× median residual norm)", fontsize=9, color=SECONDARY
                )
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, lw=1, ls="--"))
    labels.append("unsteered")
    handles.append(
        Line2D([], [], color=SECONDARY, marker="o", ls="", mfc="white", mew=1.5)
    )
    labels.append(
        f"incoherent: an answer loops or breaks, or accuracy\nor answers finished "
        f"down > {COHERENT_DROP * 100:.0f} points"
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, out)


def learnability_figure(gradients: Path, sites: Path, out: Path) -> None:
    gains = step_gains(gradients)
    site_rows = json.loads(sites.read_text())["sites"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    _frame(left)
    for i, d in enumerate(REWRITES):
        ys = [float(r["margin_cosine"]) for r in gains if r["direction"] == d]  # type: ignore[arg-type]
        left.scatter(
            [i] * len(ys), ys, s=26, color=COLOR[d], zorder=3, edgecolor="white", lw=0.8
        )
        left.hlines(
            statistics.mean(ys), i - 0.15, i + 0.15, color=INK, lw=1.5, zorder=4
        )
    left.axhline(0, color=MUTED, lw=1)
    left.set_ylim(0, 1)
    left.set_xticks(range(len(REWRITES)), [LABEL[d] for d in REWRITES])
    left.set_ylabel(
        "cosine: training excerpts' margin gradient\nvs held-out margin gradient",
        fontsize=9,
        color=SECONDARY,
    )
    left.set_title(
        "Does raising the rewrite in 4 excerpts raise it\nin the 5th? (line: mean)",
        fontsize=9.5,
        color=INK,
    )

    _frame(right)
    for i, kind in enumerate(("glue", "short")):
        ys = [r["log_ratio"] / math.log(10) for r in site_rows if r["kind"] == kind]
        xs = [i + (k % 15 - 7) * 0.02 for k in range(len(ys))]
        color = COLOR["glued" if kind == "glue" else "short"]
        right.scatter(xs, ys, s=6, color=color, alpha=0.5, lw=0, zorder=3)
        right.hlines(
            statistics.median(ys), i - 0.25, i + 0.25, color=INK, lw=1.5, zorder=4
        )
    right.set_xticks(
        [0, 1],
        [
            f"drop next space\n(n={sum(r['kind'] == 'glue' for r in site_rows)})",
            f"use short form\n(n={sum(r['kind'] == 'short' for r in site_rows)})",
        ],
    )
    right.set_ylabel("log10 P(edit) / P(as written)", fontsize=9, color=SECONDARY)
    right.set_title(
        "How likely unsteered gpt-oss is to make\neach edit by itself (line: median)",
        fontsize=9.5,
        color=INK,
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, out)


def main() -> None:
    summary = json.loads((RESULTS / "steering" / "scores.json").read_text())["summary"]
    base = next(s for s in summary if s["direction"] == "none")
    steering_figure(summary, REWRITES, REWRITE_METRICS, FIG_DIR / "steering.png")
    steering_figure(
        summary, GLUE_VARIANTS, spacing_metrics(base), FIG_DIR / "steering_spaces.png"
    )
    learnability_figure(
        RESULTS / "learnability" / "gradients.json",
        RESULTS / "learnability" / "sites.json",
        FIG_DIR / "learnability.png",
    )


if __name__ == "__main__":
    main()
