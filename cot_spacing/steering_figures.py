"""Figures for the steering and learnability results.

Usage:
    uv run python -m cot_spacing.steering_figures
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

from cot_spacing.figures import FIG_DIR, INK, MUTED, SECONDARY, _frame, _save
from cot_spacing.learnability import step_gains
from cot_spacing.steering import DIRECTIONS

RESULTS = Path(__file__).parent / "results"
#: validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all ->
#: PASS; aqua is under 3:1 against the surface, so every line is labelled.
COLOR = {"glued": "#2a78d6", "short": "#eb6834", "both": "#1baf7a"}
LABEL = {"glued": "glue", "short": "short forms", "both": "both"}
METRICS = (
    ("letters_per_space", "letters per space\n(higher = more gluing)"),
    ("short_share", "short-form share\n(of dictionary words)"),
    ("accuracy", "GSM8K accuracy\n(32 problems)"),
)


#: A run counts as coherent while it keeps GSM8K accuracy within this much of
#: unsteered gpt-oss; past that its gluing and shortening are drawn hollow.
COHERENT_DROP = 0.1


def steering_figure(scores: dict[str, list[dict[str, object]]], out: Path) -> None:
    summary = scores["summary"]
    base = next(s for s in summary if s["direction"] == "none")
    layers = sorted({int(s["layer"]) for s in summary if s["direction"] != "none"})  # type: ignore[arg-type]
    fig, axes = plt.subplots(
        len(METRICS), len(layers), figsize=(3.6 * len(layers), 8.2), sharex=True
    )
    for col, layer in enumerate(layers):
        for row, (key, label) in enumerate(METRICS):
            ax = axes[row][col]
            _frame(ax)
            ax.axhline(base[key], color=MUTED, lw=1, ls="--", zorder=1)  # type: ignore[arg-type]
            for d in DIRECTIONS:
                runs = sorted(
                    (s for s in summary if s["direction"] == d and s["layer"] == layer),
                    key=lambda s: float(s["scale"]),  # type: ignore[arg-type]
                )
                runs = [s for s in runs if s[key] is not None]
                xs = [float(s["scale"]) for s in runs]  # type: ignore[arg-type]
                ys = [float(s[key]) for s in runs]  # type: ignore[arg-type]
                ok = [
                    float(s["accuracy"]) >= float(base["accuracy"]) - COHERENT_DROP  # type: ignore[arg-type]
                    for s in runs
                ]
                ax.plot(xs, ys, color=COLOR[d], lw=2, zorder=3, label=LABEL[d])
                ax.scatter(
                    xs,
                    ys,
                    s=30,
                    zorder=4,
                    linewidths=1.5,
                    edgecolors=COLOR[d],
                    facecolors=[COLOR[d] if k else "white" for k in ok],
                )
            ax.set_xscale("log", base=2)
            if key == "letters_per_space":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
            if row == 0:
                ax.set_title(f"steered at layer {layer}", fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=SECONDARY)
            if row == len(METRICS) - 1:
                ax.set_xlabel(
                    "strength (× median residual norm)", fontsize=9, color=SECONDARY
                )
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, lw=1, ls="--"))
    labels.append("unsteered")
    handles.append(
        Line2D([], [], color=SECONDARY, marker="o", ls="", mfc="white", mew=1.5)
    )
    labels.append(f"accuracy down more than {COHERENT_DROP * 100:.0f} points")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save(fig, out)


def learnability_figure(gradients: Path, sites: Path, out: Path) -> None:
    gains = step_gains(gradients)
    site_rows = json.loads(sites.read_text())["sites"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    _frame(left)
    for i, d in enumerate(DIRECTIONS):
        for j, key in enumerate(("held_out_per_char", "same_per_char")):
            ys = [float(r[key]) for r in gains if r["direction"] == d]  # type: ignore[arg-type]
            x = i + (j - 0.5) * 0.3
            left.scatter(
                [x] * len(ys),
                ys,
                s=22,
                color=COLOR[d],
                zorder=3,
                alpha=1.0 if j == 0 else 0.45,
                edgecolor="white",
                lw=0.8,
            )
            left.hlines(
                statistics.mean(ys), x - 0.1, x + 0.1, color=INK, lw=1.5, zorder=4
            )
    left.axhline(0, color=MUTED, lw=1)
    left.set_xticks(range(len(DIRECTIONS)), [LABEL[d] for d in DIRECTIONS])
    left.set_ylabel(
        "margin gained per character saved\n(per unit training step)",
        fontsize=9,
        color=SECONDARY,
    )
    left.set_title(
        "Fine-tuning on the rewrite (solid: other excerpts;\nfaded: the same excerpt)",
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
    scores = json.loads((RESULTS / "steering" / "scores.json").read_text())
    steering_figure(scores, FIG_DIR / "steering.png")
    learnability_figure(
        RESULTS / "learnability" / "gradients.json",
        RESULTS / "learnability" / "sites.json",
        FIG_DIR / "learnability.png",
    )


if __name__ == "__main__":
    main()
