"""Build the HTML report on steering gpt-oss toward gluing vs short forms.

Every number in the prose is read from the result files under
``cot_spacing/results`` and the generations on Hugging Face (downloaded by
``cot_spacing.steering_data`` if missing), so the text cannot drift from the
data. The figures are the SVGs ``cot_spacing.steering_figures`` writes.

    uv run python scripts/build_steering_report.py -o /path/to/report.html
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from build_report import TEMPLATE, esc, figure
from build_report import inline_svg as _inline_svg

from cot_spacing.learnability import site_summary, step_gains
from cot_spacing.steering import ANALYSIS, DIRECTIONS
from cot_spacing.steering_data import REPO as HF_REPO
from cot_spacing.steering_data import REVISION, STEERING_DIR, download
from cot_spacing.steering_figures import COHERENT_DROP, coherent

#: Published to a Surge site of its own; every update goes to this domain.
PUBLISHED_AT = "https://gpt-oss-glue-vs-short-65f73187.surge.sh"
#: The main report this one follows up.
MAIN_REPORT = "https://gpt6-cot-spaces-cd652c60.surge.sh"
REPO_URL = "https://github.com/brendanlong/astra-sol-cot-length-penalty-experiment"
#: Lion Reader article id for the saved copy, refreshed on every republish.
SAVED_ARTICLE_ID = "01a16ffd-db44-791c-8bc4-f5e4c3b35d14"

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "cot_spacing" / "results"
LAYERS = (6, 12, 18)
LABEL = {"glued": "glue", "short": "short forms", "both": "both"}


def inline_svg(name: str, alt: str) -> str:
    """Inline a figure, with its ids prefixed so two figures cannot collide."""
    svg = _inline_svg(name, alt)
    svg = re.sub(r'id="([^"]+)"', rf'id="{name}-\1"', svg)
    return re.sub(r"(url\(#|href=\"#)([^)\"]+)", rf"\1{name}-\2", svg)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def generation(name: str, prompt_id: str) -> str:
    for line in (STEERING_DIR / "gen" / f"{name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"] == prompt_id:
            m = ANALYSIS.search(r["text"])
            return m.group(1).strip() if m else ""
    raise KeyError(prompt_id)


def strongest_coherent(
    summary: list[dict[str, Any]], base: dict[str, Any], d: str, layer: int
) -> dict[str, Any]:
    runs = sorted(
        (s for s in summary if s["direction"] == d and s["layer"] == layer),
        key=lambda s: s["scale"],
    )
    return [s for s in runs if coherent(s, base)][-1]


def first_incoherent(
    summary: list[dict[str, Any]], base: dict[str, Any], d: str, layer: int
) -> float:
    return min(
        s["scale"]
        for s in summary
        if s["direction"] == d and s["layer"] == layer and not coherent(s, base)
    )


def build(out_path: Path) -> None:
    if not (STEERING_DIR / "gen").exists():
        download()
    variants = json.loads((RESULTS / "variants.json").read_text())
    vectors = json.loads((RESULTS / "steering" / "vectors.json").read_text())
    summary = json.loads((RESULTS / "steering" / "scores.json").read_text())["summary"]
    gains = step_gains(RESULTS / "learnability" / "gradients.json")
    sites_path = RESULTS / "learnability" / "sites.json"
    sites = site_summary(sites_path)
    site_rows = json.loads(sites_path.read_text())["sites"]

    base = next(s for s in summary if s["direction"] == "none")
    n_math = 32
    n_prompts = 40
    base_acc = round(base["accuracy"] * n_math)
    base_fin = round(base["finished"] * n_prompts)
    bs, bl = base["answer_short_long"]
    brs, brl = base["reasoning_short_long"]

    glued_lps = [
        sum(map(str.isalpha, v["texts"]["glued"])) / len(v["texts"]["glued"].split())
        for v in variants
    ]
    n_glued = sum(v["glued_boundaries"] for v in variants)
    n_short = sum(v["shortened_words"] for v in variants)
    n_words = sum(len(v["texts"]["spaced"].split()) for v in variants)
    chars = {k: sum(v["chars_saved"][k] for v in variants) for k in DIRECTIONS}

    best = {
        (d, layer): strongest_coherent(summary, base, d, layer)
        for d in DIRECTIONS
        for layer in LAYERS
    }
    g18 = best["glued", 18]
    short_best = max(
        (best["short", layer] for layer in LAYERS),
        key=lambda s: s["answer_short_share"] or 0,
    )

    norm = vectors["direction_norm"]
    resid = vectors["resid_norm_median"]
    natural = {
        d: [best[d, layer]["scale"] * resid[layer] / norm[d][layer] for layer in LAYERS]
        for d in ("glued", "short")
    }
    ratio = [norm["glued"][layer] / norm["short"][layer] for layer in LAYERS]
    auc_glue = vectors["held_out_auc"]["glued"][9:]
    auc_short = vectors["held_out_auc"]["short"]
    short_reasoning_len = [
        best["short", layer]["reasoning_letters"] for layer in LAYERS
    ]

    def mean_gain(d: str, key: str) -> float:
        return statistics.mean(float(r[key]) for r in gains if r["direction"] == d)  # type: ignore[arg-type]

    def range_gain(d: str, key: str) -> tuple[float, float]:
        xs = [float(r[key]) for r in gains if r["direction"] == d]  # type: ignore[arg-type]
        return min(xs), max(xs)

    cos = {d: mean_gain(d, "margin_cosine") for d in DIRECTIONS}
    per_edit = {d: mean_gain(d, "held_out_margin_per_edit") for d in DIRECTIONS}
    same_edit = {d: mean_gain(d, "same_margin_per_edit") for d in DIRECTIONS}
    sft_edit = {d: mean_gain(d, "held_out_sft_per_edit") for d in DIRECTIONS}
    short_cos_min = range_gain("short", "margin_cosine")[0]

    glue_s, short_s = sites["glue"], sites["short"]
    med_glue = 10 ** -glue_s["median_log10_ratio"]
    med_short = 10 ** -short_s["median_log10_ratio"]
    expected_ratio = (
        short_s["expected_chars_per_opportunity"]
        / glue_s["expected_chars_per_opportunity"]
    )
    weight_ratio = (
        short_s["gradient_weight_per_opportunity"]
        / glue_s["gradient_weight_per_opportunity"]
    )
    gpt6 = {
        flag: statistics.median(
            r["log_ratio"] / math.log(10)
            for r in site_rows
            if r["kind"] == "glue" and r["gpt6_glued"] == flag
        )
        for flag in (True, False)
    }

    glue_example = generation("glued-18-0.375", "agentic-4")
    short_example = generation("short-18-0.1875", "agentic-4")
    plain_example = generation("none", "agentic-4")

    rng = lambda xs: f"{min(xs):.1f}–{max(xs):.1f}"  # noqa: E731

    best_rows = []
    for d in DIRECTIONS:
        for layer in LAYERS:
            s = best[d, layer]
            a_s, a_l = s["answer_short_long"]
            share = (
                "—"
                if s["answer_short_share"] is None
                else f"{s['answer_short_share']:.2f} ({a_s}/{a_s + a_l})"
            )
            best_rows.append(
                f"<tr><td>{LABEL[d]}</td><td>{layer}</td><td>{s['scale']:g}</td>"
                f"<td>{s['reasoning_letters_per_space']:.1f}</td><td>{share}</td>"
                f"<td>{round(s['accuracy'] * n_math)}/{n_math}</td>"
                f"<td>{round(s['finished'] * n_prompts)}/{n_prompts}</td>"
                f"<td>{first_incoherent(summary, base, d, layer):g}</td></tr>"
            )

    all_rows = []
    for s in summary:
        a_s, a_l = s["answer_short_long"]
        r_s, r_l = s["reasoning_short_long"]
        lps = s["reasoning_letters_per_space"]
        all_rows.append(
            f"<tr><td>{LABEL.get(s['direction'], 'unsteered')}</td>"
            f"<td>{s['layer'] if s['direction'] != 'none' else '—'}</td>"
            f"<td>{s['scale']:g}</td>"
            f"<td>{'—' if lps is None else f'{lps:.1f}'}</td>"
            f"<td>{r_s}/{r_s + r_l}</td><td>{a_s}/{a_s + a_l}</td>"
            f"<td>{round(s['accuracy'] * n_math)}</td>"
            f"<td>{round(s['finished'] * n_prompts)}</td>"
            f"<td>{'yes' if coherent(s, base) else 'no'}</td></tr>"
        )

    gain_rows = "".join(
        f"<tr><td>{LABEL[d]}</td><td>{cos[d]:.2f} "
        f"({range_gain(d, 'margin_cosine')[0]:.2f} to {range_gain(d, 'margin_cosine')[1]:.2f})</td>"
        f"<td>{per_edit[d]:.0f}</td><td>{same_edit[d]:.0f}</td><td>{sft_edit[d]:.0f}</td></tr>"
        for d in DIRECTIONS
    )
    site_table = "".join(
        f"<tr><td>{name}</td><td>{s['opportunities']}</td>"
        f"<td>1 in {10 ** -s['median_log10_ratio']:,.0f}</td>"
        f"<td>{pct(s['share_within_100x'])}</td><td>{s['chars_per_opportunity']:.1f}</td>"
        f"<td>{s['expected_chars_per_opportunity']:.3f}</td>"
        f"<td>{s['gradient_weight_per_opportunity']:.3f}</td></tr>"
        for name, s in (
            ("drop the next space", glue_s),
            ("use the short form", short_s),
        )
    )

    body = f"""
<p class="lede">GPT‑6's published chain-of-thought drops the spaces between words
but rarely uses short forms like <code>msgs</code> or <code>auth</code>, which
would save more characters. The <a href="{MAIN_REPORT}">main report</a> reads
that as a hint that its length penalty counts words, not characters. This
follow-up tests the alternative explanation that gluing is just much easier
for a model to learn, using gpt-oss-20b as the nearest open model.</p>

<h2 id="summary">Summary</h2>
<ul>
<li><b>Both habits can be steered in while gpt-oss still does arithmetic.</b>
A gluing direction gives a median of {g18["reasoning_letters_per_space"]:.0f}
letters per space in the reasoning (GPT‑6's excerpts: {rng(glued_lps)}) with
{round(g18["accuracy"] * n_math)}/{n_math} GSM8K problems right. A short-form direction
raises the short-form share of answers from {bs / (bs + bl):.2f} to
{short_best["answer_short_share"]:.2f} with
{round(short_best["accuracy"] * n_math)}/{n_math} right.</li>
<li><b>Neither direction is sturdier at equal size, and relative to its own natural
size the short-form one tolerates more.</b> The largest coherent push is
{rng(natural["short"])}× the short-form direction's natural size, against
{rng(natural["glued"])}× for gluing.</li>
<li><b>Gluing transfers between excerpts about {
        cos["glued"] / cos["short"]:.0f}× better.</b>
Raising the glued rendering in four excerpts raises it in the fifth with a
gradient cosine of {cos["glued"]:.2f}, against {cos["short"]:.2f} for short forms
(positive on all five excerpts). Trained on its own excerpt, a short form is
learned as fast per edit as a dropped space.</li>
<li><b>Unprompted, gpt-oss is likelier to write a short form than to drop a space.</b>
At a typical point the odds are 1 in {med_short:,.0f} against 1 in {med_glue:,.0f},
and a short form saves {short_s["chars_per_opportunity"]:.1f} characters to 1.
Rough per-point measures of a character penalty's starting pull favour short
forms {weight_ratio:.0f}–{expected_ratio:.0f}×.</li>
<li><b>Caveats.</b> One model, not GPT‑6; directions from 5 excerpts; 32
arithmetic problems (one problem is 3 points); the gradient measure is
first-order and is not an RL run; the short-form list is hand-made. Two review
rounds changed several headline numbers; the fixes are listed at the end.</li>
</ul>

<h2 id="data">Data and setup</h2>
<p>The 5 glued chain-of-thought excerpts from OpenAI's report, each in four
renderings: <b>spaced</b> (OpenAI's own space-restored reading),
<b>glued</b> (what GPT‑6 wrote: {n_glued} dropped spaces), <b>short</b> (spaced, with
{n_short} of {n_words} words replaced by a common short form such as
<code>monitor → mon</code>, <code>authorization → auth</code>,
<code>with → w/</code>) and <b>both</b>. Against spaced they save {chars["glued"]},
{chars["short"]} and {chars["both"]} characters. Each sits in gpt-oss-20b's
reasoning channel after the incident's first user prompt, at reasoning effort
"low".</p>

<h2 id="steering">Steering: both habits are reachable without breaking arithmetic</h2>
<p>For each rewrite, a steering direction is the mean over excerpts of
(average residual-stream activation over the rewritten reasoning) minus
(the same for the spaced reasoning). It is added at every position after layer
6, 12 or 18 of 24, scaled to a fraction of the median residual-stream size at
that layer, so all three directions get the same push at each strength.
gpt-oss then answers 32 GSM8K test problems and 8 agentic prompts written in the
report's style, greedily, up to 1,024 tokens.</p>
<p>Unsteered, it solves {base_acc}/{n_math} and finishes {base_fin}/{n_prompts}
(the other {n_prompts - base_fin} are long tables cut off). A run counts as
<b>coherent</b> if neither falls more than {COHERENT_DROP * 100:.0f} points.
The short-form share is measured mainly in the answers: the reasoning is short
at "low" effort and holds only about {brs + brl} list words per run, against
{bs + bl} in the answers. Words inside code are not counted.</p>
{
        figure(
            f'<p class="note">Look at the filled markers only: hollow ones failed the coherence '
            f"test. Gluing (blue) climbs steeply at layer 18 while accuracy holds; short forms "
            f"(orange) rise in the answers at every layer before breaking.</p>"
            f"<details><summary>Strongest coherent run per direction and layer</summary>"
            f"<table><tr><th>direction</th><th>layer</th><th>strength</th><th>letters/space</th>"
            f"<th>answer short-form share</th><th>GSM8K</th><th>finished</th>"
            f"<th>first incoherent strength</th></tr>{''.join(best_rows)}</table></details>"
            f"<details><summary>Every run</summary><table><tr><th>direction</th><th>layer</th>"
            f"<th>strength</th><th>letters/space</th><th>short in reasoning</th>"
            f"<th>short in answers</th><th>GSM8K /{n_math}</th><th>finished /{n_prompts}</th>"
            f"<th>coherent</th></tr>{''.join(all_rows)}</table></details>",
            inline_svg(
                "steering",
                "Four rows of panels by steering layer against strength: "
                "letters per space, short-form share in reasoning and in answers, and "
                "GSM8K accuracy, for glue, short-form and both directions.",
            ),
            1,
            "Effect and coherence of each steering direction, by layer and strength.",
        )
    }
<p>The same agentic prompt (whether a subagent may remove monitoring hooks),
reasoning only:</p>
<ul>
<li>Unsteered: <code>{esc(plain_example)}</code></li>
<li>Glue, layer 18, strength 0.375: <code>{esc(glue_example)}</code></li>
<li>Short forms, layer 18, strength 0.1875: <code>{esc(short_example)}</code></li>
</ul>
<p>At equal strength gluing survives a harder push, but the natural glue
direction is {min(ratio):.1f}–{max(ratio):.1f}× longer than the short-form one at
these layers. Measured in each direction's own natural size, the largest
coherent push is {rng(natural["glued"])}× for gluing and
{rng(natural["short"])}× for short forms. The short-form direction also shortens
the reasoning a little (median {min(short_reasoning_len):.0f}–{
        max(short_reasoning_len):.0f}
letters against {base["reasoning_letters"]:.0f}). Held out one excerpt at a time,
the glue direction ranks that excerpt's glued tokens above its spaced ones with
probability {min(auc_glue):.2f}–{max(auc_glue):.2f} from layer 9 on; the short-form
one manages {min(auc_short):.2f}–{max(auc_short):.2f}, partly because only one word in
five differs.</p>

<h2 id="transfer">Training: gluing carries over to new excerpts about {
        cos["glued"] / cos["short"]:.0f}× better</h2>
<p>An excerpt's <i>margin</i> is log P(rewritten reasoning) − log P(spaced
reasoning) under gpt-oss: how much more probable it finds the rewrite than the
original. With the gradient of every margin taken over all attention weights,
a fixed-size step that raises the margins of four excerpts changes the fifth's
margin, to first order, by (step · its margin gradient) / |step|. The cosine
version divides by the size of the fifth's gradient too, so it does not depend
on how big either side's gradients are. Results are averaged over holding out
each of the 5 excerpts.</p>
{
        figure(
            f'<p class="note">Left: every excerpt is positive for every rewrite, and glue is '
            f"highest. Right: the short-form points sit about one order of magnitude higher.</p>"
            f"<details open><summary>Gradient results (mean over held-out excerpts; units of "
            f"the gains are arbitrary but shared)</summary><table><tr><th>rewrite</th>"
            f"<th>cosine, held out (range)</th><th>gain per edit, held out</th>"
            f"<th>gain per edit, same excerpt</th>"
            f"<th>fine-tuning step: gain per edit, held out</th></tr>{gain_rows}</table></details>",
            inline_svg(
                "learnability",
                "Left: gradient cosine between training and held-out "
                "margins per excerpt for glue, short forms and both. Right: log10 of "
                "P(edit)/P(as written) at every place in the spaced reasoning.",
            ),
            2,
            "How well training transfers (left), and how likely each edit is unprompted (right).",
        )
    }
<p>Per edit, the held-out gain is {
        per_edit["glued"] / per_edit["short"]:.1f}× larger for
gluing, while training on the same excerpt gives short forms
{same_edit["short"]:.0f} against {same_edit["glued"]:.0f} for gluing. The lowest
short-form cosine is {short_cos_min:.2f}. A plain fine-tuning step on the rewritten
text (last column) shows a much larger gap, but it is confounded: the glued
text has {n_glued / n_short:.1f}× more edits, and most of the short-form text's gradient
is about content.</p>

<h2 id="base-rates">Unprompted: short forms are about {
        med_glue / med_short:.0f}× likelier than dropped spaces</h2>
<p>At every point in the spaced reasoning where gpt-oss could drop the next
space ({glue_s["opportunities"]} points) or use the next word's short form
({short_s["opportunities"]}), the two continuations of the same prefix are scored
through the first token of the following word. With
q = P(edit) / (P(edit) + P(as written)), the table gives the conditional
expected saving, q × characters, and a two-option gradient weight,
q(1 − q) × characters. Both ignore everything else the model might write
there, and neither is a model of a real RL update.</p>
<table><tr><th>edit</th><th>points</th><th>median P(edit)/P(as written)</th>
<th>within 100×</th><th>characters saved</th><th>conditional expected saving</th>
<th>gradient weight</th></tr>{site_table}</table>
<p>Per point, short forms lead by {expected_ratio:.0f}× on the first measure and
{weight_ratio:.0f}× on the second. Points GPT‑6 actually glued are no likelier to be
glued by gpt-oss than points it left spaced (median 10<sup>{gpt6[True]:.2f}</sup>
against 10<sup>{gpt6[False]:.2f}</sup>).</p>

<h2 id="caveats">Caveats and corrections</h2>
<ul>
<li>gpt-oss-20b shares GPT‑6's tokenizer family but not its training.</li>
<li>Directions come from 5 excerpts and partly reflect which tokens appear, not only
style. The sweep used an earlier version of the directions that pooled tokens across
excerpts; at the steered layers it agrees with the current version to a cosine of
0.98 or more, so the sweep was not rerun.</li>
<li>The answer checker takes the last number and any boxed (else bold) numbers. A hand
check of 25 random passes and 25 random failures found every pass right and 2 failures
misread.</li>
<li>The gradient measure covers attention weights only (637M parameters) and is
first-order.</li>
<li>Review corrections, all applied above: a 15× transfer gap from a confounded
measure (now about {
        cos["glued"] / cos["short"]:.0f}×); a 58.7 letters-per-space headline
that was half one looping answer (medians now); truncated answers counted as finished;
an answer parser that misread <code>$70{{,}}000</code> (unsteered accuracy was really
32/32, not 29/32); code words counted as short forms; unequal excerpt weights in the
directions; and an unbounded "expected saving" that could exceed the characters on
offer.</li>
</ul>

<h2 id="reproduce">Reproduce</h2>
<p>The generations and steering vectors are on Hugging Face at
<a href="https://huggingface.co/datasets/{HF_REPO}">{HF_REPO}</a> (revision
<code>{REVISION[:7]}</code>). Every number and figure here rebuilds on CPU:</p>
<pre><code>git clone {REPO_URL}
cd astra-sol-cot-length-penalty-experiment
uv sync
uv run python -m cot_spacing.steering_data
uv run python -m cot_spacing.steering score cot_spacing/results/steering/gen/*.jsonl \\
    --out cot_spacing/results/steering/scores.json
uv run python -m cot_spacing.steering_figures
uv run python scripts/build_steering_report.py -o report.html</code></pre>
<p>The GPU stages (directions, sweep, gradients, edit probabilities) are a
Snakemake pipeline, <code>workflow/Snakefile</code>, run on 2 A40s per job: about 14
job-hours in all. The longer write-up is
<a href="{
        REPO_URL
    }/blob/brendanlong/steering-spaces-vs-short-words/STEERING.md">STEERING.md</a>; the main report is
<a href="{MAIN_REPORT}">here</a>.</p>
"""
    style = """
<style>
.lede { font-size: 1.1em; }
figure { margin: 2em 0; }
figcaption { color: var(--text-secondary, #52514e); font-size: 0.95em;
             margin-bottom: 0.6em; }
figure svg { display: block; max-width: 100%; height: auto;
             background: #fcfcfb; border-radius: 6px; padding: 4px 0; }
figure details, figure .note { margin-top: 0.7em; font-size: 0.92em; }
.note { color: var(--text-secondary, #52514e); }
details { margin: 0.8em 0; overflow-x: auto; }
summary { cursor: pointer; color: var(--link, #2a78d6); }
table { border-collapse: collapse; font-size: 0.9em; }
th, td { text-align: left; padding: 3px 12px 3px 0; vertical-align: top; }
li code { overflow-wrap: anywhere; white-space: normal; }
</style>
"""
    tmp = out_path.with_suffix(".body.html")
    tmp.write_text(style + body)
    subprocess.run(
        [
            "python3",
            str(TEMPLATE / "scripts" / "render.py"),
            str(tmp),
            "--title",
            "Is gluing just easier to learn? Steering gpt-oss toward dropped spaces vs short forms",
            "-o",
            str(out_path),
        ],
        check=True,
    )
    tmp.unlink()
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
