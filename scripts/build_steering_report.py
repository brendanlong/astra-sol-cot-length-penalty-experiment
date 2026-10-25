"""Build the HTML report on steering gpt-oss toward gluing vs short forms.

Every number in the prose is read from the result files under
``cot_spacing/results`` and the generations on Hugging Face (downloaded by
``cot_spacing.steering_data`` if missing), so the text cannot drift from the
data. The figures are the SVGs ``cot_spacing.steering_figures`` writes.

    uv run python scripts/build_steering_report.py -o /path/to/report.html
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch
from build_report import TEMPLATE, esc, figure
from build_report import inline_svg as _inline_svg
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cot_spacing.analysis import glued_only, segmentations
from cot_spacing.data import load_excerpts
from cot_spacing.gpt_oss import MODEL
from cot_spacing.learnability import REWRITES, site_summary, step_gains
from cot_spacing.steering import ANALYSIS, COHERENT_DROP
from cot_spacing.steering_data import REPO as HF_REPO
from cot_spacing.steering_data import REVISION, STEERING_DIR, download
from cot_spacing.steering_figures import LABEL

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
DIRECTIONS = ("glued", "glued_wordwise", "nospace", "short", "both")
#: Runs compared in the spacing table: the first six remove about half the
#: spaces between words, the last two go one step further.
SPACING_RUNS = (
    ("glued", 12, 0.25),
    ("nospace", 12, 0.25),
    ("glued_wordwise", 12, 0.25),
    ("glued", 18, 0.25),
    ("nospace", 18, 0.25),
    ("glued_wordwise", 18, 0.25),
    ("glued", 18, 0.375),
    ("nospace", 18, 0.375),
)
#: Run whose non-canonical tokens are shown as examples.
CANONICAL_EXAMPLE_RUN = "glued-18-0.25"
N_MATH = 32
N_PROMPTS = 40


def inline_svg(name: str, alt: str) -> str:
    """Inline a figure, with its ids prefixed so two figures cannot collide."""
    svg = _inline_svg(name, alt)
    svg = re.sub(r'id="([^"]+)"', rf'id="{name}-\1"', svg)
    return re.sub(r"(url\(#|href=\"#)([^)\"]+)", rf"\1{name}-\2", svg)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def frac(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def records(name: str) -> list[dict[str, Any]]:
    path = STEERING_DIR / "gen" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def reasoning(name: str, prompt_id: str) -> str:
    r = next(r for r in records(name) if r["id"] == prompt_id)
    m = ANALYSIS.search(r["text"])
    return m.group(1).strip() if m else ""


def divergences(
    name: str, tok: PreTrainedTokenizerBase
) -> list[tuple[list[str], list[str]]]:
    """(canonical, generated) token strings wherever a run's output is not
    the canonical tokenization of its own text, letters only."""
    out = []
    for r in records(name):
        ids = r["ids"]
        canonical = tok.encode(tok.decode(ids), add_special_tokens=False)
        matcher = difflib.SequenceMatcher(a=canonical, b=ids, autojunk=False)
        for op, a1, a2, b1, b2 in matcher.get_opcodes():
            if op == "equal":
                continue
            c = [str(tok.decode([i])) for i in canonical[a1:a2]]
            g = [str(tok.decode([i])) for i in ids[b1:b2]]
            if all(re.search("[A-Za-z]", x) for x in c + g):
                out.append((c, g))
    return out


WORDS = frozenset(
    w.strip().lower()
    for w in Path("/usr/share/dict/words").read_text().splitlines()
    if w.strip().isalpha()
)


def is_word(token: str) -> bool:
    t = token.strip(" .,:;?!-'\"’").lower()
    return t.isalpha() and t in WORDS and (len(t) > 1 or t in ("a", "i"))


def dictionary_share(pairs: list[tuple[list[str], list[str]]]) -> tuple[float, float]:
    """Share of generated, and of canonical, tokens that are whole words."""
    gen = [t for _, g in pairs for t in g]
    can = [t for c, _ in pairs for t in c]
    return sum(map(is_word, gen)) / len(gen), sum(map(is_word, can)) / len(can)


def direction_cosines(a: str, b: str) -> list[float]:
    saved = torch.load(STEERING_DIR / "vectors.pt")["directions"]
    cos = torch.nn.functional.cosine_similarity(saved[a], saved[b], dim=-1)
    return [cos[layer].item() for layer in LAYERS]


def gpt6_spaces_kept() -> dict[str, tuple[int, int]]:
    """In GPT-6's glued excerpts: spaces kept after sentences, clauses, words."""
    kept = {"sent": [0, 0], "clause": [0, 0], "word": [0, 0]}
    for e, seg in glued_only(segmentations(load_excerpts())):
        if e.kind != "cot":
            continue
        for i, gap in enumerate(seg.gaps):
            if gap == "\n":
                continue
            w = seg.words[i]
            if re.search(r"[.?!]\W*$", w):
                kind = "sent"
            elif re.search(r"[,:;]\W*$", w):
                kind = "clause"
            else:
                kind = "word"
            kept[kind][0] += gap == " "
            kept[kind][1] += 1
    return {k: (a, n) for k, (a, n) in kept.items()}


def build(out_path: Path) -> None:
    if not (STEERING_DIR / "gen").exists():
        download()
    tok = AutoTokenizer.from_pretrained(MODEL)
    variants = json.loads((RESULTS / "variants.json").read_text())
    vectors = json.loads((RESULTS / "steering" / "vectors.json").read_text())
    summary = json.loads((RESULTS / "steering" / "scores.json").read_text())["summary"]
    gains = step_gains(RESULTS / "learnability" / "gradients.json")
    sites_path = RESULTS / "learnability" / "sites.json"
    sites = site_summary(sites_path)
    site_rows = json.loads(sites_path.read_text())["sites"]

    run = {(s["direction"], s["layer"], s["scale"]): s for s in summary}
    base = run["none", 0, 0.0]
    base_acc = round(base["accuracy"] * N_MATH)
    base_fin = round(base["finished"] * N_PROMPTS)
    bs, bl = base["answer_short_long"]
    brs, brl = base["reasoning_short_long"]
    word_per_letter = base["total_word_spaces"] / base["total_reasoning_letters"]

    def words_kept(s: dict[str, Any]) -> float:
        return s["total_word_spaces"] / s["total_reasoning_letters"] / word_per_letter

    glued_lps = [
        sum(map(str.isalpha, v["texts"]["glued"])) / len(v["texts"]["glued"].split())
        for v in variants
    ]
    n_glued = sum(v["glued_boundaries"] for v in variants)
    n_short = sum(v["shortened_words"] for v in variants)
    n_words = sum(len(v["texts"]["spaced"].split()) for v in variants)
    chars = {
        k: sum(v["chars_saved"][k] for v in variants)
        for k in ("glued", "short", "both", "nospace")
    }
    tokens = {
        k: sum(len(v["ids"][k]) for v in variants) for k in ("glued", "glued_wordwise")
    }
    tokens_differ = 0
    for v in variants:
        m = difflib.SequenceMatcher(
            a=v["ids"]["glued"], b=v["ids"]["glued_wordwise"], autojunk=False
        )
        matched = sum(b.size for b in m.get_matching_blocks())
        tokens_differ += len(v["ids"]["glued_wordwise"]) - matched

    def strongest(d: str, layer: int) -> dict[str, Any] | None:
        ok = sorted(
            (
                s
                for s in summary
                if s["direction"] == d and s["layer"] == layer and s["coherent"]
            ),
            key=lambda s: s["scale"],
        )
        return ok[-1] if ok else None

    best = {(d, layer): strongest(d, layer) for d in DIRECTIONS for layer in LAYERS}

    def top(d: str, key: str) -> dict[str, Any]:
        return max(
            (s for layer in LAYERS if (s := best[d, layer]) is not None),
            key=lambda s: s[key] or 0,
        )

    glue_top = top("glued", "reasoning_letters_per_space")
    ww_top = top("glued_wordwise", "reasoning_letters_per_space")
    short_top = top("short", "answer_short_share")

    norm = vectors["direction_norm"]
    resid = vectors["resid_norm_median"]

    def natural(d: str) -> list[float]:
        return [
            s["scale"] * resid[layer] / norm[d][layer]
            for layer in LAYERS
            if (s := best[d, layer]) is not None
        ]

    ratio = [norm["glued"][layer] / norm["short"][layer] for layer in LAYERS]
    cos_ww = direction_cosines("glued", "glued_wordwise")
    auc = vectors["held_out_auc"]
    first_broken = min(
        s["scale"] for s in summary if s["direction"] != "none" and not s["coherent"]
    )

    half = [run[k] for k in SPACING_RUNS[:6]]
    sentence_half = [
        s["total_sentence_spaces"] / s["total_sentence_ends"] for s in half
    ]
    far = [run[k] for k in SPACING_RUNS[6:]]
    sentence_far = [s["total_sentence_spaces"] / s["total_sentence_ends"] for s in far]
    noncanon_18 = [
        run[d, 18, sc]["noncanonical_tokens"] / run[d, 18, sc]["tokens"]
        for d in ("glued", "glued_wordwise", "nospace")
        for sc in (0.25, 0.375)
    ]
    shares = [
        dictionary_share(divergences(f"{d}-{layer}-{sc}", tok))
        for d, layer, sc in SPACING_RUNS
    ]
    gen_share = [g for g, _ in shares]
    can_share = [c for _, c in shares]
    examples = [
        (c, g)
        for c, g in divergences(CANONICAL_EXAMPLE_RUN, tok)
        if len(g) >= 2
        and all(is_word(t) and len(t.strip()) > 1 for t in g)
        and not all(map(is_word, c))
    ][:6]
    gk = {k: frac(*v) for k, v in gpt6_spaces_kept().items()}
    gpt6_sentences = gpt6_spaces_kept()["sent"][1]

    def mean_gain(d: str, key: str) -> float:
        return statistics.mean(float(r[key]) for r in gains if r["direction"] == d)  # type: ignore[arg-type]

    def range_gain(d: str, key: str) -> tuple[float, float]:
        xs = [float(r[key]) for r in gains if r["direction"] == d]  # type: ignore[arg-type]
        return min(xs), max(xs)

    cos = {d: mean_gain(d, "margin_cosine") for d in REWRITES}
    per_edit = {d: mean_gain(d, "held_out_margin_per_edit") for d in REWRITES}
    same_edit = {d: mean_gain(d, "same_margin_per_edit") for d in REWRITES}
    sft_edit = {d: mean_gain(d, "held_out_sft_per_edit") for d in REWRITES}
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
    gpt6_site = {
        flag: statistics.median(
            r["log_ratio"] / math.log(10)
            for r in site_rows
            if r["kind"] == "glue" and r["gpt6_glued"] == flag
        )
        for flag in (True, False)
    }

    def rng(xs: list[float], digits: int = 1) -> str:
        return f"{min(xs):.{digits}f}–{max(xs):.{digits}f}"

    def prng(xs: list[float]) -> str:
        return f"{min(xs):.0%}–{max(xs):.0%}"

    def where(s: dict[str, Any]) -> str:
        return f"layer {s['layer']}, strength {s['scale']:g}"

    best_rows = []
    for d in DIRECTIONS:
        for layer in LAYERS:
            s = best[d, layer]
            if s is None:
                best_rows.append(
                    f"<tr><td>{LABEL[d]}</td><td>{layer}</td>"
                    "<td colspan=5>no coherent run</td></tr>"
                )
                continue
            a_s, a_l = s["answer_short_long"]
            share = "—" if s["answer_short_share"] is None else frac(a_s, a_s + a_l)
            best_rows.append(
                f"<tr><td>{LABEL[d]}</td><td>{layer}</td><td>{s['scale']:g}</td>"
                f"<td>{s['reasoning_letters_per_space']:.1f}</td><td>{share}</td>"
                f"<td>{round(s['accuracy'] * N_MATH)}/{N_MATH}</td>"
                f"<td>{round(s['finished'] * N_PROMPTS)}/{N_PROMPTS}</td></tr>"
            )

    all_rows = []
    for s in summary:
        a_s, a_l = s["answer_short_long"]
        lps = s["reasoning_letters_per_space"]
        nc = f"{s['noncanonical_tokens'] / s['tokens']:.1%}" if s.get("tokens") else "—"
        all_rows.append(
            f"<tr><td>{LABEL.get(s['direction'], 'unsteered')}</td>"
            f"<td>{s['layer'] if s['direction'] != 'none' else '—'}</td>"
            f"<td>{s['scale']:g}</td>"
            f"<td>{'—' if lps is None else f'{lps:.1f}'}</td>"
            f"<td>{frac(s['total_sentence_spaces'], s['total_sentence_ends'])}</td>"
            f"<td>{pct(words_kept(s)) if s['total_reasoning_letters'] else '—'}</td>"
            f"<td>{a_s}/{a_s + a_l}</td><td>{nc}</td>"
            f"<td>{round(s['accuracy'] * N_MATH)}</td>"
            f"<td>{round(s['finished'] * N_PROMPTS)}</td>"
            f"<td>{s['total_degenerate']}</td>"
            f"<td>{'yes' if s['coherent'] else 'no'}</td></tr>"
        )

    spacing_rows = [
        "<tr><td>unsteered</td>"
        f"<td>{frac(base['total_sentence_spaces'], base['total_sentence_ends'])}</td>"
        f"<td>{frac(base['total_clause_spaces'], base['total_clause_ends'])}</td>"
        f"<td>100%</td><td>{base['noncanonical_tokens']} of {base['tokens']:,}</td>"
        "<td>—</td></tr>"
    ]
    for (d, layer, sc), (g, c) in zip(SPACING_RUNS, shares, strict=True):
        s = run[d, layer, sc]
        spacing_rows.append(
            f"<tr><td>{LABEL[d]}, {where(s)}</td>"
            f"<td>{frac(s['total_sentence_spaces'], s['total_sentence_ends'])}</td>"
            f"<td>{frac(s['total_clause_spaces'], s['total_clause_ends'])}</td>"
            f"<td>{pct(words_kept(s))}</td>"
            f"<td>{s['noncanonical_tokens'] / s['tokens']:.1%}</td>"
            f"<td>{g:.0%} / {c:.0%}</td></tr>"
        )
    example_html = "".join(
        f"<li>generated <code>{esc('|'.join(t.strip() for t in g))}</code>, "
        f"canonical <code>{esc('|'.join(t.strip() for t in c))}</code></li>"
        for c, g in examples
    )

    gain_rows = "".join(
        f"<tr><td>{LABEL[d]}</td><td>{cos[d]:.2f} "
        f"({range_gain(d, 'margin_cosine')[0]:.2f} to "
        f"{range_gain(d, 'margin_cosine')[1]:.2f})</td>"
        f"<td>{per_edit[d]:.0f}</td><td>{same_edit[d]:.0f}</td>"
        f"<td>{sft_edit[d]:.0f}</td></tr>"
        for d in REWRITES
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

    fig_steering = figure(
        '<p class="note">Filled markers are coherent runs. Gluing (blue) climbs with '
        "strength at every layer; short forms (orange) rise in the answers before "
        "breaking.</p>"
        "<details><summary>Strongest coherent run per direction and layer</summary>"
        "<table><tr><th>direction</th><th>layer</th><th>strength</th>"
        "<th>letters/space</th><th>answer short-form share</th><th>GSM8K</th>"
        f"<th>finished</th></tr>{''.join(best_rows)}</table></details>",
        inline_svg(
            "steering",
            "Four rows of panels by steering layer against strength: letters per "
            "space, short-form share in reasoning and in answers, and GSM8K accuracy, "
            "for the glue, short-form and both directions.",
        ),
        1,
        "Effect and coherence of the glue, short-form and both directions.",
    )
    fig_spaces = figure(
        '<p class="note">Top row: sentence spaces stay near 100% while the second '
        "row's word spaces fall to half, for all three directions. Third row: "
        "unsteered output is fully canonical; steered gluing is not.</p>",
        inline_svg(
            "steering_spaces",
            "Four rows of panels by steering layer for the glue, no-space and "
            "word-by-word glue directions: share of sentence spaces kept, share of "
            "word spaces kept, share of non-canonical generated tokens, and GSM8K "
            "accuracy.",
        ),
        2,
        "Which spaces survive, and how canonical the output is, for three glue "
        "directions.",
    )
    fig_learn = figure(
        '<p class="note">Left: every excerpt is positive for every rewrite, and glue '
        "is highest. Right: the short-form points sit about one order of magnitude "
        "higher.</p>"
        "<details open><summary>Gradient results (mean over held-out excerpts; units "
        "of the gains are arbitrary but shared)</summary><table><tr><th>rewrite</th>"
        "<th>cosine, held out (range)</th><th>gain per edit, held out</th>"
        "<th>gain per edit, same excerpt</th>"
        "<th>fine-tuning step: gain per edit, held out</th></tr>"
        f"{gain_rows}</table></details>",
        inline_svg(
            "learnability",
            "Left: gradient cosine between training and held-out margins per excerpt "
            "for glue, short forms and both. Right: log10 of P(edit)/P(as written) at "
            "every place in the spaced reasoning.",
        ),
        3,
        "How well training transfers (left), and how likely each edit is unprompted "
        "(right).",
    )

    body = f"""
<p class="lede">GPT‑6's published chain-of-thought drops the spaces between words
but rarely uses short forms like <code>msgs</code> or <code>auth</code>, which
would save more characters. The <a href="{MAIN_REPORT}">main report</a> reads
that as a hint that its length penalty counts words, not characters. This
follow-up tests the alternative explanation that gluing is just much easier
for a model to learn, using gpt-oss-20b as the nearest open model, and looks
at what steered gluing actually looks like.</p>

<h2 id="summary">Summary</h2>
<ul>
<li><b>Both habits can be steered in, with little room before gpt-oss breaks.</b>
A run counts as coherent only if no answer loops or breaks and accuracy holds.
The strongest coherent gluing is a median of
{ww_top["reasoning_letters_per_space"]:.0f} letters per space in the reasoning
({where(ww_top)}; unsteered {base["reasoning_letters_per_space"]:.1f}; GPT‑6's
excerpts {rng(glued_lps)}). A short-form direction raises the short-form share of
answers from {bs / (bs + bl):.2f} to {short_top["answer_short_share"]:.2f} with
{round(short_top["accuracy"] * N_MATH)}/{N_MATH} right. Relative to each
direction's natural size, the short-form one tolerates the larger push
({rng(natural("short"))}× against {rng(natural("glued"))}×).</li>
<li><b>Steered gluing keeps the spaces after sentences.</b> Where about half the
spaces between words are gone, {prng(sentence_half)} of spaces after sentences
remain, much as GPT‑6 kept all {gpt6_sentences} of its own. A direction built from
text with every space removed does the same, so this is gpt-oss's own
preference, not copied from GPT‑6's examples.</li>
<li><b>Steered gluing is written a word at a time.</b> Unsteered, every generated
token is the canonical tokenization of the text. Steered to glue at layer 18,
{prng(noncanon_18)} of tokens are not, and {prng(gen_share)} of those are whole
dictionary words, against {prng(can_share)} of the canonical tokens for the same
characters. Building the direction from canonical or from word-by-word tokens
makes no visible difference.</li>
<li><b>Gluing transfers between excerpts about
{cos["glued"] / cos["short"]:.0f}× better.</b> Raising the glued rendering in four
excerpts raises it in the fifth with a gradient cosine of {cos["glued"]:.2f},
against {cos["short"]:.2f} for short forms. Trained on its own excerpt, a short
form is learned as fast per edit as a dropped space.</li>
<li><b>Unprompted, gpt-oss is likelier to write a short form than to drop a
space.</b> At a typical point the odds are 1 in {med_short:,.0f} against 1 in
{med_glue:,.0f}, and a short form saves {short_s["chars_per_opportunity"]:.1f}
characters to 1. Rough per-point measures of a character penalty's starting pull
favour short forms {weight_ratio:.0f}–{expected_ratio:.0f}×.</li>
<li><b>Caveats.</b> One model, not GPT‑6; directions from 5 excerpts; short
rollouts (reasoning effort "low", at most 1,024 tokens); 32 arithmetic problems
(one is 3 points); the gradient measure is first-order and is not an RL run; the
short-form list is hand-made. Review rounds changed several headline numbers;
the corrections are listed at the end.</li>
</ul>

<h2 id="data">Data and setup</h2>
<p>The 5 glued chain-of-thought excerpts from OpenAI's report, in these
renderings:</p>
<table>
<tr><th>rendering</th><th>what it is</th><th>characters saved</th></tr>
<tr><td>spaced</td><td>OpenAI's own space-restored reading; the baseline</td><td>—</td></tr>
<tr><td>glue</td><td>what GPT‑6 wrote ({n_glued} dropped spaces)</td><td>{chars["glued"]}</td></tr>
<tr><td>short forms</td><td>spaced, with {n_short} of {n_words} words replaced by a common
short form (<code>monitor → mon</code>, <code>authorization → auth</code>,
<code>with → w/</code>)</td><td>{chars["short"]}</td></tr>
<tr><td>both</td><td>glued and shortened</td><td>{chars["both"]}</td></tr>
<tr><td>no spaces at all</td><td>what GPT‑6 wrote, also without the spaces it kept after
sentences and clauses</td><td>{chars["nospace"]}</td></tr>
<tr><td>glue, word-by-word tokens</td><td>the glued characters, each word tokenized on its
own and the tokens joined: {tokens["glued_wordwise"]} tokens against
{tokens["glued"]} canonical, {tokens_differ} of them different</td><td>{chars["glued"]}</td></tr>
</table>
<p>Each sits in gpt-oss-20b's reasoning channel after the incident's first user
prompt, at reasoning effort "low".</p>

<h2 id="steering">Steering: both habits are reachable, with little margin</h2>
<p>For each rendering, a steering direction is the mean over excerpts of
(average residual-stream activation over that rendering's reasoning) minus
(the same for the spaced reasoning), read at each decoder block's output. It is
added at every position after layer 6, 12 or 18 of 24, scaled to a fraction of
the median residual-stream size at that layer, so all directions get the same
push at each strength. gpt-oss then answers 32 GSM8K test problems and 8 agentic
prompts written in the report's style, greedily, up to 1,024 tokens, and its token
ids are saved.</p>
<p>Unsteered, it solves {base_acc}/{N_MATH}, finishes {base_fin}/{N_PROMPTS} (the
other {N_PROMPTS - base_fin} are long tables cut off) and never loops or breaks. A
run is <b>coherent</b> if no answer loops or breaks (a third of its 30-character
windows repeating, or undecodable bytes) and neither accuracy nor the share
finished falls more than {COHERENT_DROP * 100:.0f} points. Some flagged answers
recover and still answer correctly; one such answer is enough to make a run
incoherent, so coherence can flip between neighbouring strengths. The short-form
share is measured mainly in the answers: the reasoning holds only about
{brs + brl} list words per run, against {bs + bl} in the answers. Words inside code
are not counted.</p>
{fig_steering}
<p>The glue direction's strongest coherent run is
{glue_top["reasoning_letters_per_space"]:.1f} letters per space ({where(glue_top)});
the word-by-word version reaches {ww_top["reasoning_letters_per_space"]:.0f} at
{where(ww_top)}. The two directions agree to a cosine of {rng(cos_ww, 2)} at the
steered layers. The natural glue direction is {rng(ratio)}× longer than the
short-form one; in multiples of each direction's own natural size, the largest
coherent push is {rng(natural("glued"))}× for gluing and
{rng(natural("short"))}× for short forms. Some direction has an incoherent run
from strength {first_broken:g} up. Held out one excerpt at a time, the glue
direction ranks that excerpt's glued tokens above its spaced ones with
probability {rng(auc["glued"][9:], 2)} from layer 9 on; the short-form one
{rng(auc["short"], 2)}, partly because only one word in five differs.</p>
<p>The same agentic prompt (whether a subagent may remove monitoring hooks),
reasoning only:</p>
<ul>
<li>Unsteered: <code>{esc(reasoning("none", "agentic-4"))}</code></li>
<li>Glue, layer 18, strength 0.25:
<code>{esc(reasoning("glued-18-0.25", "agentic-4"))}</code></li>
<li>Glue, word-by-word tokens, layer 18, strength 0.375:
<code>{esc(reasoning("glued_wordwise-18-0.375", "agentic-4"))}</code></li>
<li>Short forms, layer 18, strength 0.125:
<code>{esc(reasoning("short-18-0.125", "agentic-4"))}</code></li>
</ul>

<h2 id="spaces">Steered gluing keeps sentence spaces and writes whole-word tokens</h2>
<p>In GPT‑6's glued excerpts, a space follows every sentence end ({gk["sent"]})
and every comma, colon or semicolon ({gk["clause"]}), but only {gk["word"]} of other
word boundaries. Two controls test where gpt-oss's own habits lie. The
<b>no-space</b> direction is built from text with every space removed: if steered
gluing kept sentence spaces only because GPT‑6's examples did, this direction
should drop them as readily as the rest. The <b>word-by-word</b> direction is built
from the same glued characters tokenized one word at a time: if building
directions from canonical tokens pushed gpt-oss toward odd tokenizations, the two
would differ.</p>
<p>In the reasoning, a sentence space is one after <code>.</code>, <code>?</code> or
<code>!</code> followed by more words; clause spaces are the same for
<code>,:;</code>; spaces between words are counted per letter, as a share of
unsteered gpt-oss's. A generated token is <i>non-canonical</i> if re-tokenizing the
decoded output does not reproduce it (the two sequences are aligned).</p>
{fig_spaces}
<table>
<tr><th>run</th><th>space after sentence</th><th>after clause</th><th>between words</th>
<th>non-canonical tokens</th><th>whole words among them: generated / canonical</th></tr>
{"".join(spacing_rows)}
</table>
<p>Where about half the word spaces are gone, {prng(sentence_half)} of sentence
spaces survive, with clause spaces in between; one step stronger at layer 18,
only {prng(sentence_far)} do. The no-space direction keeps sentence spaces about as
well as the glue direction. Where the output is not canonical, the generated
tokens are mostly whole words written without their leading space. Examples from
glue at layer 18, strength 0.25:</p>
<ul>{example_html}</ul>

<h2 id="transfer">Training: gluing carries over to new excerpts about
{cos["glued"] / cos["short"]:.0f}× better</h2>
<p>An excerpt's <i>margin</i> is log P(rewritten reasoning) − log P(spaced
reasoning) under gpt-oss: how much more probable it finds the rewrite than the
original. With the gradient of every margin taken over all attention weights, a
fixed-size step that raises the margins of four excerpts changes the fifth's
margin, to first order, by (step · its margin gradient) / |step|. The cosine
version divides by the size of the fifth's gradient too, so it does not depend on
how big either side's gradients are. Results are averaged over holding out each
of the 5 excerpts.</p>
{fig_learn}
<p>Per edit, the held-out gain is {per_edit["glued"] / per_edit["short"]:.1f}× larger
for gluing, while training on the same excerpt gives short forms
{same_edit["short"]:.0f} against {same_edit["glued"]:.0f} for gluing. The lowest
short-form cosine is {short_cos_min:.2f}. A plain fine-tuning step on the rewritten
text (last column) shows a much larger gap, but it is confounded: the glued text
has {n_glued / n_short:.1f}× more edits, and most of the short-form text's gradient is
about content.</p>

<h2 id="base-rates">Unprompted: short forms are about {med_glue / med_short:.0f}×
likelier than dropped spaces</h2>
<p>At every point in the spaced reasoning where gpt-oss could drop the next space
({glue_s["opportunities"]} points) or use the next word's short form
({short_s["opportunities"]}), the two continuations of the same prefix are scored
through the first token of the following word. With
q = P(edit) / (P(edit) + P(as written)), the table gives the conditional expected
saving, q × characters, and a two-option gradient weight, q(1 − q) × characters.
Both ignore everything else the model might write there, and neither is a model
of a real RL update.</p>
<table><tr><th>edit</th><th>points</th><th>median P(edit)/P(as written)</th>
<th>within 100×</th><th>characters saved</th><th>conditional expected saving</th>
<th>gradient weight</th></tr>{site_table}</table>
<p>Per point, short forms lead by {expected_ratio:.0f}× on the first measure and
{weight_ratio:.0f}× on the second. Points GPT‑6 actually glued are no likelier to be
glued by gpt-oss than points it left spaced (median
10<sup>{gpt6_site[True]:.2f}</sup> against 10<sup>{gpt6_site[False]:.2f}</sup>).</p>

<details><summary>Every steering run</summary>
<table><tr><th>direction</th><th>layer</th><th>strength</th><th>letters/space</th>
<th>sentence spaces</th><th>word spaces</th><th>short in answers</th>
<th>non-canonical</th><th>GSM8K /{N_MATH}</th><th>finished /{N_PROMPTS}</th>
<th>broken</th><th>coherent</th></tr>{"".join(all_rows)}</table></details>

<h2 id="caveats">Caveats and corrections</h2>
<ul>
<li>gpt-oss-20b shares GPT‑6's tokenizer family but not its training.</li>
<li>Directions come from 5 excerpts and partly reflect which tokens appear, not
only style.</li>
<li>Rollouts are short, so coherence over long reasoning is untested.</li>
<li>The answer checker takes the last number and any boxed (else bold) numbers. A
hand check of 25 random passes and 25 random failures (on an earlier sweep) found
every pass right and 2 failures misread.</li>
<li>Spaces between words are estimated per letter, since glued text does not show
where its words end.</li>
<li>The gradient measure covers attention weights only (637M parameters) and is
first-order.</li>
<li>Review corrections, all applied above: a 15× transfer gap from a confounded
measure (now about {cos["glued"] / cos["short"]:.0f}×); a 58.7 letters-per-space
headline that was half one looping answer (medians now); truncated answers
counted as finished; an answer parser that misread <code>$70{{,}}000</code>
(unsteered accuracy was really 32/32, not 29/32); code words counted as short
forms; unequal excerpt weights in the directions; an unbounded "expected
saving"; and a coherence test that let looping answers through. The sweep was
then rerun in full with the corrected directions.</li>
</ul>

<h2 id="reproduce">Reproduce</h2>
<p>The generations (with token ids) and steering vectors are on Hugging Face at
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
Snakemake pipeline, <code>workflow/Snakefile</code>, run on 2 A40s per job: about 15
job-hours in all. The longer write-up is
<a href="{REPO_URL}/blob/brendanlong/steering-spaces-vs-short-words/STEERING.md">STEERING.md</a>;
the main report is <a href="{MAIN_REPORT}">here</a>.</p>
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
table { border-collapse: collapse; font-size: 0.9em; display: block;
        overflow-x: auto; }
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
            "Is gluing just easier to learn? Steering gpt-oss toward dropped spaces "
            "vs short forms",
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
