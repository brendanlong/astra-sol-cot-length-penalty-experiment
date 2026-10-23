"""Build the HTML report from the result files.

Every number in the prose and every mark in the figures is read from
``cot_spacing/results/*.json`` — the run recorded in RESULTS.md — so the text
cannot drift from the data. Figures are inline SVG with literal light-theme
colours, so the page survives being saved by a reader that strips scripts and
stylesheets.

    uv run python scripts/build_report.py -o /path/to/report.html
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

import tiktoken

#: Published to one Surge site of its own; every update goes to this domain so
#: links and reader-saved copies stay valid.
PUBLISHED_AT = "https://gpt6-cot-spaces-cd652c60.surge.sh"
#: The repo this is built from, and the thread it is discussed in. Every page
#: links both ways so a reader landing on either can find the other.
REPO_URL = "https://github.com/brendanlong/astra-sol-cot-length-penalty-experiment"
DISCUSSION_URL = "https://www.lesswrong.com/posts/igEogGD9TAgAeAM7u/jimrandomh-s-shortform?commentId=su76ee7QNov6NqhBw"
#: Lion Reader article id for the saved copy, refreshed on every republish.
SAVED_ARTICLE_ID = "01a16091-299a-7db8-b77a-216504c89963"

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "cot_spacing" / "results"

# Palette: the dataviz skill's documented instance, as used by figures.py.
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DOWN = "#2a78d6"
UP = "#e34948"
SURFACE = "#fcfcfb"

TEMPLATE = (
    Path.home() / "personal-agent-skills/plugins/personal-agent-skills/skills/html-doc"
)


def ordinal(n: int) -> str:
    """1 -> 1st, 42 -> 42nd, 3602 -> 3,602nd."""
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n:,}{suffix}"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# Figure 1: the same characters, three token sequences
# --------------------------------------------------------------------------

#: A phrase the report actually contains, in excerpt eval_pipeline-04.
PHRASE_WORDS = ("not", "send", "messages")


def _token_row(
    enc: tiktoken.Encoding, pieces: list[str], x0: float, y: float, width: float
) -> tuple[str, int]:
    """One row of token boxes, laid out proportionally to character count."""
    total = sum(len(p) for p in pieces)
    out: list[str] = []
    x = x0
    for piece in pieces:
        w = width * len(piece) / total
        label = piece.replace(" ", "␣")
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w - 3:.1f}" height="30" rx="4" '
            f'fill="{SURFACE}" stroke="{AXIS}" stroke-width="1.5"/>'
            f'<text x="{x + (w - 3) / 2:.1f}" y="{y + 20}" text-anchor="middle" '
            f'font-family="ui-monospace,monospace" font-size="14" fill="{INK}">'
            f"{esc(label)}</text>"
        )
        x += w
    return "".join(out), len(pieces)


def figure_tokens(enc: tiktoken.Encoding) -> tuple[str, dict[str, int]]:
    from cot_spacing.segment import segment
    from cot_spacing.tokenize_cost import boundaries

    spaced = " ".join(PHRASE_WORDS)
    glued = "".join(PHRASE_WORDS)
    straddles = [
        b for b in boundaries("fig", segment(glued, spaced), enc) if b.straddled
    ]
    note = (
        f"<<STRADDLE>> spans the "
        f"{straddles[0].left}/{straddles[0].right} boundary: emitting it means "
        f"committing to the next word before finishing this one"
    )

    def decode(ids: list[int]) -> list[str]:
        return [enc.decode_single_token_bytes(t).decode() for t in ids]

    rows = [
        (
            "with spaces, as anyone would write it",
            decode(enc.encode(spaced)),
            None,
        ),
        (
            "glued, tokenized canonically — what the stored transcript re-encodes to",
            decode(enc.encode(glued)),
            note,
        ),
        (
            "glued, one word at a time — what the model most likely emitted",
            [t for w in PHRASE_WORDS for t in decode(enc.encode(w))],
            None,
        ),
    ]

    width, x0 = 560.0, 20.0
    parts = [
        '<svg viewBox="0 0 640 260" width="100%" role="img" '
        'aria-label="The same characters tokenized three ways. Spaced '
        "&quot;not send messages&quot; is 3 tokens. Glued "
        "&quot;notsendmessages&quot; encoded canonically is 4 tokens, two of "
        "which straddle word boundaries. The same glued string emitted one "
        'word at a time is 3 tokens that each match a word.">'
    ]
    counts: dict[str, int] = {}
    y = 34
    for label, pieces, note in rows:
        parts.append(
            f'<text x="{x0}" y="{y - 10}" font-size="13" fill="{SECONDARY}">'
            f"{esc(label)}</text>"
        )
        row_svg, n = _token_row(enc, pieces, x0, y, width)
        parts.append(row_svg)
        parts.append(
            f'<text x="{x0 + width + 14}" y="{y + 20}" font-size="14" '
            f'font-weight="600" fill="{INK}">{n}</text>'
        )
        if note:
            # Name the straddling chunk from the tokenization just drawn.
            straddler = next(p for p in pieces if p not in PHRASE_WORDS and len(p) > 1)
            text = esc(note).replace("&lt;&lt;STRADDLE&gt;&gt;", "")
            parts.append(
                f'<text x="{x0}" y="{y + 48}" font-size="12" fill="{MUTED}">'
                f'<tspan font-family="ui-monospace,monospace" fill="{INK}">'
                f"{esc(straddler)}</tspan>{text}</text>"
            )
        counts[label.split(",")[0]] = n
        y += 84 if note else 74
    parts.append(
        f'<text x="{x0 + width + 6}" y="20" font-size="12" fill="{MUTED}">tokens</text>'
    )
    parts.append("</svg>")
    return "".join(parts), counts


# --------------------------------------------------------------------------
# Figure 2: what gluing saves and costs
# --------------------------------------------------------------------------

FIGURES = REPO / "cot_spacing" / "figures"


def inline_svg(name: str, alt: str) -> str:
    """Inline a figure matplotlib already wrote, rather than redrawing it.

    One implementation of each chart, in ``cot_spacing/figures.py``, so the
    page and the repo cannot disagree. The width/height attributes are dropped
    so the viewBox scales the figure to the column.
    """
    svg = (FIGURES / f"{name}.svg").read_text()
    svg = svg[svg.index("<svg") :]
    svg = re.sub(r'\s(width|height)="[^"]*"', "", svg, count=2)
    svg = svg.replace(
        "<svg", f'<svg width="100%" role="img" aria-label="{esc(alt)}"', 1
    )
    return svg


# --------------------------------------------------------------------------
# Figure 3: where the two tokenizations disagree, which token is likelier
# --------------------------------------------------------------------------


def figure_ranks(divergences: list[dict[str, Any]], model: str) -> str:
    import math

    rows = sorted(
        (d for d in divergences if d["model"] == model and d["text_kind"] == "glued"),
        key=lambda d: d["rank_wordwise"] / max(d["rank_canonical"], 1),
    )
    w, left, right = 640, 60.0, 600.0
    top, height = 34.0, 300.0

    def x_of(rank: int) -> float:
        lo, hi = 0.0, math.log10(50000)
        return left + (right - left) * (math.log10(max(rank, 1)) - lo) / (hi - lo)

    parts = [
        f'<svg viewBox="0 0 {w} 400" width="100%" role="img" '
        f'aria-label="Dot plot of {len(rows)} positions where the two '
        f"tokenizations choose different next tokens. For most of them the "
        f"word-respecting token sits far to the left of the canonical token, "
        f'meaning the model ranks it much higher.">'
    ]
    for tick in (1, 10, 100, 1000, 10000):
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" '
            f'y2="{top + height}" stroke="{GRID}" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{top + height + 18}" text-anchor="middle" '
            f'font-size="11" fill="{MUTED}">{tick:,}</text>'
        )
    step = height / len(rows)
    for i, d in enumerate(rows):
        y = top + i * step + step / 2
        xc, xw = x_of(d["rank_canonical"]), x_of(d["rank_wordwise"])
        parts.append(
            f'<line x1="{min(xc, xw):.1f}" y1="{y:.1f}" x2="{max(xc, xw):.1f}" '
            f'y2="{y:.1f}" stroke="{GRID}" stroke-width="2"/>'
            f'<circle cx="{xc:.1f}" cy="{y:.1f}" r="4" fill="{UP}"/>'
            f'<circle cx="{xw:.1f}" cy="{y:.1f}" r="4" fill="{DOWN}"/>'
        )
    parts.append(
        f'<text x="{left}" y="20" font-size="13" fill="{DOWN}" '
        f'font-weight="600">word-respecting token</text>'
        f'<text x="{left + 190}" y="20" font-size="13" fill="{UP}" '
        f'font-weight="600">canonical token</text>'
        f'<text x="{(left + right) / 2:.0f}" y="{top + height + 40}" '
        f'text-anchor="middle" font-size="12" fill="{MUTED}">'
        f"rank in the model's next-token distribution (1 = its first choice)</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Figure 4: glued English against GPT-2 integers, three metrics
# --------------------------------------------------------------------------


def figure_chunking(chunking: dict[str, Any]) -> str:
    word = chunking["context_sweep"]
    three = chunking["gpt2_integer_context_sweep"]["3_digit"]
    four = chunking["gpt2_integer_context_sweep"]["4_digit"]
    metrics = [
        (
            "distinct chunkings|per item",
            [
                word["clipped"]["mean_distinct_patterns_per_item"],
                three["clipped"]["mean_distinct_patterns_per_item"],
                four["clipped"]["mean_distinct_patterns_per_item"],
            ],
            "{:.2f}",
        ),
        (
            "same, counting each|straddling token whole",
            [
                word["unclipped"]["mean_distinct_patterns_per_item"],
                three["unclipped"]["mean_distinct_patterns_per_item"],
                four["unclipped"]["mean_distinct_patterns_per_item"],
            ],
            "{:.2f}",
        ),
        (
            "% of pairs chunked|unlike the item alone",
            [
                100 * word["clipped"]["frac_pairs_differing_from_standalone"],
                100 * three["clipped"]["frac_pairs_differing_from_standalone"],
                100 * four["clipped"]["frac_pairs_differing_from_standalone"],
            ],
            "{:.1f}%",
        ),
    ]
    series = ("glued English", "3-digit integers", "4-digit integers")
    colors = (DOWN, "#9aa7b4", "#c7ccd2")
    panel_w, gap = 180.0, 30.0
    parts = [
        '<svg viewBox="0 0 640 230" width="100%" role="img" '
        'aria-label="Three grouped bar panels comparing glued English with '
        "GPT-2 integers. The three metrics do not agree on an ordering: "
        "glued English is between the two integer lengths on one, above both "
        'on another, and below 3-digit integers on the third.">'
    ]
    for pi, (title, values, fmt) in enumerate(metrics):
        x0 = 20 + pi * (panel_w + gap)
        top = 46.0
        hmax = 110.0
        vmax = max(values) * 1.25
        line1, line2 = title.split("|")
        parts.append(
            f'<text x="{x0}" y="{top - 30}" font-size="12" fill="{SECONDARY}">'
            f"{esc(line1)}"
            f'<tspan x="{x0}" dy="14">{esc(line2)}</tspan></text>'
        )
        bw = 38.0
        for bi, (val, color) in enumerate(zip(values, colors, strict=True)):
            h = hmax * val / vmax
            x = x0 + bi * (bw + 14)
            parts.append(
                f'<rect x="{x:.1f}" y="{top + hmax - h:.1f}" width="{bw}" '
                f'height="{h:.1f}" fill="{color}"/>'
                f'<text x="{x + bw / 2:.1f}" y="{top + hmax - h - 6:.1f}" '
                f'text-anchor="middle" font-size="12" font-weight="600" '
                f'fill="{INK}">{fmt.format(val)}</text>'
            )
        parts.append(
            f'<line x1="{x0 - 4}" y1="{top + hmax}" x2="{x0 + 3 * (bw + 14):.1f}" '
            f'y2="{top + hmax}" stroke="{AXIS}" stroke-width="1"/>'
        )
    for si, (name, color) in enumerate(zip(series, colors, strict=True)):
        x = 20 + si * 200
        parts.append(
            f'<rect x="{x}" y="200" width="12" height="12" fill="{color}"/>'
            f'<text x="{x + 18}" y="210" font-size="12" fill="{SECONDARY}">'
            f"{esc(name)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------


def figure(caption_html: str, svg: str, number: int, title: str) -> str:
    """A numbered figure. An empty ``title`` means it captions itself."""
    label = f"<b>Figure {number}.</b>{' ' + title if title else ''}"
    return f"<figure><figcaption>{label}</figcaption>{svg}{caption_html}</figure>"


def decision_table(points: list[dict[str, Any]]) -> str:
    """Every decision the lead model made, the boundary-crossing ones first."""
    rows = sorted(
        points, key=lambda d: (-d["canonical_crosses_word"], -d["advantage_wordwise"])
    )
    out = [
        "<table><tr><th>writing</th><th>after</th><th>canonical</th><th>rank</th>"
        "<th>word's own</th><th>rank</th><th>×more likely</th></tr>"
    ]
    for d in rows:
        ratio = math.exp(d["advantage_wordwise"])
        shown = f"{ratio:.0f}×" if ratio >= 1 else f"1/{1 / ratio:.0f}×"
        word = esc(d["word"])
        if d["canonical_crosses_word"]:
            word = f"<b>{word}</b>"
        out.append(
            f"<tr><td>{word}</td>"
            f"<td><code>…{esc(d['prefix_tail'])}</code></td>"
            f"<td><code>{esc(d['canonical_token'])}</code></td>"
            f"<td>{d['rank_canonical']:,}</td>"
            f"<td><code>{esc(d['wordwise_token'])}</code></td>"
            f"<td>{d['rank_wordwise']:,}</td><td>{shown}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def build(out_path: Path) -> None:
    analysis = json.loads((RESULTS / "analysis.json").read_text())
    emission = json.loads((RESULTS / "emission_logprob.json").read_text())
    abbrev = json.loads((RESULTS / "abbreviations.json").read_text())
    enc = tiktoken.get_encoding("o200k_base")

    t = analysis["length"]["glued_only"]["o200k_base"]["totals"]
    b = analysis["boundaries"]["o200k_base"]
    c = analysis["chunking"]
    ch = analysis["channels"]
    kinds = analysis["gluing_by_kind_boundaries"]
    sens = analysis["segmentation_sensitivity"]
    n_glued = analysis["n_glued"]

    #: gpt-oss leads: it is an OpenAI model whose tokenizer gives the same ids
    #: as the o200k the length analysis uses, so its divergences are the same
    #: positions. The others are a cross-lineage check in their own
    #: vocabularies, and are reported but not in the main flow.
    order = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen3-1.7B",
        "HuggingFaceTB/SmolLM2-1.7B",
    ]
    present = {s["model"] for s in emission["sequences"]}
    models = [m for m in order if m in present]
    headline = [m for m in models if m.startswith("openai/")] or models[:1]
    # Only decisions that sit at a boundary the model actually glued. The two
    # emission models also disagree at the report's paragraph breaks, where
    # canonical merges ".\n"; that is the report's formatting, not a choice
    # the model made, and it accounted for 12 of 126 before this filter.
    div = {
        m: [
            d
            for d in emission["divergences"]
            if d["model"] == m and d["text_kind"] == "glued" and d["at_glued_boundary"]
        ]
        for m in models
    }
    seq = {
        m: [
            s
            for s in emission["sequences"]
            if s["model"] == m and s["text_kind"] == "glued"
        ]
        for m in models
    }

    def median(xs: list[int]) -> int:
        return round(statistics.median(xs))

    tok_svg, _ = figure_tokens(enc)
    from transformers import AutoTokenizer

    others = [m for m in models if m not in headline]
    vocab_sizes = ", ".join(
        f"{len(AutoTokenizer.from_pretrained(m)):,}" for m in others
    )
    standout = max(div[headline[0]], key=lambda d: d["advantage_wordwise"])

    body: list[str] = []
    add = body.append

    add(f"""
<p class="lede">OpenAI's
<a href="https://alignment.openai.com/misalignment-reports/suppressing-misalignment-monitoring-alerts-and-disabling-misalignment-monitoring/">misalignment
report</a> of 20 October 2026 is the first public source with chain-of-thought
from GPT&#8209;6.1 Sol and GPT&#8209;6 Astra. Much of it is missing the spaces
between words.</p>

<blockquote><p>Need asknotifications ifsendno-- since noSlack withwrite
no_notifytrue preventsalertsbut disabling? parent shouldn'tset no_notifytrue
since policy triggeruserno messages requiresnotproactive tool.</p></blockquote>

<p><a href="https://www.lesswrong.com/posts/igEogGD9TAgAeAM7u/jimrandomh-s-shortform?commentId=5yQXwLtLgJqWwg5nf">jimrandomh
noticed this</a> and read it as a possible side channel. This measures what the
gluing does to the length of the text, and what it would take to produce.</p>

<p class="note">Code, data and the full run log:
<a href="{REPO_URL}">github.com/brendanlong/astra-sol-cot-length-penalty-experiment</a>
· discussion: <a href="{DISCUSSION_URL}">LessWrong</a></p>

<p><b>Over the {n_glued} glued excerpts, gluing makes the text
{abs(t["pct_chars_vs_spaced"]):.1f}% shorter in characters and
{t["pct_canonical_glued_vs_spaced"]:.1f}% <i>longer</i> in tokens.</b> That is
consistent with a length penalty counting characters, and hard to square with
one counting tokens. But GPT&#8209;6 also passes up shorter words that would
save characters at no cost in tokens, which fits a penalty on the number of
words better than either.</p>
""")

    add("<h2>The same characters, three different token sequences</h2>")
    add("""
<p>The model's text and a normally spaced version of it contain the same words.
They do not cost the same number of tokens, and the glued version has a chunk
that belongs to two words at once.</p>
""")
    add(
        figure(
            """<p class="note">o200k_base, the closest public stand-in for
GPT&#8209;6's unpublished tokenizer. <code>notsendmessages</code> is from the
report, in the first Sol chain-of-thought. Where the chunks fall is a property
of the tokenizer, not of the text: SmolLM2's vocabulary splits this string the
way o200k does, Qwen's happens to split it at the word boundaries.</p>""",
            tok_svg,
            1,
            "Gluing changes where the token boundaries fall.",
        )
    )

    add("<h2>Gluing costs tokens</h2>")
    add(
        figure(
            f"""<details><summary>The counts behind the bars, and what holds
elsewhere</summary>
<table>
<tr><th>measure</th><th>as written</th><th>spaces restored</th><th>change</th></tr>
<tr><td>characters</td><td>{t["chars_glued"]:,}</td><td>{t["chars_spaced"]:,}</td><td>{t["pct_chars_vs_spaced"]:+.1f}%</td></tr>
<tr><td>tokens, canonical</td><td>{t["canonical_glued"]}</td><td>{t["canonical_spaced"]}</td><td>{t["pct_canonical_glued_vs_spaced"]:+.1f}%</td></tr>
<tr><td>tokens, word-at-a-time</td><td>{t["wordwise_glued"]}</td><td>{t["wordwise_spaced"]}</td><td>{t["pct_wordwise_glued_vs_wordwise_spaced"]:+.1f}%</td></tr>
</table>
<p>{t["n_excerpts_canonical_positive"]} of the {n_glued} excerpts are individually
positive in canonical tokens, by between 1.7% and
{t["per_excerpt_pct_canonical_max"]:.1f}%; the seventh is a five-boundary
fragment that comes out exactly even. Under <code>cl100k_base</code> the
canonical count rises
{analysis["length"]["glued_only"]["cl100k_base"]["totals"]["pct_canonical_glued_vs_spaced"]:+.1f}%
instead; <code>o200k_harmony</code> returns counts byte-identical to
<code>o200k_base</code>.</p>
<p>The reference spacing is OpenAI's own, and its least certain calls are the
digit labels — it renders <code>severity4?</code> as two words while leaving
<code>L2</code> and <code>P0</code> as one. Merging all
{sens["n_boundaries_merged"]} of those moves the result to
{sens["digit_labels_merged"]["pct_chars_vs_spaced"]:+.1f}% /
{sens["digit_labels_merged"]["pct_canonical_glued_vs_spaced"]:+.1f}%.</p>
</details>""",
            inline_svg(
                "length",
                "Bar chart. Against the same words with the spaces restored, "
                f"the glued text is {abs(t['pct_chars_vs_spaced']):.1f}% shorter "
                f"in characters (3,361 against 3,639) but "
                f"{t['pct_wordwise_glued_vs_wordwise_spaced']:.1f}% longer in "
                "tokens emitted one word at a time (652 against 621) and "
                f"{t['pct_canonical_glued_vs_spaced']:.1f}% longer in tokens when "
                "re-encoded as one string (667 against 618).",
            ),
            2,
            "",
        )
    )
    add("""
<p>Neither GPT&#8209;6's tokenizer nor its emitted tokens are observable, so
this does not identify the training objective. A token penalty could still sit
alongside learned shorthand, imitation of terse human notes, or other rewards
pushing the same way.</p>
""")

    add("<h2>Shorter words were available, and went unused</h2>")
    ex = abbrev["example"]
    pairs = abbrev["pairs"]
    ctrl = {c["other"]: c for c in abbrev["controls"]}
    short_used = abbrev["short_forms_used"]
    add(f"""
<p>Gluing is not the only way to save characters. Writing <code>msgs</code>
for <code>messages</code> saves four at once, and since both are a single
token it costs nothing in tokens. One sentence from the first Sol
chain-of-thought, three ways:</p>
<table>
<tr><th></th><th>characters</th></tr>
<tr><td>as written<br><code>{esc(ex["as_written"])}</code></td><td>{
        ex["chars_as_written"]
    }</td></tr>
<tr><td>spaces restored<br><code>{esc(ex["spaced"])}</code></td><td>{
        ex["chars_spaced"]
    }</td></tr>
<tr><td>spaces restored, three short forms<br><code>{
        esc(ex["abbreviated"])
    }</code></td><td>{ex["chars_abbreviated"]}</td></tr>
</table>
<p>Three word choices get back {ex["chars_spaced"] - ex["chars_abbreviated"]} of
the {ex["chars_spaced"] - ex["chars_as_written"]} characters the gluing saved,
without touching a space.</p>

<p>Across the {abbrev["n_glued_excerpts"]} glued excerpts, {len(pairs)}
everyday short forms like these would have saved
<b>{abbrev["chars_saved_by_short_forms"]} characters</b>,
{100 * abbrev["chars_saved_by_short_forms"] / abbrev["chars_saved_by_gluing"]:.0f}%
of the {abbrev["chars_saved_by_gluing"]} the gluing saved, and
{abbrev["tokens_saved_by_short_forms"]} tokens. GPT&#8209;6 wrote the full form
{sum(p["uses"] for p in pairs)} times and the short one
{sum(short_used.values())} times ({
        ", ".join(f"<code>{w}</code> ×{n}" for w, n in short_used.items())
    }). It does abbreviate now and then — <code>req</code>,
<code>aux</code> and <code>config</code> appear — but rarely.</p>

<details><summary>The short forms, and how close gpt-oss puts each one</summary>
<table>
<tr><th>GPT&#8209;6 wrote</th><th>times</th><th>short form</th><th>rank</th><th>characters saved</th></tr>
{
        "".join(
            f"<tr><td>{p['word']}</td><td>{p['uses']}</td>"
            f"<td><code>{p['short']}</code></td><td>{p['rank']}</td>"
            f"<td>{p['chars_saved']}</td></tr>"
            for p in pairs
        )
    }
</table>
<p>The rank is how close the short form sits to the full word in
{abbrev["model"].split("/")[-1]}'s input embeddings, among the
{abbrev["n_word_tokens"]:,} tokens that are a lowercase English-looking word:
rank 1 means no other word is closer. Most of the words ranked above a short
form are other forms of the same word and translations — <code>message</code>,
<code>mensajes</code>.</p>
<p>Closeness alone does not make two words interchangeable:
<code>lower</code> is rank {ctrl["lower"]["rank"]} for <code>higher</code>, and
<code>dog</code> rank {ctrl["dog"]["rank"]} for <code>cat</code>. So the pairs
were picked by hand and checked at each place they are used, leaving out ones
that only fit some of them (<code>appropriate</code> → <code>proper</code>);
the embeddings show only that gpt-oss treats each pair as near-neighbours.</p>
</details>

<p>The two shortcuts come apart on what they save. Gluing saves characters and
words and costs tokens; a short form saves characters only. A model that takes
the first and leaves the second fits a penalty on the number of words better
than one on characters. It is not decisive: dropping a space is one habit that
applies at every boundary, while short forms are learned word by word, so
reinforcement learning could find gluing first under a character penalty
too.</p>
""")

    add("<h2>The stored transcript is probably not what ran</h2>")
    lead = headline[0]
    q = div[lead]
    n_canon_deep = sum(d["rank_canonical"] > 100 for d in q)
    n_word_shallow = sum(d["rank_wordwise"] <= 100 for d in q)
    crossing = [d for d in q if d["canonical_crosses_word"]]
    inside = [d for d in q if not d["canonical_crosses_word"]]
    w_cross = f"{sum(d['advantage_wordwise'] > 0 for d in crossing)}/{len(crossing)}"
    w_inside = f"{sum(d['advantage_wordwise'] > 0 for d in inside)}/{len(inside)}"
    pooled = [
        d
        for d in emission["divergences"]
        if d["text_kind"] == "glued"
        and d["at_glued_boundary"]
        and d["canonical_crosses_word"]
    ]
    w_pooled = f"{sum(d['advantage_wordwise'] > 0 for d in pooled)}/{len(pooled)}"
    add(f"""
<p>If the models emit glued text one word at a time, the transcript OpenAI
stores re-encodes to a different, {t["pct_retokenization_inflation"]:.1f}% longer
sequence than the one that ran. Both versions decode to identical characters,
so a model can be asked which of the two it would emit.</p>

<p>{lead.split("/")[-1]} is the one to ask: it is an OpenAI model, and its
tokenizer gives token-for-token identical ids to the o200k used everywhere
above, so the points where the two tokenizations disagree are the same
positions the rest of this analysis measures.</p>

<p><b>Which positions.</b> The two tokenizations cut the same characters in
different places. Line up their cut points: wherever both put a boundary, the
two paths are in the same state, and between two such shared boundaries they
either chose the same tokens or they did not. Where they did not, the first
token of that stretch is the decision — at that moment both paths have emitted
an <i>identical</i> prefix and are choosing differently about what comes next.
Those are the positions, and there is nothing to choose: they fall out of where
the tokenizations disagree. One forward pass over the excerpt gives the
model's full next-token distribution after that shared prefix, and the two
candidates' ranks are read off it.</p>

<details><summary>Worked example</summary>
<p>For <code>notsendmessages</code>, with the preceding text as context:</p>
<table>
<tr><td>canonical</td><td><code>n</code> <code>ots</code> <code>end</code> <code>messages</code></td><td>cuts after character 1, 4, 7, 15</td></tr>
<tr><td>word-at-a-time</td><td><code>not</code> <code>send</code> <code>messages</code></td><td>cuts after character 3, 7, 15</td></tr>
</table>
<p>They share the cuts at 7 and 15 (and the start). Between 0 and 7 they
disagree, so the first tokens of that stretch are the two candidates:
<code>n</code> against <code>not</code>. Between 7 and 15 both emit
<code>messages</code>, so there is no decision there. That excerpt contributes
one divergence, not two — which is why there are
{len(q)} of them across the {n_glued} glued excerpts rather than one per
straddling chunk.</p>
</details>
""")
    add(
        figure(
            f"""<details><summary>Every model scored, and the whole-sequence
measurement that disagrees</summary>
<table>
<tr><th>model</th><th>divergences</th><th>word-at-a-time more likely</th><th>median rank: canonical vs word-at-a-time</th><th>whole excerpt (nats)</th></tr>
{
                "".join(
                    f"<tr><td>{m.split('/')[-1]}</td><td>{len(div[m])}</td>"
                    f"<td>{sum(d['advantage_wordwise'] > 0 for d in div[m])}"
                    f"/{len(div[m])}</td>"
                    f"<td>{median([d['rank_canonical'] for d in div[m]]):,} vs "
                    f"<b>{median([d['rank_wordwise'] for d in div[m]]):,}</b></td>"
                    f"<td>{sum(x['advantage_wordwise'] for x in seq[m]):+.1f}</td></tr>"
                    for m in models
                )
            }
</table>
<p>Counted here are only the decisions that fall at a boundary the model
actually glued. The two emission models also disagree at the report's
paragraph breaks, where canonical merges <code>.\n</code> into one token and
word-at-a-time does not; that is the report's formatting rather than anything
the model chose, and it accounted for 12 of 126 such decisions before this
filter.</p>
<p>Ranks are 1-indexed, so rank 1 is the model's first choice. The small
models are a cross-lineage check and each works in its own vocabulary
({vocab_sizes} tokens, none of them o200k), so their divergences are at
different positions than the gpt-oss ones and are not directly comparable.
Where a tokenizer already splits the glued string at its word boundaries there
is nothing to diverge about, which is why the counts differ.</p>
<p>The last column is the whole-excerpt total, and it does not agree with the
one-step measurement: summed over a whole excerpt the canonical path comes out
better for gpt-oss-120b and both Qwen models, and worse for gpt-oss-20b and
SmolLM2. The one-step number is the one that corresponds to sampling, since a
decoder makes the local choice; the whole-sequence numbers do not support the
conclusion, and this data cannot separate the two readings.</p>
<p>On the spaced version of the same text the question barely arises: every
model tokenizes it the same way either way at all but 3 positions across all
{n_glued} excerpts.</p>
</details>""",
            figure_ranks(emission["divergences"], lead),
            3,
            f"{lead.split('/')[-1]}, the {len(q)} positions in the glued text "
            "where the two tokenizations disagree.",
        )
    )
    add(f"""
<p>At {sum(d["advantage_wordwise"] > 0 for d in q)} of the {len(q)} positions
the word-respecting token is the likelier of the two, median rank
{median([d["rank_canonical"] for d in q]):,} for the canonical token against
{median([d["rank_wordwise"] for d in q]):,} for the word-respecting one. The
starkest is after <code>{esc(standout["prefix_tail"])}</code>, where the
canonical <code>{esc(standout["canonical_token"])}</code> is the model's
{ordinal(standout["rank_canonical"])} choice and
<code>{esc(standout["wordwise_token"])}</code> its
{ordinal(standout["rank_wordwise"])}.</p>

<p><b>Only some of these are about word boundaries at all.</b> At
{len(inside)} of the {len(q)} the canonical chunk stays <i>inside</i> one word
— the two tokenizations merely cut the same word in different places, as with
<code>undermine</code> taken as <code>erm</code> or <code>ermine</code>. That
is a fact about BPE, not about gluing, and there the result is {w_inside},
near chance.</p>

<p>The hypothesis is only tested where the canonical chunk runs <b>past the
end of the word</b>, committing to the next word before this one is finished.
That happens at {len(crossing)} of the {len(q)} positions, and the word's own
token is likelier at <b>{w_cross}</b> of them. Pooled over all five models it
is <b>{w_pooled}</b>, though that is {len(pooled)} model-position pairs over
only 12 distinct positions, since models sharing a tokenizer see the same
ones.</p>

<details><summary>Every decision, and which token it preferred</summary>
{decision_table(q)}
<p>Bold words are the boundary-crossing cases, listed first. Read the first
row as: while writing <b>is</b>, after <code>…assinstruction? This</code>, the
word's own token <code>is</code> is 213 times likelier than <code>isa</code> —
the chunk that runs on into the following word. The rest stay inside one word:
<code>monitor</code> against <code>mon</code> is both tokenizations cutting
<i>monitors</i> differently, with neither crossing anything.</p>
<p>The "word's own" column is whatever
<code>encode(word)</code> gives for that word alone, which is not always one
token — <code>monitors</code> is <code>mon</code>+<code>itors</code>, while
<code>Dummy</code> really is a single token.</p>
</details>

<p><b>Both candidates are deep in the tail.</b> The canonical token is ranked
below 100th at {n_canon_deep} of {len(q)} of these positions, but so is the
word-respecting one at {len(q) - n_word_shallow}. This model finds the glued
text unlikely however it is cut up — what the measurement shows is which of
two improbable options is the less improbable, not that either is something
the model would readily produce.</p>

<p>gpt-oss was not trained to glue words, so this says the straddling tokens
sit in the tail for a BPE model of this kind and scale. It does not say what a
model that learned to glue would do, since reinforcement learning on a
text-scored reward cannot distinguish the two tokenizations and the pretrained
prior is what breaks the tie.</p>
""")

    add("<h2>What cuts the other way</h2>")
    add(f"""
<p>Four results, in descending order of how much they cost the story.</p>

<p><b>Most of this text needs no unusual tokens at all.</b>
{c["n_chunked_as_standalone"]} of {c["n_glued_words"]} glued words
({100 * c["frac_chunked_as_standalone"]:.0f}%) are canonically chunked exactly
as if emitted alone, because o200k has a bare-word entry for most common
English words. The straddling chunks in Figure 1 happen at
{b["straddled_dropped_boundaries"]} of the {b["n_dropped"]} glued
boundaries.</p>

<p><b>The longest Sol chain-of-thought in the report has no gluing at all</b> —
143 boundaries, all spaced, a few turns after one in the same conversation that
glued 80 of 109.</p>

<p><b>The model glues the boundaries where gluing is free.</b> Mid-sentence,
gluing costs {b["dropped_mid_sentence"]["mean_canonical_delta"]:+.2f} canonical
tokens where the model did it against
{b["kept_mid_sentence"]["mean_canonical_delta"]:+.2f} where it did not
(p&nbsp;=&nbsp;{b["permutation_test_mean_delta"]["p_one_sided_glued_cheaper"]:.4f},
labels shuffled within each excerpt). We predicted no relationship. The likely
explanation is lexical rather than token-aware: common words have a bare
vocabulary entry, so gluing before them costs nothing, and the model glues
before common words.</p>

<p><b>Glued English is no more ad hoc than GPT&#8209;2 integer tokenization</b>
— the standard example of tokenization forcing memorization over an algorithm.
Running the same sweep on both gives Figure 4, and the three metrics disagree
on an ordering.</p>
""")
    add(
        figure(
            f"""<details><summary>What the sweep does</summary>
<p>For each of {c["context_sweep"]["n_right"]} items, glue it after each of
{c["context_sweep"]["n_left"]} left-context items and count how many distinct
chunkings it takes. Left-hand items for the English sweep are the word types
from the glued excerpts; for the integer sweeps they are random integers of the
stated length, drawn with the seeds <code>analysis.py</code> uses.</p>
<p>Beren Millidge's
<a href="https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/">original
observation</a> is about integers in isolation: 2249 is
<code>2</code>+<code>249</code>,
2250 is <code>22</code>+<code>50</code>, 2251 is <code>225</code>+<code>1</code>
— which this sweep generalizes to items glued after a left neighbour.</p>
</details>""",
            figure_chunking(c),
            4,
            "Glued English against GPT&#8209;2 integers, same sweep, three ways "
            "of counting.",
        )
    )

    add("<h2>Is it a side channel?</h2>")
    add(f"""
<p>The space/no-space choice is visible: anyone reading the transcript can see
it, which is a strange property for a covert channel. The alternative that
leaves no trace is much larger. Re-segmenting the same characters into
different vocabulary tokens — which changes nothing a reader can see — offers
{ch["invisible_resegmentation"]["bits_per_token"]:.2f} bits per token against at
most {ch["visible_space_choice"]["bits_per_token"]:.2f} for the space choice,
with {ch["invisible_resegmentation"]["n_tokens_with_alternatives"]} of
{ch["invisible_resegmentation"]["n_tokens"]} tokens having at least one
alternative segmentation.</p>

<details><summary>Why neither number is a usable rate</summary>
<p>Both are upper bounds on capacity at a fixed reference text, counted per
canonical reference token, over all 15 excerpts with a reference spacing.
Spending the invisible capacity emits more tokens than the reference has, so
per <i>emitted</i> token it is much smaller; and the visible figure assumes
every boundary is free when sentence boundaries are never glued
({b["predictors"]["left_ends_sentence"]["n_true"]} of them, none glued).</p>
</details>

<p>None of this rules out steganography. It says this artifact is a poor
candidate for it.</p>
""")

    add("<h2>What would settle it</h2>")
    add("""
<ul>
<li><b>Token IDs.</b> Everything here is an emission <i>model</i>. The direct
test — whether re-encoding a model's own output reproduces the tokens it
emitted — needs IDs or logprobs that no API exposes.</li>
<li><b>A length-penalty ablation.</b> Train two models with a per-character and
a per-token chain-of-thought penalty and see which one glues.</li>
<li><b>An independent segmentation.</b> The word boundaries here are OpenAI's
own "Spaces added" view. It is an exact space-insertion of the original, but it
is not blind, and in places it is a reading rather than a fact.</li>
</ul>
""")

    add("<h2>Caveats</h2>")
    add(f"""
<p>Seven glued excerpts from two incidents, and <b>OpenAI chose which excerpts
to publish</b> — plausibly in part because the gluing was striking. The
concentration in chain-of-thought ({kinds["cot"]["glued"]} of
{kinds["cot"]["boundaries"]} boundaries glued, against
{kinds["agent_message"]["glued"]} of {kinds["agent_message"]["boundaries"]} in
agent-to-agent messages and {kinds["user_facing"]["glued"]} of
{kinds["user_facing"]["boundaries"]} in user-facing text) could be a selection
effect end to end. Nothing here establishes a rate at which GPT&#8209;6 does
this.</p>

<details><summary>What the dataset counts as the model's text</summary>
<p>All 22 excerpts in the report, labelled by model, speaker and kind. The
report's HTML ships OpenAI's own "Spaces added" view for each chain-of-thought
excerpt, and a test asserts it differs from the original by spaces only, so the
word segmentation is theirs. The two glued agent messages have no such view and
are segmented by hand; every other model output is recorded as an explicit "not
glued" judgement rather than a silent default.</p>
<p>Editorial insertions — the report's bracketed paraphrases and its elisions —
break the text rather than being measured, since their characters, internal
spaces and adjacent boundaries are the report authors' writing, not the
model's.</p>
</details>

<details><summary>Claims withdrawn during review</summary>
<p>Two adversarial review rounds and a presentation review withdrew five claims
from earlier drafts of this analysis. The dataset was gitignored and so never
committed; editorial insertions were being measured as model text; a clustering
test used a statistic that is fixed by construction; "no token-awareness in
boundary choice" was the null reading of a comparison that was never run, and
is false; "gluing respects code identifiers" was an artifact of a detector that
missed PascalCase; the GPT&#8209;2 comparison compared a per-item count against
a whole-population count; and every reported rank was off by one, because the
code produced 0-indexed ranks that were printed as ordinals. Each is recorded
in the run log with what replaced it.</p>
</details>
""")

    add("<h2>Reproducing</h2>")
    add(f"""
<p>Everything runs from the committed corpus — the report as saved, plus the
extracted dataset. No GPU, no accounts.</p>
<pre><code>git clone https://github.com/brendanlong/astra-sol-cot-length-penalty-experiment
cd astra-sol-cot-length-penalty-experiment
uv sync
SKIP_MODELS=1 bash scripts/reproduce.sh   # every number and figure
bash scripts/reproduce.sh      # plus the small-model scoring (~15 min, CPU)</code></pre>
<p>The repo is
<a href="{REPO_URL}">brendanlong/astra-sol-cot-length-penalty-experiment</a>.
The full run log, with every command and the numbers this page draws from, is
in <a href="{REPO_URL}/blob/main/RESULTS.md">RESULTS.md</a>; the predictions
registered beforehand are in
<a href="{REPO_URL}/blob/main/EXPERIMENT_PLAN.md">EXPERIMENT_PLAN.md</a>.
This is discussed on
<a href="{DISCUSSION_URL}">LessWrong</a>.
Related earlier work on re-tokenization is in
<a href="https://github.com/brendanlong/tokenization-hidden-computation-experiment">tokenization-hidden-computation-experiment</a>.</p>
""")

    body_html = "\n".join(body)
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
details { margin: 0.8em 0; }
summary { cursor: pointer; color: var(--link, #2a78d6); }
table { border-collapse: collapse; font-size: 0.93em; }
th, td { text-align: left; padding: 3px 14px 3px 0; }
td code { overflow-wrap: anywhere; }
</style>
"""
    tmp = out_path.with_suffix(".body.html")
    tmp.write_text(style + body_html)
    subprocess.run(
        [
            "python3",
            str(TEMPLATE / "scripts" / "render.py"),
            str(tmp),
            "--title",
            "What the missing spaces in GPT-6's chain-of-thought are optimizing",
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
