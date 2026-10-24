from __future__ import annotations

import tiktoken
from transformers import AutoTokenizer

from cot_spacing.gpt_oss import MODEL, with_reasoning
from cot_spacing.learnability import site_sequences
from cot_spacing.steering import measure
from cot_spacing.variants import build, shorten


def test_variants_differ_from_spaced_only_by_their_own_edit() -> None:
    for e in build():
        spaced, glued, short, both = (
            e.texts[v] for v in ("spaced", "glued", "short", "both")
        )
        assert glued.replace(" ", "") == spaced.replace(" ", "")
        assert both.replace(" ", "") == short.replace(" ", "")
        assert len(short.split()) == len(spaced.split())
        assert e.chars_saved["both"] == e.chars_saved["glued"] + e.chars_saved["short"]
        assert e.glued_boundaries == e.chars_saved["glued"]


def test_shorten_keeps_identifiers_and_capitals() -> None:
    assert shorten("Preserve no_notify monitoring") == "Preserve no_notify mon"
    assert (
        shorten("Messages to MonitorPublicationScope")
        == "Msgs to MonitorPublicationScope"
    )
    assert shorten("authorization?") == "auth?"


def test_reasoning_span_is_the_reasoning() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, span = with_reasoning(tok, "a question", "Needs autoalarm severity4?")
    assert tok.decode(ids[span]) == "Needs autoalarm severity4?"
    assert tok.decode(ids[span.stop :]) == "<|end|>"


def test_site_continuations_decode_to_the_edit() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    seqs = site_sequences(tok)
    kinds = {s["kind"] for s, *_ in seqs}
    assert kinds == {"glue", "short"}
    for site, _prefix, orig, edit in seqs:
        o, e = tok.decode(orig), tok.decode(edit)
        if site["kind"] == "glue":
            assert o == " " + e
        else:
            assert o.replace(str(site["word"]), str(site["short"]), 1) == e


def test_measure_counts_gluing_shortening_and_answers() -> None:
    enc = tiktoken.get_encoding("o200k_base")
    spaced = measure(
        "<|channel|>analysis<|message|>We need the number of minutes.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>It is 18.<|return|>",
        "18",
        enc,
    )
    glued = measure(
        "<|channel|>analysis<|message|>Weneedthe num of mins.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>It is 17.<|return|>",
        "18",
        enc,
    )
    assert spaced["correct"] is True and glued["correct"] is False
    assert (spaced["n_long"], spaced["n_short"]) == (2, 0)
    assert (glued["n_long"], glued["n_short"]) == (0, 2)
    assert glued["letters_per_space"] > spaced["letters_per_space"]  # type: ignore[operator]
