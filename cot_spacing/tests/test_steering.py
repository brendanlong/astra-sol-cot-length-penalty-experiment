from __future__ import annotations

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


def test_site_prefix_and_original_rebuild_the_spaced_text() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    head = "<|channel|>analysis<|message|>"
    spaced = {e.id: e.texts["spaced"] for e in build()}
    seqs = site_sequences(tok)
    assert {s["kind"] for s, *_ in seqs} == {"glue", "short"}
    for site, prefix, orig, edit in seqs:
        text = tok.decode(prefix + orig)
        reasoning = text[text.index(head) + len(head) :]
        assert reasoning.removesuffix("<|end|>") in spaced[str(site["excerpt"])]
        assert spaced[str(site["excerpt"])].startswith(
            reasoning.removesuffix("<|end|>")
        )
        o, e = tok.decode(orig), tok.decode(edit)
        if site["kind"] == "glue":
            assert o == " " + e
        else:
            assert o.replace(str(site["word"]), str(site["short"]), 1) == e


def test_measure_counts_gluing_shortening_and_answers() -> None:
    spaced = measure(
        "<|channel|>analysis<|message|>We need the number of minutes.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>It is 18 minutes.<|return|>",
        "18",
    )
    glued = measure(
        "<|channel|>analysis<|message|>Weneedthe num of mins.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>It is 17.<|return|>",
        "18",
    )
    unfinished = measure("<|channel|>analysis<|message|>We need 18", "18")
    assert spaced["correct"] is True and glued["correct"] is False
    assert unfinished["correct"] is False and unfinished["finished"] is False
    assert (spaced["reasoning_long"], spaced["reasoning_short"]) == (2, 0)
    assert (spaced["answer_long"], spaced["answer_short"]) == (1, 0)
    assert (glued["reasoning_long"], glued["reasoning_short"]) == (0, 2)
    assert glued["reasoning_letters_per_space"] > spaced["reasoning_letters_per_space"]  # type: ignore[operator]
