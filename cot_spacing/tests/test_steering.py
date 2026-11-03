from __future__ import annotations

from transformers import AutoTokenizer

from cot_spacing.gpt_oss import MODEL, with_reasoning
from cot_spacing.learnability import site_sequences
from cot_spacing.steering import degenerate, measure, spacing, stated_numbers
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
    cot = tok.encode("Needs autoalarm severity4?", add_special_tokens=False)
    ids, span = with_reasoning(tok, "a question", cot)
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


def test_stated_numbers_reads_formatted_answers() -> None:
    assert 70000 in stated_numbers(r"\boxed{\$70{,}000}")
    assert 70000 in stated_numbers(r"\[ \boxed{70\,000} \]")
    assert 70000 in stated_numbers("**$70\u202f000**")
    assert 64 in stated_numbers("pay **$64** for the 16 glasses")
    assert 160 in stated_numbers("**160 minutes** (or 2 hours 40 minutes)")
    assert 18 not in stated_numbers("So she makes **$17** a day, not 16.")


def test_truncated_answer_is_neither_finished_nor_correct() -> None:
    cut = measure(
        "<|channel|>analysis<|message|>ok<|end|><|start|>assistant"
        "<|channel|>final<|message|>The answer is **18**",
        "18",
    )
    assert cut["finished"] is False and cut["correct"] is False


def test_short_forms_in_code_are_not_counted() -> None:
    m = measure(
        "<|channel|>final<|message|>Run `git diff` then\n```\nnpm view pkg\n```\n"
        "Check the env and the repository.<|return|>",
        None,
    )
    assert (m["answer_short"], m["answer_long"]) == (1, 1)


def test_nospace_and_wordwise_renderings() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    for e in build():
        assert " " not in e.texts["nospace"]
        assert e.texts["nospace"].replace("\n", "") == e.texts["glued"].replace(
            " ", ""
        ).replace("\n", "")
        assert tok.decode(e.ids["glued_wordwise"]) == e.texts["glued"]
        assert e.ids["glued"] == tok.encode(e.texts["glued"], add_special_tokens=False)
    assert any(e.ids["glued_wordwise"] != e.ids["glued"] for e in build())


def test_spacing_separates_sentence_clause_and_word_spaces() -> None:
    counts = spacing("We need thisone. Thenext, item,done.Last one 3.5 here")
    assert (counts["sentence_spaces"], counts["sentence_ends"]) == (1, 2)
    assert (counts["clause_spaces"], counts["clause_ends"]) == (1, 2)
    assert counts["word_spaces"] == 5


def test_degenerate_catches_loops_and_garbage() -> None:
    assert degenerate("withthejsonencoder" * 40)
    assert degenerate("ok \ufffd\ufffd\ufffd")
    assert not degenerate("A normal answer that says one thing, then stops.")
