from __future__ import annotations

import statistics

import tiktoken
from transformers import AutoTokenizer

from cot_spacing.chunking import (
    chunk_pattern,
    chunks_covering,
    integer_chunk_patterns,
    words_in_context,
)
from cot_spacing.data import load_excerpts, model_outputs
from cot_spacing.emission_logprob import (
    canonical_ids,
    divergences,
    wordwise_ids,
)
from cot_spacing.reference import reference_spaced
from cot_spacing.segment import Segmentation, segment
from cot_spacing.tokenize_cost import (
    _looks_like_identifier,
    boundaries,
    count,
)


def enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def test_openai_spaced_view_is_pure_space_insertion() -> None:
    """The dataset's whole premise: the report's "Spaces added" view adds only spaces.

    If OpenAI had also reworded anything, it could not be used as a word
    segmentation of the original.
    """
    checked = 0
    for e in load_excerpts():
        spaced = e.spaced_text()
        if spaced is None:
            continue
        checked += 1
        assert spaced.replace(" ", "") == e.text().replace(" ", "")
    assert checked == 6


def test_every_model_output_has_a_spacing_judgement() -> None:
    for e in model_outputs(load_excerpts()):
        assert reference_spaced(e) is not None, e.id


def test_segment_labels_gaps_and_round_trips() -> None:
    seg = segment("Need asknotifications ifsendno", "Need ask notifications if send no")
    assert seg.words == ["Need", "ask", "notifications", "if", "send", "no"]
    assert seg.gaps == [" ", "", " ", "", ""]
    assert seg.original() == "Need asknotifications ifsendno"
    assert seg.spaced() == "Need ask notifications if send no"
    assert [len(r) for r in seg.runs()] == [1, 2, 3]


def test_segment_round_trips_every_excerpt() -> None:
    for e in model_outputs(load_excerpts()):
        spaced = reference_spaced(e)
        assert spaced is not None
        seg = segment(e.text(), spaced)
        assert seg.original() == e.text()
        assert seg.spaced() == spaced


def test_counts_agree_when_nothing_is_glued() -> None:
    text = "Need durable results and honest context"
    seg = segment(text, text)
    c = count(seg, enc())
    assert not seg.dropped
    assert c.chars_glued == c.chars_spaced
    assert c.canonical_glued == c.canonical_spaced
    assert c.wordwise_glued == c.wordwise_spaced


def test_gluing_costs_tokens_in_the_canonical_example() -> None:
    """ "not send messages" is 3 tokens spaced, 4 glued, 3 word-at-a-time."""
    seg = segment("notsendmessages", "not send messages")
    c = count(seg, enc())
    assert c.canonical_spaced == 3
    assert c.canonical_glued == 4
    assert c.wordwise_glued == 3


def test_boundary_deltas_and_straddling() -> None:
    seg = segment("notsendmessages", "not send messages")
    bs = boundaries("x", seg, enc())
    assert [b.dropped for b in bs] == [True, True]
    assert [b.left for b in bs] == ["not", "send"]
    # Gluing "not"+"send" turns 'not'+' send' into 'n'+'ots'+'end': one extra
    # token, with a token straddling the boundary.
    assert bs[0].canonical_delta == 1
    assert bs[0].straddled
    assert not bs[1].straddled


def test_words_in_context_detects_non_standalone_chunking() -> None:
    seg = segment("notsendmessages", "not send messages")
    ws = words_in_context("x", seg, enc())
    assert [w.word for w in ws] == ["not", "send", "messages"]
    assert ws[1].in_context == ["ots", "end"]
    assert ws[1].standalone == ["send"]
    assert not ws[1].matches_standalone
    assert ws[2].matches_standalone


def test_chunks_covering_spans_only_the_requested_bytes() -> None:
    text = "abcdef"
    e = enc()
    cover = chunks_covering(text, 0, len(text.encode()), e)
    assert b"".join(cover) == text.encode()


def test_chunk_pattern_depends_on_left_context() -> None:
    e = enc()
    assert chunk_pattern("send", e) == (4,)
    # Glued after "not", 'send' is covered by 'ots'+'end' — two tokens, and the
    # first one reaches back into the previous word, so the lengths sum to more
    # than the word.
    assert chunk_pattern("send", e, prefix="not") == (3, 3)
    assert chunk_pattern("send", e, prefix="do") != chunk_pattern("send", e)


def test_gpt2_integer_baseline_reproduces_published_examples() -> None:
    """Beren Millidge's examples: 2249 is 1-3, 2250 is 2-2, 2251 is 3-1."""
    pats = integer_chunk_patterns(tiktoken.get_encoding("gpt2"), 2249, 2252)
    assert pats[2249] == (1, 3)
    assert pats[2250] == (2, 2)
    assert pats[2251] == (3, 1)


def test_tokenizations_partition_the_same_text() -> None:
    """Both emission models cover the same characters, differently split.

    GPT-2 happens to tokenize "notsendmessages" as 'not'+'send'+..., so the
    divergent example here is "donotsend" -> 'don'+'ots'+'end'.
    """
    tok = AutoTokenizer.from_pretrained("gpt2")
    seg = segment("donotsend", "do not send")
    canon = canonical_ids(seg.original(), tok)
    wordwise = wordwise_ids(seg, tok, spaced=False)
    assert tok.decode(canon.ids) == tok.decode(wordwise.ids) == "donotsend"
    assert canon.ends[-1] == wordwise.ends[-1] == len("donotsend")
    assert canon.ids != wordwise.ids


def test_wordwise_carries_the_leading_space_only_where_written() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    seg = segment("Need asknotifications", "Need ask notifications")
    wordwise = wordwise_ids(seg, tok, spaced=False)
    pieces = [tok.decode([i]) for i in wordwise.ids]
    assert pieces[0] == "Need"
    assert pieces[1] == " ask"  # the model wrote this space
    assert not pieces[2].startswith(" ")  # it did not write this one


def test_divergences_finds_the_choice_points() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    seg = segment("donotsend", "do not send")
    canon = canonical_ids(seg.original(), tok)
    wordwise = wordwise_ids(seg, tok, spaced=False)
    points = divergences(canon, wordwise)
    assert points
    for n_prefix, tok_c, tok_w in points:
        assert tok_c != tok_w
        assert canon.ids[:n_prefix] == wordwise.ids[:n_prefix]


def test_identical_tokenizations_have_no_divergences() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    text = "Need durable results"
    seg = segment(text, text)
    canon = canonical_ids(text, tok)
    wordwise = wordwise_ids(seg, tok, spaced=True)
    if canon.ids == wordwise.ids:
        assert divergences(canon, wordwise) == []


def test_editorial_insertions_break_the_text_rather_than_joining_it() -> None:
    """A redacted identifier is not model text and must not create a boundary.

    "no longer[runtime sample identifier] sofallback..." tells us nothing about
    what the model wrote on either side of the identifier, so neither boundary
    may be counted.
    """
    excerpt = next(
        e for e in load_excerpts() if e.id == "reference_impl-15-subagent-cot"
    )
    text = excerpt.text()
    assert "[" not in text
    assert "runtime sample identifier" not in text
    assert "longer" not in text.replace("longer\n", "")  # it ends a line
    assert "longerso" not in text  # and is not glued to what followed


def test_inline_elision_does_not_create_a_boundary() -> None:
    excerpt = next(
        e for e in load_excerpts() if e.id == "reference_impl-21-parent-user_facing"
    )
    assert "No I did not" not in excerpt.text()
    assert excerpt.text().startswith("No\n")


def test_identifier_detector_covers_pascal_case() -> None:
    """The report's class names are PascalCase; the predictor must see them."""
    for word in ("RuntimeContext", "MonitorPublicationScope", "no_notifytrue"):
        assert _looks_like_identifier(word)
    for word in ("monitoring", "P0", "L2", "the"):
        assert not _looks_like_identifier(word)


def test_mean_run_length_is_fixed_by_the_drop_rate() -> None:
    """Why run_structure tests dispersion instead.

    Two arrangements of the same number of drops have identical mean run
    length, so the mean cannot detect clustering.
    """
    clustered = Segmentation(words=list("abcdef"), gaps=["", "", " ", " ", " "])
    spread = Segmentation(words=list("abcdef"), gaps=["", " ", "", " ", " "])
    assert len(clustered.dropped) == len(spread.dropped)
    mean = lambda s: statistics.mean(len(r) for r in s.runs())  # noqa: E731
    assert mean(clustered) == mean(spread)
    assert statistics.variance(
        [len(r) for r in clustered.runs()]
    ) != statistics.variance([len(r) for r in spread.runs()])


def test_clipped_chunk_pattern_counts_only_the_word() -> None:
    """The convention the GPT-2 integer baseline uses, so comparable to it."""
    e = enc()
    assert chunk_pattern("send", e, prefix="not") == (3, 3)  # straddle counted whole
    assert chunk_pattern("send", e, prefix="not", clip=True) == (1, 3)
    assert sum(chunk_pattern("send", e, prefix="not", clip=True)) == len("send")


def test_merge_undoes_a_boundary_without_changing_the_text() -> None:
    seg = segment("severity4? Wecould", "severity 4? We could")
    merged = seg.merge({0})
    assert merged.words[0] == "severity4?"
    assert merged.original() == seg.original()
    assert len(merged.gaps) == len(seg.gaps) - 1
