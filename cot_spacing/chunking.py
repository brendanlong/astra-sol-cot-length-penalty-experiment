"""How hard the glued text is to emit canonically.

A BPE tokenizer merges greedily across the whole string, so the tokens covering
a word inside a glued run depend on what precedes it. Emitting the canonical
encoding therefore means committing to chunks that are not the word's own
chunks — the same situation Beren Millidge described for GPT-2 integers, where
2249 is '2'+'249' but 2250 is '22'+'50'. ``integer_chunk_patterns`` reproduces
that baseline so the two can be compared on the same scale.
"""

from __future__ import annotations

import functools
import itertools
import math
from typing import TYPE_CHECKING

import tiktoken
from pydantic import BaseModel

from cot_spacing.tokenize_cost import token_byte_offsets

if TYPE_CHECKING:
    from cot_spacing.segment import Segmentation


def chunks_covering(
    text: str, start: int, end: int, enc: tiktoken.Encoding
) -> list[bytes]:
    """The canonical tokens of ``text`` overlapping byte span [start, end)."""
    offsets = token_byte_offsets(text, enc)
    toks = enc.encode(text)
    out: list[bytes] = []
    for tok, lo, hi in zip(toks, offsets, offsets[1:], strict=False):
        if hi > start and lo < end:
            out.append(enc.decode_single_token_bytes(tok))
    return out


class WordInContext(BaseModel):
    excerpt_id: str
    word: str
    glued_left: bool
    #: Canonical tokens of the full excerpt that overlap this word.
    in_context: list[str]
    #: Canonical tokens of the word emitted on its own, bare.
    standalone: list[str]
    #: Canonical tokens of the word emitted on its own with a leading space.
    standalone_spaced: list[str]
    #: True if the in-context tokens are exactly the word's own bare tokens,
    #: i.e. the canonical encoding happens to agree with word-at-a-time here.
    matches_standalone: bool
    #: True if some covering token also covers neighbouring text.
    straddles: bool


def words_in_context(
    excerpt_id: str, seg: Segmentation, enc: tiktoken.Encoding
) -> list[WordInContext]:
    text = seg.original()
    offsets = set(token_byte_offsets(text, enc))
    out: list[WordInContext] = []
    pos = 0
    for i, word in enumerate(seg.words):
        start = pos
        end = start + len(word.encode())
        pos = end + len(seg.gaps[i].encode()) if i < len(seg.gaps) else end
        glued_left = i > 0 and seg.gaps[i - 1] == ""
        cover = chunks_covering(text, start, end, enc)
        standalone = [enc.decode_single_token_bytes(t) for t in enc.encode(word)]
        spaced = [enc.decode_single_token_bytes(t) for t in enc.encode(" " + word)]
        out.append(
            WordInContext(
                excerpt_id=excerpt_id,
                word=word,
                glued_left=glued_left,
                in_context=[c.decode(errors="replace") for c in cover],
                standalone=[c.decode(errors="replace") for c in standalone],
                standalone_spaced=[c.decode(errors="replace") for c in spaced],
                matches_standalone=cover == standalone,
                straddles=not (start in offsets and end in offsets),
            )
        )
    return out


def chunk_pattern(
    word: str, enc: tiktoken.Encoding, prefix: str = "", *, clip: bool = False
) -> tuple[int, ...]:
    """Byte lengths of the tokens covering ``word`` when glued after ``prefix``.

    The pattern, not the token ids: this is the "2-2 vs 1-3" description Beren
    used for integers.

    Two conventions, because they answer different questions and give very
    different numbers. Unclipped (the default) gives each covering token its
    full byte length, so a token that eats two bytes of the left word differs
    from one that eats three even when the word itself is carved identically;
    that is what an emitter has to produce. Clipped counts only the bytes
    inside the word, which is the convention the integer baseline uses (where
    there is no prefix to straddle) and so the one to use for comparing
    against it.
    """
    text = prefix + word
    start = len(prefix.encode())
    end = start + len(word.encode())
    if not clip:
        return tuple(len(c) for c in chunks_covering(text, start, end, enc))
    offsets = token_byte_offsets(text, enc)
    return tuple(
        min(hi, end) - max(lo, start)
        for lo, hi in itertools.pairwise(offsets)
        if hi > start and lo < end
    )


def context_sweep(
    right_words: list[str], left_words: list[str], enc: tiktoken.Encoding
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Chunk pattern of each right word glued after each left word."""
    return {
        right: {left: chunk_pattern(right, enc, prefix=left) for left in left_words}
        for right in right_words
    }


def integer_chunk_patterns(
    enc: tiktoken.Encoding, lo: int = 1000, hi: int = 10000
) -> dict[int, tuple[int, ...]]:
    """Beren's GPT-2 integer baseline: the chunk pattern of each integer."""
    return {
        n: tuple(len(enc.decode_single_token_bytes(t)) for t in enc.encode(str(n)))
        for n in range(lo, hi)
    }


def neighbour_disagreement(patterns: dict[int, tuple[int, ...]]) -> float:
    """Fraction of consecutive integers chunked differently from their successor."""
    pairs = list(itertools.pairwise(sorted(patterns)))
    return sum(patterns[a] != patterns[b] for a, b in pairs) / len(pairs)


@functools.cache
def _vocab_bytes(encoding: str) -> frozenset[bytes]:
    enc = tiktoken.get_encoding(encoding)
    out: set[bytes] = set()
    for i in range(enc.n_vocab):
        try:
            out.add(enc.decode_single_token_bytes(i))
        except KeyError:  # special tokens have no byte form
            continue
    return frozenset(out)


@functools.cache
def _n_segmentations(piece: bytes, vocab: frozenset[bytes]) -> int:
    if not piece:
        return 1
    return sum(
        _n_segmentations(piece[k:], vocab)
        for k in range(1, len(piece) + 1)
        if piece[:k] in vocab
    )


def invisible_channel_capacity(text: str, encoding: str) -> dict[str, float]:
    """Bits available from re-segmenting the text without changing a character.

    Splitting one canonical token into several vocabulary tokens leaves the
    decoded text identical, so a transcript stored as text cannot show it. This
    counts only within-token splits, so it is a lower bound on the capacity of
    that channel — the point of comparison for the visible space-dropping the
    models actually do.
    """
    enc = tiktoken.get_encoding(encoding)
    vocab = _vocab_bytes(encoding)
    tokens = enc.encode(text)
    bits = 0.0
    ambiguous = 0
    for t in tokens:
        n = _n_segmentations(enc.decode_single_token_bytes(t), vocab)
        bits += math.log2(n)
        ambiguous += n > 1
    return {
        "n_tokens": len(tokens),
        "n_tokens_with_alternatives": ambiguous,
        "bits": bits,
        "bits_per_token": bits / len(tokens) if tokens else 0.0,
    }
