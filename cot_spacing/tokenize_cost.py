"""Tokenization cost of glued vs spaced text, under several emission models.

The question is what a length penalty would have been measuring. Four counts
per excerpt, all for the same visible text:

``canonical_spaced``
    Tokens if the model had written normal English and a BPE tokenizer encoded
    it. The honest baseline.
``canonical_glued``
    Tokens in the model's actual text, encoded canonically. This is what a
    stored transcript costs when re-tokenized, and what a token counter run
    over the transcript text reports.
``wordwise_glued``
    Tokens if the model emitted its actual text one word at a time, each word
    encoded canonically on its own (carrying a leading space only where it
    actually wrote one). The obvious non-canonical way to produce glued text,
    and the hypothesis for what it really emitted.
``wordwise_spaced``
    Same word-at-a-time emission, but of the spaced text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    import tiktoken

    from cot_spacing.segment import Segmentation

#: Primary tokenizer. GPT-6's is not public; o200k is OpenAI's current public
#: family (GPT-4o/4.1) and o200k_harmony is the gpt-oss reasoning variant.
PRIMARY_ENCODING = "o200k_base"
ENCODINGS = ("o200k_base", "o200k_harmony", "cl100k_base")


class Counts(BaseModel):
    chars_glued: int
    chars_spaced: int
    canonical_spaced: int
    canonical_glued: int
    wordwise_glued: int
    wordwise_spaced: int


def wordwise_tokens(
    seg: Segmentation, enc: tiktoken.Encoding, *, spaced: bool
) -> list[list[int]]:
    """Tokens per word under word-at-a-time emission."""
    gaps = [g if not spaced else (g if g == "\n" else " ") for g in seg.gaps]
    prefixes = ["", *gaps]
    return [enc.encode(p + w) for p, w in zip(prefixes, seg.words, strict=True)]


def count(seg: Segmentation, enc: tiktoken.Encoding) -> Counts:
    glued, spaced = seg.original(), seg.spaced()
    return Counts(
        chars_glued=len(glued),
        chars_spaced=len(spaced),
        canonical_spaced=len(enc.encode(spaced)),
        canonical_glued=len(enc.encode(glued)),
        wordwise_glued=sum(len(t) for t in wordwise_tokens(seg, enc, spaced=False)),
        wordwise_spaced=sum(len(t) for t in wordwise_tokens(seg, enc, spaced=True)),
    )


def token_byte_offsets(text: str, enc: tiktoken.Encoding) -> list[int]:
    """Byte offset of each canonical token boundary, including 0 and len."""
    offsets = [0]
    for tok in enc.encode(text):
        offsets.append(offsets[-1] + len(enc.decode_single_token_bytes(tok)))
    return offsets


class Boundary(BaseModel):
    """One word boundary in an excerpt, as the model did or didn't write it."""

    excerpt_id: str
    index: int
    left: str
    right: str
    dropped: bool
    #: Canonical tokens added by gluing this one boundary, holding the rest of
    #: the excerpt as the model wrote it. Negative means gluing saved tokens.
    canonical_delta: int
    #: Tokens added by gluing it under word-at-a-time emission: the cost of
    #: switching ``right`` from its with-space token(s) to its bare token(s).
    wordwise_delta: int
    #: Whether a canonical token spans this boundary in the model's own text
    #: (only meaningful when ``dropped``).
    straddled: bool
    left_ends_sentence: bool
    right_is_identifier: bool


def _toggle(seg: Segmentation, index: int, gap: str) -> str:
    gaps = list(seg.gaps)
    gaps[index] = gap
    return "".join(w + g for w, g in zip(seg.words, [*gaps, ""], strict=True))


def boundaries(
    excerpt_id: str, seg: Segmentation, enc: tiktoken.Encoding
) -> list[Boundary]:
    glued_text = seg.original()
    offsets = set(token_byte_offsets(glued_text, enc))
    out: list[Boundary] = []
    byte_pos = 0
    for i, (word, gap) in enumerate(zip(seg.words, seg.gaps, strict=False)):
        byte_pos += len(word.encode()) + len(gap.encode())
        if gap == "\n":
            continue
        with_space = _toggle(seg, i, " ")
        without_space = _toggle(seg, i, "")
        canonical_delta = len(enc.encode(without_space)) - len(enc.encode(with_space))
        right = seg.words[i + 1]
        wordwise_delta = len(enc.encode(right)) - len(enc.encode(" " + right))
        out.append(
            Boundary(
                excerpt_id=excerpt_id,
                index=i,
                left=word,
                right=right,
                dropped=gap == "",
                canonical_delta=canonical_delta,
                wordwise_delta=wordwise_delta,
                straddled=gap == "" and byte_pos not in offsets,
                left_ends_sentence=word.endswith((".", "?", "!", ";", ":")),
                right_is_identifier=_looks_like_identifier(right),
            )
        )
    return out


def _looks_like_identifier(word: str) -> bool:
    """snake_case, dotted, or mixed case after the first character.

    The mixed-case test covers PascalCase (``RuntimeContext``) as well as
    camelCase (``no_notifytrue``): requiring a lowercase first character, as an
    earlier version did, excluded every PascalCase class name in the report and
    made the category look like something the model never glues.
    """
    core = word.strip("\"'`.,?;:()[]")
    if not core:
        return False
    if "_" in core or "." in core:
        return True
    return any(c.isupper() for c in core[1:]) and any(c.islower() for c in core)
