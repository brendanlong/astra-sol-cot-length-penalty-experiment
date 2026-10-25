"""Align a glued excerpt with its space-restored version.

Everything downstream works on the word sequence plus a label per gap saying
what the model actually emitted there: a space, a newline (a paragraph break in
the report), or nothing at all.
"""

from __future__ import annotations

from pydantic import BaseModel


class Segmentation(BaseModel):
    words: list[str]
    #: What separated word i from word i+1 in the model's actual output:
    #: "" (glued), " ", or "\n".
    gaps: list[str]

    @property
    def dropped(self) -> list[int]:
        return [i for i, g in enumerate(self.gaps) if g == ""]

    @property
    def kept(self) -> list[int]:
        return [i for i, g in enumerate(self.gaps) if g != ""]

    def original(self) -> str:
        return "".join(w + g for w, g in zip(self.words, [*self.gaps, ""], strict=True))

    def spaced_gaps(self) -> list[str]:
        return [g if g == "\n" else " " for g in self.gaps]

    def spaced(self) -> str:
        return "".join(
            w + g for w, g in zip(self.words, [*self.spaced_gaps(), ""], strict=True)
        )

    def merge(self, indices: set[int]) -> Segmentation:
        """Treat the given boundaries as internal to a single word.

        Used to test how much the headline depends on contestable segmentation
        calls: the report splits "severity4?" into two words while leaving
        "L2" and "P0" as one, so whether a digit label is one word or two is a
        reading, not a fact.
        """
        words: list[str] = []
        gaps: list[str] = []
        current = self.words[0]
        for i, gap in enumerate(self.gaps):
            if i in indices:
                current += gap + self.words[i + 1]
                continue
            words.append(current)
            gaps.append(gap)
            current = self.words[i + 1]
        words.append(current)
        return Segmentation(words=words, gaps=gaps)

    def runs(self) -> list[list[int]]:
        """Maximal groups of word indices that the model emitted glued together."""
        out: list[list[int]] = [[0]]
        for i, gap in enumerate(self.gaps):
            if gap == "":
                out[-1].append(i + 1)
            else:
                out.append([i + 1])
        return out


def segment(original: str, spaced: str) -> Segmentation:
    """Word-align ``original`` to ``spaced``, which differs only by added spaces.

    Raises ValueError if the two differ by anything other than spaces, which
    would mean the space-restored version is not a faithful segmentation.
    """
    words: list[str] = []
    gaps: list[str] = []
    current: list[str] = []
    i = 0  # index into original
    for ch in spaced:
        in_original = i < len(original) and original[i] == ch
        if ch in " \n":
            # A word boundary in the spaced version. The model emitted whatever
            # whitespace (possibly none) sits at this point in the original.
            if in_original:
                gap = ch
                i += 1
            elif i < len(original) and original[i] == "\n":
                gap = "\n"
                i += 1
            else:
                gap = ""
            words.append("".join(current))
            gaps.append(gap)
            current = []
            continue
        if not in_original:
            raise ValueError(
                f"spaced version differs from original by more than spaces at "
                f"original offset {i}: {original[max(0, i - 30) : i + 30]!r}"
            )
        current.append(ch)
        i += 1
    words.append("".join(current))
    if i != len(original):
        raise ValueError(f"unconsumed original text: {original[i:]!r}")
    # Empty words appear where the original itself had adjacent whitespace.
    keep = [n for n, w in enumerate(words) if w]
    if len(keep) != len(words):
        merged_gaps: list[str] = []
        for n in keep[:-1]:
            # Fold any whitespace belonging to dropped empty words into the gap.
            nxt = keep[keep.index(n) + 1]
            gap = "".join(gaps[n:nxt])
            merged_gaps.append("\n" if "\n" in gap else (" " if " " in gap else ""))
        words = [words[n] for n in keep]
        gaps = merged_gaps
    seg = Segmentation(words=words, gaps=gaps)
    if seg.original() != original:
        raise ValueError("round-trip failed")
    return seg
