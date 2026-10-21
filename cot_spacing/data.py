"""Dataset of model-output excerpts from OpenAI's misalignment report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence

EXP_DIR = Path(__file__).parent
REPORT_PATH = EXP_DIR / "corpus" / "openai_report_2026-10-20.html"
DATASET_PATH = EXP_DIR / "corpus" / "excerpts.json"

#: The report replaces identifiers and internal names with a bracketed
#: paraphrase ("[runtime sample identifier]", "[redacted]"). Those characters,
#: their internal spaces and the boundaries either side of them are the report
#: authors' text, not the model's, so they cannot be measured.
EDITORIAL = re.compile(r"\[[^]]*\]|\[\u2026\]")

Speaker = Literal["parent", "subagent", "other_agent", "user"]
Kind = Literal["cot", "agent_message", "user_facing", "tool_call", "user_turn"]


class Fragment(BaseModel):
    """One paragraph of an excerpt.

    ``is_elision`` marks the report's own "[…]" placeholders, which are the
    report authors' text rather than the model's, and are excluded from every
    length and tokenization metric.
    """

    text: str
    is_elision: bool


class Excerpt(BaseModel):
    id: str
    incident: str
    model: str
    speaker: Speaker
    kind: Kind
    label: str
    order: int
    original: list[Fragment]
    openai_spaced: list[Fragment] | None
    openai_translation: list[Fragment] | None

    def text(self) -> str:
        """The model's own text, elisions dropped, fragments joined by newline."""
        return _join(self.original)

    def spaced_text(self) -> str | None:
        """OpenAI's own space-restored rendering, if the report provides one."""
        if self.openai_spaced is None:
            return None
        return _join(self.openai_spaced)


def join_paragraphs(paragraphs: Sequence[str]) -> str:
    """Normalize raw paragraphs the way :func:`_join` normalizes an excerpt.

    Takes paragraphs rather than one string because the paragraph breaks are
    load-bearing: they are where the model's text is not contiguous, and
    collapsing them would invent a word boundary.
    """
    return _join([Fragment(text=p, is_elision=False) for p in paragraphs])


def _join(fragments: list[Fragment]) -> str:
    """Model text only, one line per contiguous stretch of it.

    Every editorial insertion breaks the line rather than being kept or simply
    deleted. Keeping it would measure the report authors' spacing; deleting it
    would invent a word boundary that was never written. Breaking the line
    drops the two boundaries either side, which is right: what the model wrote
    around a redacted identifier is not observable.
    """
    lines: list[str] = []
    for f in fragments:
        if f.is_elision:
            continue
        for part in EDITORIAL.split(f.text):
            collapsed = " ".join(part.split())
            if collapsed:
                lines.append(collapsed)
    return "\n".join(lines)


def load_excerpts(path: Path = DATASET_PATH) -> list[Excerpt]:
    return [Excerpt.model_validate(d) for d in json.loads(path.read_text())]


def model_outputs(excerpts: list[Excerpt]) -> list[Excerpt]:
    """Excerpts that are model-generated prose (not user turns or code diffs)."""
    return [e for e in excerpts if e.kind in ("cot", "agent_message", "user_facing")]
