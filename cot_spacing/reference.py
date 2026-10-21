"""Space-restored reference for each model-generated excerpt.

For the six chain-of-thought excerpts the report itself ships a "Spaces added"
view, which is an exact space-insertion of the original (``tests`` check that).
That is the reference used here.

The report provides no such view for agent messages, so the two glued ones are
segmented by hand below. The remaining model outputs have no glued word
boundaries at all; they are listed in ``ALREADY_SPACED`` so that "not glued" is
a recorded judgement rather than a silent default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cot_spacing.data import join_paragraphs

if TYPE_CHECKING:
    from cot_spacing.data import Excerpt

#: One entry per excerpt, as the report's paragraphs, hand-spaced.
MANUAL_SPACED: dict[str, tuple[str, ...]] = {
    "reference_impl-12-other_agent-agent_message": (
        "Do not disable monitor even temporarily",
    ),
    "reference_impl-14-parent-agent_message": (
        "User explicitly authorizes: copy [reference PR] model→tool→model serving"
        " path, keep all current tool definitions/functionality/bias.",
        "Preserve current goal/system/developer prompt. Preserve physical"
        " runtime/cancel/no-replay protections and monitoring as appropriate to"
        " exact PR rather than inventing new required monitor framework.",
    ),
}

ALREADY_SPACED = frozenset(
    {
        "eval_pipeline-02-parent-user_facing",
        "eval_pipeline-05-parent-agent_message",
        "eval_pipeline-08-parent-agent_message",
        "eval_pipeline-09-parent-agent_message",
        "eval_pipeline-11-parent-user_facing",
        "reference_impl-18-subagent-agent_message",
        "reference_impl-21-parent-user_facing",
    }
)


def reference_spaced(excerpt: Excerpt) -> str | None:
    """The space-restored text, or None if this excerpt has no reference."""
    openai_version = excerpt.spaced_text()
    if openai_version is not None:
        return openai_version
    if excerpt.id in MANUAL_SPACED:
        return join_paragraphs(MANUAL_SPACED[excerpt.id])
    if excerpt.id in ALREADY_SPACED:
        return excerpt.text()
    return None
