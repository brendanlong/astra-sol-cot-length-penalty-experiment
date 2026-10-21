"""Extract the model-output excerpts from OpenAI's misalignment report HTML.

The report is saved under ``data/`` so the dataset is reproducible without
re-fetching; see README.md for the URL and fetch date.

Usage:
    uv run python -m cot_spacing.extract
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from cot_spacing.data import (
    DATASET_PATH,
    REPORT_PATH,
    Excerpt,
    Fragment,
    Kind,
    Speaker,
)

# The report covers two incidents, in document order. Everything between the
# first and second "What happened" sub-heading is the Sol incident.
INCIDENTS = [
    ("eval_pipeline", "GPT-6.1 Sol", "Repairing an evaluation pipeline"),
    ("reference_impl", "GPT-6 Astra", "Copying a reference implementation"),
]

# Block labels in the report mapped to (speaker, kind). "kind" distinguishes
# hidden reasoning from text the user or another agent could read.
LABELS: dict[str, tuple[Speaker, Kind]] = {
    "Parent CoT": ("parent", "cot"),
    "Subagent CoT": ("subagent", "cot"),
    "Parent message": ("parent", "agent_message"),
    "Parent message to subagent (excerpt)": ("parent", "agent_message"),
    "Subagent message": ("subagent", "agent_message"),
    "Model response (excerpt)": ("parent", "user_facing"),
    "Final response": ("parent", "user_facing"),
    "Tool call": ("subagent", "tool_call"),
    "New test": ("subagent", "tool_call"),
    "Earlier agent-to-agent instruction": ("other_agent", "agent_message"),
    "User prompt": ("user", "user_turn"),
    "User question": ("user", "user_turn"),
    "User instruction": ("user", "user_turn"),
    "User": ("user", "user_turn"),
}


def _fragments(paragraphs: list[Tag]) -> list[Fragment]:
    """Paragraph text, with the report's own "[…]" elisions marked as such."""
    out: list[Fragment] = []
    for p in paragraphs:
        text = p.get_text()
        omitted = p.has_attr("data-report-omission") or text.strip() == "[…]"
        out.append(Fragment(text=text, is_elision=omitted))
    return out


def _incident_of(el: Tag, headings: dict[str, int]) -> tuple[str, str]:
    """Which incident an element belongs to, by document position."""
    pos = _source_index(el)
    chosen = INCIDENTS[0]
    for slug, model, heading in INCIDENTS:
        if heading in headings and pos >= headings[heading]:
            chosen = (slug, model, heading)
    return chosen[0], chosen[1]


def _source_index(el: Tag) -> int:
    """The report's own character offset, used as a document-order key."""
    node: Tag | None = el
    while node is not None:
        if node.has_attr("data-source-index"):
            return int(str(node["data-source-index"]))
        first = node.find(attrs={"data-source-index": True})  # pyright: ignore[reportArgumentType]
        if isinstance(first, Tag):
            return int(str(first["data-source-index"]))
        node = node.parent if isinstance(node.parent, Tag) else None
    raise ValueError("no data-source-index found for element")


def extract(report: Path) -> list[Excerpt]:
    soup = BeautifulSoup(report.read_text(), "html.parser")

    headings: dict[str, int] = {}
    for h in soup.find_all(["h3", "h4"]):
        assert isinstance(h, Tag)
        headings[h.get_text(strip=True)] = _source_index(h)

    blocks: list[tuple[int, Tag]] = []
    for el in soup.select("section.cot-excerpt, [data-report-role-block]"):
        blocks.append((_source_index(el), el))
    blocks.sort(key=lambda kv: kv[0])

    excerpts: list[Excerpt] = []
    for order, (_, el) in enumerate(blocks):
        incident, model = _incident_of(el, headings)
        if el.name == "section":
            label_el = el.select_one(".cot-title")
            assert label_el is not None
            label = label_el.get_text(strip=True)
            panels = {
                str(p["data-cot-panel"]): _fragments(p.find_all("p"))
                for p in el.select("[data-cot-panel]")
                if isinstance(p, Tag)
            }
            original = panels["original"]
            spaced = panels.get("spaced")
            translation = panels.get("translation")
        else:
            paragraphs = [
                p for p in el.find_all("p", recursive=False) if isinstance(p, Tag)
            ]
            label = paragraphs[0].get_text(strip=True)
            original = _fragments(paragraphs[1:])
            spaced = None
            translation = None

        speaker, kind = LABELS[label]
        # The report attributes this one only to "another agent", not to the
        # incident's model.
        if speaker == "other_agent":
            model = "unspecified OpenAI agent"
        excerpts.append(
            Excerpt(
                id=f"{incident}-{order:02d}-{speaker}-{kind}",
                incident=incident,
                model=model,
                speaker=speaker,
                kind=kind,
                label=label,
                order=order,
                original=original,
                openai_spaced=spaced,
                openai_translation=translation,
            )
        )
    return excerpts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--out", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    excerpts = extract(args.report)
    payload = [e.model_dump(mode="json") for e in excerpts]
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")

    print(f"wrote {len(excerpts)} excerpts to {args.out}")
    for e in excerpts:
        has_spaced = "spaced" if e.openai_spaced else "-"
        print(f"  {e.id:<44} {e.model:<12} {has_spaced:<7} {len(e.text()):>5} chars")


if __name__ == "__main__":
    main()
