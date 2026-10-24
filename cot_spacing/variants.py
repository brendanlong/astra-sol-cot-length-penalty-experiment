"""Four renderings of each glued chain-of-thought, for contrasting activations.

- ``spaced``: OpenAI's own space-restored reading.
- ``glued``: what GPT-6 actually wrote.
- ``short``: the spaced reading with every word in :data:`SHORT_FORMS` shortened.
- ``both``: what GPT-6 wrote, with the same words shortened.

``spaced`` is the baseline each of the other three is contrasted with, so the
three directions differ only in what was done to the same words.

Each is wrapped as the analysis channel of a gpt-oss conversation whose user
turn is the user prompt from the same incident, so the excerpt reads as the
model's own reasoning about that request.

Usage:
    uv run python -m cot_spacing.variants
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from cot_spacing.analysis import RESULTS_DIR, glued_only, segmentations
from cot_spacing.data import Excerpt, load_excerpts

if TYPE_CHECKING:
    from cot_spacing.segment import Segmentation

Variant = Literal["spaced", "glued", "short", "both"]
VARIANTS: tuple[Variant, ...] = ("spaced", "glued", "short", "both")

#: Every everyday short form for a word in the glued excerpts that reads the
#: same in context, plus common ones for words gpt-oss writes in arithmetic
#: reasoning, so the steering evaluation can see the behaviour there. A short
#: form is only counted where it stands as its own word (not inside
#: ``no_notify`` or ``RuntimeContext``).
SHORT_FORMS: dict[str, str] = {
    # in the glued excerpts
    "authorization": "auth",
    "authorized": "authed",
    "authorizes": "auths",
    "unauthorized": "unauthed",
    "automatically": "auto",
    "automatic": "auto",
    "automatics": "autos",
    "notifications": "notifs",
    "notification": "notif",
    "functionality": "func",
    "functional": "func",
    "architecture": "arch",
    "instructions": "instrs",
    "instruction": "instr",
    "instruct": "instr",
    "appropriate": "approp",
    "conditional": "cond",
    "definitions": "defs",
    "interpretation": "interp",
    "engineering": "eng",
    "requirement": "req",
    "required": "reqd",
    "requires": "reqs",
    "temporarily": "temp",
    "additional": "addl",
    "dependency": "dep",
    "monitoring": "mon",
    "monitor": "mon",
    "monitors": "mons",
    "originally": "orig",
    "original": "orig",
    "permission": "perm",
    "preference": "pref",
    "assertion": "assert",
    "auxiliary": "aux",
    "canonical": "canon",
    "developer": "dev",
    "framework": "fw",
    "necessary": "nec",
    "recipient": "recip",
    "external": "ext",
    "metadata": "meta",
    "messages": "msgs",
    "optional": "opt",
    "physical": "phys",
    "priority": "prio",
    "separate": "sep",
    "severity": "sev",
    "standard": "std",
    "current": "curr",
    "default": "dflt",
    "system": "sys",
    "unknown": "unk",
    "remove": "rm",
    "server": "srv",
    "error": "err",
    "images": "imgs",
    "okay": "ok",
    "with": "w/",
    # common elsewhere, including in arithmetic reasoning
    "message": "msg",
    "information": "info",
    "repository": "repo",
    "directory": "dir",
    "environment": "env",
    "parameters": "params",
    "arguments": "args",
    "function": "fn",
    "implementation": "impl",
    "reference": "ref",
    "configuration": "config",
    "database": "db",
    "request": "req",
    "response": "resp",
    "number": "num",
    "numbers": "nums",
    "though": "tho",
    "through": "thru",
    "probably": "prob",
    "without": "w/o",
    "because": "bc",
    "question": "q",
    "answer": "ans",
    "calculate": "calc",
    "calculation": "calc",
    "minutes": "mins",
    "seconds": "secs",
    "hours": "hrs",
    "maximum": "max",
    "minimum": "min",
    "average": "avg",
    "approximately": "approx",
    "previous": "prev",
    "command": "cmd",
    "variable": "var",
    "package": "pkg",
    "between": "btwn",
    "difference": "diff",
    "different": "diff",
    "percent": "pct",
    "people": "ppl",
    "something": "sth",
}

#: A letters-only run that is a whole word, not part of an identifier.
_WORD = re.compile(r"(?<![\w/])[A-Za-z]+(?![\w/])")


def shorten(text: str) -> str:
    def replace(m: re.Match[str]) -> str:
        word = m.group(0)
        short = SHORT_FORMS.get(word.lower())
        if short is None:
            return word
        return short[0].upper() + short[1:] if word[0].isupper() else short

    return _WORD.sub(replace, text)


def render(seg: Segmentation, variant: Variant) -> str:
    words = (
        [shorten(w) for w in seg.words] if variant in ("short", "both") else seg.words
    )
    gaps = seg.gaps if variant in ("glued", "both") else seg.spaced_gaps()
    return "".join(w + g for w, g in zip(words, [*gaps, ""], strict=True))


class Example(BaseModel):
    id: str
    user: str
    texts: dict[Variant, str]
    glued_boundaries: int
    shortened_words: int
    #: Characters each variant saves against ``spaced``.
    chars_saved: dict[Variant, int]


def user_prompt(excerpt: Excerpt, excerpts: list[Excerpt]) -> str:
    """The incident's first user turn, redactions kept as the report shows them."""
    turn = min(
        (
            e
            for e in excerpts
            if e.incident == excerpt.incident and e.kind == "user_turn"
        ),
        key=lambda e: e.order,
    )
    return "\n".join(f.text for f in turn.original if not f.is_elision)


def build() -> list[Example]:
    excerpts = load_excerpts()
    out = []
    for e, seg in glued_only(segmentations(excerpts)):
        if e.kind != "cot":
            continue
        texts: dict[Variant, str] = {v: render(seg, v) for v in VARIANTS}
        out.append(
            Example(
                id=e.id,
                user=user_prompt(e, excerpts),
                texts=texts,
                glued_boundaries=len(seg.dropped),
                shortened_words=sum(shorten(w) != w for w in seg.words),
                chars_saved={v: len(texts["spaced"]) - len(texts[v]) for v in VARIANTS},
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "variants.json")
    args = parser.parse_args()
    examples = build()
    args.out.write_text(
        json.dumps([e.model_dump() for e in examples], indent=1, ensure_ascii=False)
        + "\n"
    )
    for e in examples:
        print(
            f"{e.id:<45} glued {e.glued_boundaries:>3}  shortened "
            f"{e.shortened_words:>3}  chars saved {e.chars_saved}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
