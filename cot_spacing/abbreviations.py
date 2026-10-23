"""Shorter words GPT-6 could have used in the glued text, and didn't.

Gluing saves one character per boundary. If the penalty counts characters,
writing ``msgs`` for ``messages`` saves four for one decision, and costs no
tokens. This lists the common short forms that were available, how close
gpt-oss puts each one to the word GPT-6 actually wrote, and what using them
would have saved.

Closeness is cosine similarity of gpt-oss-120b's input embeddings, given as a
rank among the tokens that are a lowercase word with a leading space (rank 1 =
the nearest such word). Without that filter the ranks are dominated by
` Messages`, `_messages` and the like. Being close is not
the same as meaning the same thing — ``lower`` is among the nearest tokens to
``higher`` — so the pairs are chosen by hand, checked in context, and the
embedding only shows that gpt-oss treats them as near-neighbours.
``CONTROLS`` are there to calibrate the ranks.

Only the embedding table is downloaded (about 1.2 GB, read straight out of
the safetensors shard); no GPU.

Usage:
    uv run python -m cot_spacing.abbreviations
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
from pathlib import Path

import numpy as np
import tiktoken
import torch
from huggingface_hub import HfFileSystem

from cot_spacing.analysis import RESULTS_DIR, glued_only, segmentations
from cot_spacing.data import load_excerpts
from cot_spacing.segment import Segmentation

MODEL = "openai/gpt-oss-120b"
EMBEDDING = "model.embed_tokens.weight"

#: Full word GPT-6 wrote in the glued excerpts -> a common short form that
#: means the same thing at every place it was used. Read in context; pairs
#: that only work in some of their uses (appropriate -> proper,
#: originally -> orig, temporarily -> temp) are left out.
PAIRS: tuple[tuple[str, str], ...] = (
    ("messages", "msgs"),
    ("automatically", "auto"),
    ("authorization", "auth"),
    ("automatic", "auto"),
    ("preference", "pref"),
    ("permission", "perm"),
    ("additional", "extra"),
    ("necessary", "needed"),
    ("original", "orig"),
    ("project", "proj"),
    ("current", "curr"),
    ("images", "imgs"),
    ("error", "err"),
    ("okay", "ok"),
)

#: Close in embedding space without being interchangeable.
CONTROLS: tuple[tuple[str, str], ...] = (
    ("higher", "lower"),
    ("big", "large"),
    ("cat", "dog"),
)

#: A sentence from the first Sol chain-of-thought, shown rewritten.
EXAMPLE_EXCERPT = "eval_pipeline-04-parent-cot"
EXAMPLE_START = ("Need", "log", "we", "write")


def letters(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def read_range(fs: HfFileSystem, path: str, start: int, end: int) -> bytes:
    data = fs.cat_file(path, start=start, end=end)
    assert isinstance(data, bytes)
    return data


def load_embeddings() -> torch.Tensor:
    """gpt-oss's input embedding table, unit-normalized, without the rest."""
    fs = HfFileSystem()
    index = json.loads(fs.read_text(f"{MODEL}/model.safetensors.index.json"))
    shard = f"{MODEL}/{index['weight_map'][EMBEDDING]}"
    (header_len,) = struct.unpack("<Q", read_range(fs, shard, 0, 8))
    tensor = json.loads(read_range(fs, shard, 8, 8 + header_len))[EMBEDDING]
    assert tensor["dtype"] == "BF16", tensor["dtype"]
    start, end = (8 + header_len + o for o in tensor["data_offsets"])
    raw = np.frombuffer(read_range(fs, shard, start, end), dtype=np.uint16)
    # bfloat16 is the top half of a float32.
    weights = torch.from_numpy(raw.astype(np.int32) << 16).view(torch.float32)
    weights = weights.reshape(tensor["shape"])
    return weights / weights.norm(dim=1, keepdim=True)


def single_token(enc: tiktoken.Encoding, word: str) -> int:
    ids = enc.encode(" " + word)
    assert len(ids) == 1, (word, ids)
    return ids[0]


def word_tokens(enc: tiktoken.Encoding) -> torch.Tensor:
    return torch.tensor(
        [
            re.fullmatch(rb" [a-z]+", enc.decode_single_token_bytes(i)) is not None
            for i in range(enc.n_vocab)
        ]
    )


def neighbour_rank(
    emb: torch.Tensor,
    words: torch.Tensor,
    enc: tiktoken.Encoding,
    word: str,
    other: str,
) -> dict[str, float]:
    a, b = single_token(enc, word), single_token(enc, other)
    sims = emb @ emb[a]
    closer = (sims > sims[b]) & words
    closer[a] = False
    return {"cosine": round(sims[b].item(), 3), "rank": int(closer.sum().item()) + 1}


def example(seg: Segmentation) -> dict[str, str | int]:
    n = len(EXAMPLE_START)
    i = next(
        k for k in range(len(seg.words)) if tuple(seg.words[k : k + n]) == EXAMPLE_START
    )
    j = next(k for k in range(i, len(seg.words)) if seg.words[k].endswith("?"))
    sentence = Segmentation(words=seg.words[i : j + 1], gaps=seg.gaps[i:j])
    short = dict(PAIRS)

    def shorten(word: str) -> str:
        key = letters(word)
        return word.replace(key, short[key]) if key in short else word

    abbreviated = " ".join(shorten(w) for w in sentence.words)
    return {
        "as_written": sentence.original(),
        "spaced": sentence.spaced(),
        "abbreviated": abbreviated,
        "chars_as_written": len(sentence.original()),
        "chars_spaced": len(sentence.spaced()),
        "chars_abbreviated": len(abbreviated),
    }


def run() -> dict[str, object]:
    enc = tiktoken.get_encoding("o200k_harmony")
    segs = glued_only(segmentations(load_excerpts()))
    used = collections.Counter(letters(w) for _, s in segs for w in s.words)
    emb = load_embeddings()
    words = word_tokens(enc)[: emb.shape[0]]

    pairs = []
    for long, short in PAIRS:
        n = used[long]
        assert n, f"{long!r} does not occur in the glued excerpts"
        pairs.append(
            {
                "word": long,
                "short": short,
                "uses": n,
                "chars_saved": n * (len(long) - len(short)),
                "tokens_saved": n
                * (len(enc.encode(" " + long)) - len(enc.encode(" " + short))),
                **neighbour_rank(emb, words, enc, long, short),
            }
        )
    controls = [
        {"word": a, "other": b, **neighbour_rank(emb, words, enc, a, b)}
        for a, b in CONTROLS
    ]
    by_id = {e.id: s for e, s in segs}
    return {
        "model": MODEL,
        "n_word_tokens": int(words.sum().item()),
        "n_glued_excerpts": len(segs),
        "chars_saved_by_gluing": sum(len(s.dropped) for _, s in segs),
        "chars_saved_by_short_forms": sum(p["chars_saved"] for p in pairs),
        "tokens_saved_by_short_forms": sum(p["tokens_saved"] for p in pairs),
        "short_forms_used": {
            short: used[short] for short in sorted({s for _, s in PAIRS}) if used[short]
        },
        "pairs": pairs,
        "controls": controls,
        "example": example(by_id[EXAMPLE_EXCERPT]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "abbreviations.json")
    args = parser.parse_args()
    results = run()
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    for p in results["pairs"]:  # type: ignore[union-attr]
        print(
            f"{p['word']:>14} ×{p['uses']} -> {p['short']:<6} "
            f"rank {p['rank']:>4}  cos {p['cosine']:.2f}  saves {p['chars_saved']}"
        )
    print(
        f"short forms would save {results['chars_saved_by_short_forms']} characters; "
        f"gluing saved {results['chars_saved_by_gluing']}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
