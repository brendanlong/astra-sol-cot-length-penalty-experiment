"""Loading gpt-oss-20b for activation work, and laying out a conversation.

The checkpoint ships its experts in MXFP4. They are dequantized to bf16 here so
the same model takes gradients and runs on cards without native MXFP4 (the
A40s this is run on).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import Mxfp4Config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"
#: The template stamps today's date into the system message; pinning it keeps
#: activations and generations reproducible.
DATE = "2026-10-20"


def load() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=Mxfp4Config(dequantize=True),
    )
    model.eval()
    model.requires_grad_(False)
    return tok, model


def prompt(tok: PreTrainedTokenizerBase, user: str) -> str:
    """Everything before the model's first token of reasoning."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True,
        tokenize=False,
        reasoning_effort=REASONING_EFFORT,
        strftime_now=lambda _fmt: DATE,
    )
    assert isinstance(text, str)
    return text


def with_reasoning(
    tok: PreTrainedTokenizerBase, user: str, cot: list[int]
) -> tuple[list[int], slice]:
    """Token ids of ``user`` answered by the reasoning ids ``cot``.

    Returns the ids and the slice of them that is ``cot``. The reasoning comes
    already tokenized, so it can be tokenized some way other than canonically.
    """
    head = tok.encode(
        prompt(tok, user) + "<|channel|>analysis<|message|>", add_special_tokens=False
    )
    tail = tok.encode("<|end|>", add_special_tokens=False)
    return head + cot + tail, slice(len(head), len(head) + len(cot))


def layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    return model.model.layers  # type: ignore[union-attr]


@contextmanager
def steer(model: PreTrainedModel, layer: int, vector: torch.Tensor) -> Iterator[None]:
    """Add ``vector`` to the residual stream after ``layer``, at every position."""
    block = layers(model)[layer]

    def hook(
        _module: torch.nn.Module, _inputs: object, output: torch.Tensor
    ) -> torch.Tensor:
        return output + vector.to(output.device, output.dtype)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
