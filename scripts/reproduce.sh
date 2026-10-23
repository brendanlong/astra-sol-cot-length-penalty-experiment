#!/usr/bin/env bash
# Every number, table and figure in RESULTS.md, from the committed corpus.
# No GPU, no accounts, no downloads for the first two steps.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== rebuilding the dataset from the saved report HTML =="
uv run python -m cot_spacing.extract

echo
echo "== tokenizer analysis -> cot_spacing/results/{analysis.json,report.md} =="
uv run python -m cot_spacing.analysis

echo
echo "== figures -> cot_spacing/figures/*.png =="
uv run python -m cot_spacing.figures

echo
echo "== short forms GPT-6 passed up (downloads gpt-oss-120b's embeddings, ~1.2 GB) =="
if [ "${SKIP_MODELS:-0}" != "1" ]; then
  uv run python -m cot_spacing.abbreviations
fi

echo
echo "== emission scoring, small models (downloads ~10 GB, CPU, ~15 min) =="
echo "   skip with SKIP_MODELS=1"
if [ "${SKIP_MODELS:-0}" != "1" ]; then
  uv run python -m cot_spacing.emission_logprob
fi

# The gpt-oss pair needs a GPU that runs MXFP4 natively (Hopper or Blackwell)
# and ~70 GB of VRAM for the 120b. Not part of the default reproduction:
#   uv run python -m cot_spacing.emission_logprob --gpt-oss --device cuda \
#       --out results/emission_logprob_gpt_oss.json
