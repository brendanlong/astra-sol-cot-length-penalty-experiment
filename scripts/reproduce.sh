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
echo "== emission scoring (downloads ~10 GB of open models, CPU, ~15 min) =="
echo "   skip with SKIP_MODELS=1"
if [ "${SKIP_MODELS:-0}" != "1" ]; then
  uv run python -m cot_spacing.emission_logprob
fi
