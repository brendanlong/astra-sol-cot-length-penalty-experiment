#!/usr/bin/env bash
# Score the gpt-oss pair on a GPU. Run by the gpuc job spec; see
# scripts/gpt_oss_job.yaml.
set -euo pipefail
mkdir -p results
# Keep the HF cache on the host's persistent data dir when gpuc provides one,
# so a re-run does not download 60 GB again.
export HF_HOME="${GPUC_DATA_DIR:-$PWD/.cache}/hf"
uv run --no-sync python -m cot_spacing.emission_logprob \
    --gpt-oss --device cuda --out results/emission_logprob_gpt_oss.json
