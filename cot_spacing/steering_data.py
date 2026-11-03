"""Download the steering generations and vectors from Hugging Face.

They are the GPU outputs of ``workflow/Snakefile`` (about 14 job-hours), kept
out of git. Everything downstream of them (``steering score``, the figures)
runs on CPU.

Usage:
    uv run python -m cot_spacing.steering_data
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "brendanlong/cot-spacing-steering"
#: The upload the committed scores and figures were computed from.
REVISION = "4c059a69ecbe67ef3a9b1f2f5eb18a972c140db9"
STEERING_DIR = Path(__file__).parent / "results" / "steering"


def download() -> None:
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        local_dir=STEERING_DIR,
        allow_patterns=["gen/*.jsonl", "vectors.pt"],
    )
    print(f"downloaded {REPO}@{REVISION[:7]} into {STEERING_DIR}")


if __name__ == "__main__":
    download()
