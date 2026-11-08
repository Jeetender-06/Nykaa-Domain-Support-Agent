"""One-time ONLINE provisioning. Runtime always uses local_files_only=True."""

import os
from pathlib import Path

# This script intentionally does not import bootstrap (which enforces offline mode).
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download  # noqa: E402

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    destination = ROOT / "models" / "all-MiniLM-L6-v2"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=destination,
        cache_dir=ROOT / ".cache" / "huggingface",
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.safetensors",
            "1_Pooling/*",
            "README.md",
        ],
        token=False,
    )
    print(f"Downloaded pinned model {MODEL_ID}@{REVISION} to {destination}")
