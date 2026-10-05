#!/usr/bin/env python3
"""Download only pinned, local-inference weights into ignored project storage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "vision": {"repo": "mlx-community/Qwen3-VL-2B-Instruct-4bit", "revision": "9c4f5209e57b31f4b9dfba735de3fb983739c9cc"},
    "embedding": {"repo": "openai/clip-vit-base-patch32", "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"},
}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download
    result = {}
    for purpose, model in MODELS.items():
        patterns = ["*.json", "*.jinja", "vocab.txt", "merges.txt", "tokenizer.model"]
        patterns += ["*.safetensors"] if purpose == "vision" else ["pytorch_model.bin"]
        print(f"Downloading {purpose}: {model['repo']} at {model['revision']}", flush=True)
        path = Path(snapshot_download(repo_id=model["repo"], revision=model["revision"],
                                     cache_dir=ROOT / ".runtime/model-cache", allow_patterns=patterns, max_workers=2))
        hashes = {}
        for file in sorted(path.iterdir()):
            if not file.is_file():
                continue
            digest = hashlib.sha256()
            with file.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            hashes[file.name] = {"sha256": digest.hexdigest(), "bytes": file.stat().st_size}
        result[purpose] = {**model, "path": str(path), "files": hashes}
        (ROOT / ".runtime/multimodal-models.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Verified {purpose}: {sum(v['bytes'] for v in hashes.values())} bytes", flush=True)
    public = {key: {k: v for k, v in val.items() if k != "path"} for key, val in result.items()}
    (ROOT / "reports/multimodal-models.json").write_text(json.dumps(public, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
