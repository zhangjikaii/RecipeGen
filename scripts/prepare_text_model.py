#!/usr/bin/env python3
"""下载固定版本的 E5 文本模型；不下载 RecipeGen 媒体或运行 LLM。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.text_embeddings import MODEL, REVISION, DIMENSION, MAX_TOKENS


def main():
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=str(ROOT / ".runtime/text-model-cache"),
                             allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                             "sentencepiece.bpe.model", "special_tokens_map.json"], max_workers=2)
    info = {"model": MODEL, "revision": REVISION, "dimension": DIMENSION, "max_tokens": MAX_TOKENS,
            "path": str(Path(path).resolve()), "pooling": "attention_mask_mean", "normalization": "l2",
            "query_prefix": "query: ", "passage_prefix": "passage: ", "status": "downloaded"}
    target = ROOT / ".runtime/graphrag-model.json"
    target.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    target.chmod(0o600)
    print(json.dumps({key: info[key] for key in ("model", "revision", "dimension", "status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
