#!/usr/bin/env python3
"""单次离线查询编码；模型日志写 stderr，stdout 仅输出 JSON。"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipegen.text_embeddings import MODEL, REVISION, MAX_TOKENS, model_info, validate_vector


def load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    info = model_info(ROOT)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(info["path"], local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(info["path"], local_files_only=True, trust_remote_code=False, use_safetensors=True)
    # 查询使用 CPU，减少与用户其他 GPU 工作的竞争；全库构建可显式指定 MPS。
    model.eval()
    return tokenizer, model


def encode(texts, tokenizer, model, *, prefix, device="cpu"):
    import torch
    tokens = tokenizer([prefix + text for text in texts], max_length=MAX_TOKENS,
                       padding=True, truncation=True, return_tensors="pt")
    tokens = {name: value.to(device) for name, value in tokens.items()}
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).bool()
        pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).float().cpu().tolist()
    return [validate_vector(vector) for vector in vectors]


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    try:
        request = json.loads(sys.stdin.read(8000))
        query = request["query"]
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("invalid query")
        with contextlib.redirect_stdout(sys.stderr):
            tokenizer, model = load_encoder()
            vector = encode([query], tokenizer, model, prefix="query: ")[0]
        print(json.dumps({"status": "ok", "model": MODEL, "revision": REVISION, "embedding": vector}))
        return 0
    except Exception:
        print(json.dumps({"status": "unavailable", "model": MODEL, "revision": REVISION}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
