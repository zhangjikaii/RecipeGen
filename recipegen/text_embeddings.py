"""本地 E5 文本编码入口；与 CLIP 视觉向量使用独立模型和索引。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from collections import OrderedDict
from threading import Lock

from .config import PROJECT_ROOT

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSION = 384
MAX_TOKENS = 512


class TextEmbeddingError(RuntimeError):
    pass


def validate_vector(vector):
    if (not isinstance(vector, list) or len(vector) != DIMENSION
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
        raise TextEmbeddingError("文本向量必须为 384 维有限数值")
    norm = math.sqrt(sum(value * value for value in vector))
    if not 0.99 <= norm <= 1.01:
        raise TextEmbeddingError("文本向量未进行单位长度归一化")
    return vector


def model_info(root=PROJECT_ROOT):
    root = Path(root).resolve()
    try:
        info = json.loads((root / ".runtime/graphrag-model.json").read_text())
        path = Path(info["path"]).resolve()
        if (info["model"] != MODEL or info["revision"] != REVISION or info["dimension"] != DIMENSION
                or not path.is_relative_to(root / ".runtime/text-model-cache")
                or not (path / "model.safetensors").is_file() or not (path / "tokenizer.json").is_file()):
            raise ValueError("invalid model")
        return info
    except (KeyError, OSError, TypeError, ValueError):
        raise TextEmbeddingError("本地多语言文本模型尚未准备完成") from None


class TextEmbedder:
    def __init__(self, root=PROJECT_ROOT):
        self.root = Path(root).resolve()
        self._cache = OrderedDict()
        self._cache_lock = Lock()
        self._encoding_lock = Lock()

    def status(self):
        try:
            info = model_info(self.root)
            ready = (self.root / ".venv-mm/bin/python").is_file()
            return {"available": ready, "name": info["model"], "revision": info["revision"],
                    "dimension": DIMENSION, "max_tokens": MAX_TOKENS, "local": True}
        except TextEmbeddingError:
            return {"available": False, "name": MODEL, "revision": REVISION, "dimension": DIMENSION, "local": True}

    def embed_query(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("查询须为 1～500 个字符")
        model_info(self.root)
        with self._cache_lock:
            if query in self._cache:
                self._cache.move_to_end(query)
                return self._cache[query][:]
        # 单个进程编码，避免同时加载多份权重；同问题只在本次服务内缓存。
        if not self._encoding_lock.acquire(blocking=False):
            raise TextEmbeddingError("本地文本编码正忙，请稍后重试")
        try:
            vector = self._encode(query)
            with self._cache_lock:
                self._cache[query] = vector[:]
                if len(self._cache) > 32:
                    self._cache.popitem(last=False)
            return vector
        finally:
            self._encoding_lock.release()

    def _encode(self, query):
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        try:
            result = subprocess.run([str(self.root / ".venv-mm/bin/python"),
                                     str(self.root / "scripts/local_text_embedding.py")],
                                    input=json.dumps({"query": query}, ensure_ascii=False),
                                    capture_output=True, text=True, cwd=self.root, env=env, timeout=90)
            payload = json.loads(result.stdout)
            if result.returncode or payload.get("status") != "ok" or payload.get("model") != MODEL or payload.get("revision") != REVISION:
                raise TextEmbeddingError("本地文本编码未完成")
            return validate_vector(payload["embedding"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.TimeoutExpired):
            raise TextEmbeddingError("本地文本编码未完成或超时") from None
