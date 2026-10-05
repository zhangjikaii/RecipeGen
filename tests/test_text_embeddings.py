"""验证离线编码进程边界，不加载权重或连接网络。"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from recipegen import text_embeddings as module


def vector():
    return [1.0] + [0.0] * 383


@pytest.mark.parametrize("bad", [[True] + [0.0] * 383, [float("nan")] * 384,
                                  [float("inf")] * 384, [0.0] * 384, [1.0] * 383])
def test_invalid_worker_vectors_are_rejected(bad):
    with pytest.raises(module.TextEmbeddingError):
        module.validate_vector(bad)


def test_query_cache_is_local_and_does_not_expose_mutable_vector(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "model_info", lambda root: {})
    calls = []
    def run(_command, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "ok", "model": module.MODEL,
                               "revision": module.REVISION, "embedding": vector()}))
    monkeypatch.setattr(module.subprocess, "run", run)
    embedder = module.TextEmbedder(tmp_path)
    result = embedder.embed_query("清爽的凉菜")
    result[0] = 0
    assert embedder.embed_query("清爽的凉菜") == vector()
    assert len(calls) == 1
    assert calls[0]["env"]["HF_HUB_OFFLINE"] == "1"
    assert calls[0]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert json.loads(calls[0]["input"])["query"] == "清爽的凉菜"


@pytest.mark.parametrize("result", [
    SimpleNamespace(returncode=0, stdout=json.dumps({"status": "ok", "model": module.MODEL,
                    "revision": "0" * 40, "embedding": vector()})),
    SimpleNamespace(returncode=0, stdout="[]"),
    SimpleNamespace(returncode=1, stdout="upstream private secret"),
])
def test_worker_identity_and_json_failures_are_safe(tmp_path, monkeypatch, result):
    monkeypatch.setattr(module, "model_info", lambda root: {})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(module.TextEmbeddingError) as error:
        module.TextEmbedder(tmp_path).embed_query("凉菜")
    assert "private secret" not in str(error.value)


def test_timeout_releases_encoding_slot_for_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "model_info", lambda root: {})
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("private command", 90)
    monkeypatch.setattr(module.subprocess, "run", timeout)
    embedder = module.TextEmbedder(tmp_path)
    with pytest.raises(module.TextEmbeddingError):
        embedder.embed_query("凉菜")
    assert embedder._encoding_lock.acquire(blocking=False)
    embedder._encoding_lock.release()


def test_busy_encoder_does_not_launch_another_model(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "model_info", lambda root: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("second model must not launch")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    embedder = module.TextEmbedder(tmp_path)
    embedder._encoding_lock.acquire()
    try:
        with pytest.raises(module.TextEmbeddingError, match="正忙"):
            embedder.embed_query("凉菜")
    finally:
        embedder._encoding_lock.release()
