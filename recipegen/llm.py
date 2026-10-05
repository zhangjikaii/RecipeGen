from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Explicit opt-in client for chat-completions-compatible APIs and Ollama."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport
        self.calls: list[dict] = []

    def json(self, system: str, payload: dict[str, Any], stage: str) -> dict:
        if not self.settings.llm_configured:
            raise LLMError("模型接口尚未配置；本次使用规则解析与可追溯模板")
        base = self.settings.llm_base_url.rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LLMError("模型接口地址必须是 http(s) URL")
        if parsed.username or parsed.password:
            raise LLMError("模型 URL 不应包含凭据，请使用专用 API_KEY 配置")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_provider == "ollama":
            url = base + "/api/chat"
            body = {"model": self.settings.llm_model, "messages": messages, "stream": False, "format": "json", "options": {"temperature": 0}}
        else:
            url = base + "/chat/completions"
            headers["Authorization"] = "Bearer " + self.settings.llm_api_key
            body = {"model": self.settings.llm_model, "messages": messages, "response_format": {"type": "json_object"}}
        try:
            with httpx.Client(timeout=self.settings.llm_timeout, transport=self.transport, trust_env=False) as client:
                response = client.post(url, json=body, headers=headers)
            response.raise_for_status()
            raw = response.json()
            content = raw["message"]["content"] if self.settings.llm_provider == "ollama" else raw["choices"][0]["message"]["content"]
            if len(content) > 50_000:
                raise LLMError("模型响应超过大小限制")
            if content.strip().startswith("```"):
                raise LLMError("模型未按要求返回纯 JSON")
            result = json.loads(content)
            if not isinstance(result, dict):
                raise LLMError("模型返回的 JSON 必须是对象")
            self.calls.append({"stage": stage, "provider": self.settings.llm_provider, "model": self.settings.llm_model, "usage": raw.get("usage"), "success": True})
            return result
        except LLMError:
            raise
        except httpx.HTTPStatusError as error:
            # Do not echo upstream body or URL, which may contain secrets.
            raise LLMError(f"模型接口返回 HTTP {error.response.status_code}") from None
        except httpx.TimeoutException:
            raise LLMError("模型接口调用超时") from None
        except httpx.RequestError:
            raise LLMError("模型接口连接失败") from None
        except (KeyError, TypeError, ValueError):
            raise LLMError("模型响应格式无效") from None
