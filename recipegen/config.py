from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """Load local .env without evaluating shell expressions or overriding env."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key.replace("_", "").isalnum():
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


@dataclass
class Settings:
    graph_backend: str = "sqlite"
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "var" / "recipegen.sqlite3")
    example_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "example_graph.json")
    llm_provider: str = "disabled"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_timeout: float = 30.0
    context_candidates: int = 12
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = field(default="", repr=False)
    neo4j_database: str = "neo4j"
    trace_path: Path | None = None
    retrieval_mode: str = "hybrid"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            graph_backend=os.getenv("RECIPEGEN_GRAPH_BACKEND", "sqlite"),
            db_path=Path(os.getenv("RECIPEGEN_DB_PATH", str(PROJECT_ROOT / "var" / "recipegen.sqlite3"))),
            example_path=Path(os.getenv("RECIPEGEN_DATA_PATH", str(PROJECT_ROOT / "data" / "example_graph.json"))),
            llm_provider=os.getenv("RECIPEGEN_LLM_PROVIDER", "disabled"),
            llm_base_url=os.getenv("RECIPEGEN_LLM_BASE_URL", ""),
            llm_model=os.getenv("RECIPEGEN_LLM_MODEL", ""),
            llm_api_key=os.getenv("RECIPEGEN_LLM_API_KEY", ""),
            llm_timeout=float(os.getenv("RECIPEGEN_LLM_TIMEOUT", "30")),
            context_candidates=int(os.getenv("RECIPEGEN_CONTEXT_CANDIDATES", "12")),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            trace_path=Path(os.environ["RECIPEGEN_TRACE_PATH"]) if os.getenv("RECIPEGEN_TRACE_PATH") else None,
            retrieval_mode=os.getenv("RECIPEGEN_RETRIEVAL_MODE", "hybrid"),
        )

    def __post_init__(self) -> None:
        if self.graph_backend not in {"sqlite", "neo4j"}:
            raise ValueError("RECIPEGEN_GRAPH_BACKEND 必须是 sqlite 或 neo4j")
        if self.llm_provider not in {"disabled", "chat_completions", "ollama"}:
            raise ValueError("不支持的模型接口类型")
        if self.retrieval_mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("检索模式须为 keyword、semantic 或 hybrid")
        if not 1 <= self.context_candidates <= 30 or not 1 <= self.llm_timeout <= 120:
            raise ValueError("上下文候选数须为 1～30，模型超时须为 1～120 秒")

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "disabled":
            return False
        if not self.llm_base_url or not self.llm_model:
            return False
        return self.llm_provider == "ollama" or bool(self.llm_api_key)
