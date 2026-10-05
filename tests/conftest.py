from pathlib import Path

import pytest

from recipegen.config import PROJECT_ROOT, Settings
from recipegen.graph import SQLiteGraphStore, load_document


@pytest.fixture
def graph_store(tmp_path):
    store = SQLiteGraphStore(tmp_path / "graph.sqlite3")
    store.import_document(load_document(PROJECT_ROOT / "data" / "example_graph.json"))
    return store


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=tmp_path / "graph.sqlite3")
