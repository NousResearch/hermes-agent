import json
import re
from types import SimpleNamespace

from tools.codebase_index_tool import (
    CODEBASE_INDEX_SCHEMA,
    CODEBASE_SEARCH_SCHEMA,
    codebase_index,
    codebase_search,
)


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        vectors = []
        for index, text in enumerate(inputs):
            base = float(len(text) % 10)
            vectors.append(SimpleNamespace(index=index, embedding=[base, 1.0, 0.5]))
        return SimpleNamespace(
            data=vectors,
            usage=SimpleNamespace(prompt_tokens=len(inputs), total_tokens=len(inputs)),
        )


class _FakeOpenAI:
    embeddings = _FakeEmbeddings()

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeSearch:
    def __init__(self, rows):
        self.rows = rows
        self.n = 5

    def limit(self, n):
        self.n = n
        return self

    def to_list(self):
        return [dict(row, _distance=0.1) for row in self.rows[: self.n]]


class _FakeTable:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def add(self, rows):
        self.rows.extend(rows)

    def count_rows(self):
        return len(self.rows)

    def head(self, n):
        return [dict(row) for row in self.rows[:n]]

    def search(self, _vector):
        return _FakeSearch(self.rows)

    def limit(self, n):
        return _FakeSearch(self.rows).limit(n)

    def delete(self, predicate):
        repo_match = re.search(r"repo_id = '([^']+)'", predicate)
        path_match = re.search(r"path = '([^']+)'", predicate)
        repo_id = repo_match.group(1) if repo_match else None
        path = path_match.group(1) if path_match else None
        self.rows = [
            row
            for row in self.rows
            if not (
                (repo_id is None or row.get("repo_id") == repo_id)
                and (path is None or row.get("path") == path)
            )
        ]


class _FakeDB:
    def __init__(self):
        self.tables = {}

    def table_names(self):
        return list(self.tables)

    def open_table(self, name):
        return self.tables[name]

    def create_table(self, name, data):
        table = _FakeTable(data)
        self.tables[name] = table
        return table


class _FakeLanceDB:
    def __init__(self):
        self.dbs = {}

    def connect(self, path):
        return self.dbs.setdefault(path, _FakeDB())


def _write_sample_repo(tmp_path, text=None):
    source = text or '''
"""sample module"""

import os


def greet(name):
    return f"hello {name}"


class Worker:
    def run(self):
        return "semantic codebase search"
'''
    path = tmp_path / "app.py"
    path.write_text(source.strip() + "\n")
    (tmp_path / "README.md").write_text("# Search Notes\n\nLanceDB stores code chunks.\n")
    return tmp_path


def test_codebase_index_schema_exposes_safe_actions():
    params = CODEBASE_INDEX_SCHEMA["parameters"]["properties"]
    assert params["action"]["enum"] == ["dry_run", "index", "stats"]
    assert params["embed"]["default"] is False
    assert CODEBASE_SEARCH_SCHEMA["parameters"]["properties"]["mode"]["default"] == "hybrid"


def test_codebase_index_dry_run_chunks_python_and_markdown(tmp_path):
    root = _write_sample_repo(tmp_path)
    result = json.loads(
        codebase_index(
            root=str(root),
            scope="repo",
            max_files=10,
            max_chunks=20,
            db_path=str(tmp_path / "db"),
        )
    )
    assert result["success"] is True
    assert result["would_call_openai"] is False
    assert result["chunk_count"] >= 3
    kinds = {chunk["chunk_kind"] for chunk in result["sample_chunks"]}
    assert {"module_context", "function"} & kinds


def test_codebase_index_requires_explicit_embed_true(tmp_path):
    root = _write_sample_repo(tmp_path)
    result = json.loads(
        codebase_index(
            action="index",
            root=str(root),
            scope="paths",
            paths=["app.py"],
            db_path=str(tmp_path / "db"),
        )
    )
    assert result["success"] is False
    assert "embed=true" in result["error"]


def test_codebase_index_embeds_rows_and_stats_hides_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _write_sample_repo(tmp_path)
    fake_lancedb = _FakeLanceDB()

    result = json.loads(
        codebase_index(
            action="index",
            embed=True,
            root=str(root),
            scope="paths",
            paths=["app.py"],
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
            openai_client_cls=_FakeOpenAI,
        )
    )
    assert result["success"] is True
    assert result["stored_chunk_count"] >= 2
    assert result["dimensions"] == 3
    assert result["row_count"] == result["stored_chunk_count"]

    stats = json.loads(
        codebase_index(
            action="stats",
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
        )
    )
    assert stats["table_exists"] is True
    assert "vector" not in stats["sample"][0]
    assert stats["sample"][0]["vector_dimensions"] == 3


def test_codebase_index_deletes_stale_rows_for_reindexed_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _write_sample_repo(tmp_path, "def alpha():\n    return 'old'\n")
    fake_lancedb = _FakeLanceDB()

    first = json.loads(
        codebase_index(
            action="index",
            embed=True,
            root=str(root),
            scope="paths",
            paths=["app.py"],
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
            openai_client_cls=_FakeOpenAI,
        )
    )
    (tmp_path / "app.py").write_text("def beta():\n    return 'new semantic body'\n")
    second = json.loads(
        codebase_index(
            action="index",
            embed=True,
            root=str(root),
            scope="paths",
            paths=["app.py"],
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
            openai_client_cls=_FakeOpenAI,
        )
    )
    assert first["row_count"] == first["stored_chunk_count"]
    assert second["row_count"] == second["stored_chunk_count"]


def test_codebase_index_deletes_rows_for_deleted_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _write_sample_repo(tmp_path, "def alpha():\n    return 'old'\n")
    fake_lancedb = _FakeLanceDB()

    first = json.loads(
        codebase_index(
            action="index",
            embed=True,
            root=str(root),
            scope="paths",
            paths=["app.py"],
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
            openai_client_cls=_FakeOpenAI,
        )
    )
    (tmp_path / "app.py").unlink()
    second = json.loads(
        codebase_index(
            action="index",
            embed=True,
            root=str(root),
            scope="paths",
            paths=["app.py"],
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
            openai_client_cls=_FakeOpenAI,
        )
    )
    assert first["row_count"] > 0
    assert second["stored_chunk_count"] == 0
    assert second["row_count"] == 0


def test_codebase_search_hybrid_falls_back_to_keyword(tmp_path):
    root = _write_sample_repo(tmp_path)
    result = json.loads(
        codebase_search(
            query="where is LanceDB mentioned",
            root=str(root),
            mode="hybrid",
            paths=["README.md"],
            db_path=str(tmp_path / "missing-db"),
            max_files=10,
            max_chunks=20,
        )
    )
    assert result["success"] is True
    assert result["semantic_available"] is False
    assert result["semantic_error"]
    assert result["results"][0]["path"] == "README.md"


def test_codebase_search_semantic_returns_indexed_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _write_sample_repo(tmp_path)
    fake_lancedb = _FakeLanceDB()
    json.loads(
        codebase_index(
            action="index",
            embed=True,
            root=str(root),
            scope="paths",
            paths=["app.py"],
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
            openai_client_cls=_FakeOpenAI,
        )
    )
    result = json.loads(
        codebase_search(
            query="semantic codebase search worker",
            root=str(root),
            mode="semantic",
            db_path=str(tmp_path / "db"),
            lancedb_module=fake_lancedb,
            openai_client_cls=_FakeOpenAI,
        )
    )
    assert result["success"] is True
    assert result["count"] >= 1
    assert result["results"][0]["path"] == "app.py"
    assert result["results"][0]["vector_dimensions"] == 3
