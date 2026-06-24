import json
from types import SimpleNamespace

from tools.semantic_index_tool import SEMANTIC_INDEX_SCHEMA, semantic_index


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["input"][0]
        base = float(len(text) % 10)
        vector = [base, 1.0, 0.5]
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=vector)],
            usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
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
        return [dict(row, _distance=0.0) for row in self.rows[: self.n]]


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


def test_schema_exposes_safe_first_stage_actions():
    params = SEMANTIC_INDEX_SCHEMA["parameters"]["properties"]
    assert params["action"]["enum"] == ["dry_run", "stage", "query", "stats"]
    assert params["embed"]["default"] is False
    assert "payload" in params


def test_dry_run_does_not_require_embedding_or_lancedb(tmp_path):
    result = json.loads(semantic_index(payload="hello semantic world", db_path=str(tmp_path)))
    assert result["success"] is True
    assert result["action"] == "dry_run"
    assert result["would_call_openai"] is False
    assert result["would_write_lancedb"] is False
    assert result["payload"]["char_count"] == len("hello semantic world")


def test_stage_requires_explicit_embed_true(tmp_path):
    result = json.loads(semantic_index(action="stage", payload="hello", db_path=str(tmp_path)))
    assert result["success"] is False
    assert "embed=true" in result["error"]


def test_stage_refuses_secret_like_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = json.loads(semantic_index(
        action="stage",
        embed=True,
        payload="OPENAI_API_KEY=sk-supersecretvalue1234567890",
        db_path=str(tmp_path),
        lancedb_module=_FakeLanceDB(),
        openai_client_cls=_FakeOpenAI,
    ))
    assert result["success"] is False
    assert "credentials" in result["error"]


def test_stage_embeds_and_writes_lancedb(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_lancedb = _FakeLanceDB()
    result = json.loads(semantic_index(
        action="stage",
        embed=True,
        payload="tiny payload for vector storage",
        corpus="manual",
        uri="manual://tiny",
        db_path=str(tmp_path),
        lancedb_module=fake_lancedb,
        openai_client_cls=_FakeOpenAI,
    ))
    assert result["success"] is True
    assert result["dimensions"] == 3
    assert result["row_count"] == 1
    assert result["record"]["uri"] == "manual://tiny"

    stats = json.loads(semantic_index(
        action="stats",
        db_path=str(tmp_path),
        lancedb_module=fake_lancedb,
    ))
    assert stats["table_exists"] is True
    assert stats["row_count"] == 1
    assert stats["sample"][0]["vector_dimensions"] == 3


def test_query_embeds_query_and_returns_vector_db_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_lancedb = _FakeLanceDB()
    json.loads(semantic_index(
        action="stage",
        embed=True,
        payload="find me with semantic query",
        db_path=str(tmp_path),
        lancedb_module=fake_lancedb,
        openai_client_cls=_FakeOpenAI,
    ))
    result = json.loads(semantic_index(
        action="query",
        query="semantic query",
        db_path=str(tmp_path),
        lancedb_module=fake_lancedb,
        openai_client_cls=_FakeOpenAI,
    ))
    assert result["success"] is True
    assert result["count"] == 1
    assert result["results"][0]["vector_dimensions"] == 3
    assert "semantic query" in result["results"][0]["text_preview"]
