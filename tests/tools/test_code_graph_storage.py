from pathlib import Path

from tools.code_graph.models import Edge, FileRecord, Symbol
from tools.code_graph.storage import CodeGraphStore, cache_path_for_root


def test_code_graph_models_are_plain_data_containers():
    file_record = FileRecord(
        id=1,
        repo_id=1,
        path="tools/example.py",
        language="python",
        size=120,
        mtime_ns=123,
        sha256="abc",
        indexed_at=456.0,
    )
    symbol = Symbol(
        id=1,
        repo_id=1,
        file_id=1,
        name="example",
        qualname="tools.example.example",
        kind="function",
        start_line=10,
        end_line=12,
        signature="example()",
        docstring="Example function.",
    )
    edge = Edge(
        id=1,
        repo_id=1,
        source_kind="symbol",
        source_id=1,
        target_kind="symbol",
        target_id=2,
        edge_type="calls",
        metadata_json="{}",
    )

    assert file_record.path == "tools/example.py"
    assert symbol.kind == "function"
    assert edge.edge_type == "calls"


def test_cache_path_for_root_uses_repo_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    root = tmp_path / "repo"
    root.mkdir()

    cache_path = cache_path_for_root(root)

    assert cache_path.parent.name == "code_graph"
    assert cache_path.name.endswith(".sqlite")
    assert str(cache_path).startswith(str(tmp_path / "hermes_home"))


def test_store_initializes_schema(tmp_path):
    db_path = tmp_path / "graph.sqlite"
    store = CodeGraphStore(db_path)
    store.initialize()

    tables = store.table_names()

    assert {"repos", "files", "symbols", "edges", "chunks", "index_runs"}.issubset(tables)


def test_store_ensure_repo_returns_stable_id(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()

    first = store.ensure_repo(root)
    second = store.ensure_repo(Path(root))

    assert first == second
    assert store.get_repo_id(root) == first

