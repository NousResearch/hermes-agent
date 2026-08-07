import json

from tools.code_graph.indexer import index_repo
from tools.code_graph.query import graph_status
from tools.code_graph.storage import CodeGraphStore


def test_index_repo_extracts_python_symbols(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "sample.py"
    source.write_text(
        '''"""Module doc."""

class Greeter:
    """Greeter docs."""
    def hello(self, name: str) -> str:
        """Say hello."""
        return f"hello {name}"


def helper(value: int) -> int:
    return value + 1
''',
        encoding="utf-8",
    )
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()

    summary = index_repo(repo, store=store)

    assert summary["files_indexed"] == 1
    with store.connect() as conn:
        symbols = conn.execute("SELECT name, kind FROM symbols ORDER BY name").fetchall()
    assert [(row["name"], row["kind"]) for row in symbols] == [
        ("Greeter", "class"),
        ("hello", "function"),
        ("helper", "function"),
    ]


def test_index_repo_records_python_import_edges(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("import json\nfrom pathlib import Path\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()

    index_repo(repo, store=store)

    with store.connect() as conn:
        edges = conn.execute("SELECT edge_type, metadata_json FROM edges").fetchall()
    imports = [
        json.loads(row["metadata_json"])["module"]
        for row in edges
        if row["edge_type"] == "imports"
    ]
    assert imports == ["json", "pathlib"]


def test_index_repo_records_textual_reference_edges(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def helper():\n    return helper()\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()

    index_repo(repo, store=store)

    with store.connect() as conn:
        edges = conn.execute("SELECT edge_type, metadata_json FROM edges").fetchall()
    names = {
        json.loads(row["metadata_json"])["name"]
        for row in edges
        if row["edge_type"] in {"references", "calls"}
    }
    assert "helper" in names


def test_graph_status_reports_missing_before_index(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()

    status = graph_status(repo, store=store)

    assert status["graph_status"] == "missing"


def test_graph_status_reports_stale_after_file_change(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "sample.py"
    path.write_text("def one():\n    return 1\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    index_repo(repo, store=store)

    path.write_text("def two():\n    return 2\n", encoding="utf-8")
    status = graph_status(repo, store=store)

    assert status["graph_status"] == "stale"
    assert status["stale_files"] == 1


def test_index_repo_skips_large_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    big = repo / "big.py"
    big.write_text("x = '" + ("a" * 600_000) + "'\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()

    summary = index_repo(repo, store=store, max_file_size_bytes=512_000)

    assert summary["files_seen"] == 0
    assert summary["skipped_files"] == 1


def test_force_reindex_rebuilds_unchanged_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()

    first = index_repo(repo, store=store)
    second = index_repo(repo, store=store)
    forced = index_repo(repo, store=store, force=True)

    assert first["files_indexed"] == 1
    assert second["files_indexed"] == 0
    assert forced["files_indexed"] == 1

