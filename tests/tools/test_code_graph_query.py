from tools.code_graph.indexer import index_repo
from tools.code_graph.query import (
    context_for_goal,
    impact_for_paths,
    neighbors_for_symbol,
    search_symbols,
    symbol_detail,
)
from tools.code_graph.storage import CodeGraphStore


def test_search_symbols_returns_compact_matches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    index_repo(repo, store=store)

    result = search_symbols(repo, "help", store=store, limit=5)

    assert result["success"] is True
    assert result["matches"][0]["name"] == "helper"
    assert result["matches"][0]["path"] == "sample.py"


def test_symbol_detail_returns_definition_location(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    index_repo(repo, store=store)

    result = symbol_detail(repo, "helper", store=store)

    assert result["success"] is True
    assert result["symbol"]["name"] == "helper"
    assert result["symbol"]["path"] == "sample.py"
    assert result["symbol"]["start_line"] == 1


def test_neighbors_for_symbol_returns_imports_and_references(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("import json\n\ndef helper():\n    return 1\n", encoding="utf-8")
    (repo / "consumer.py").write_text("from sample import helper\n\nvalue = helper()\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    index_repo(repo, store=store)

    result = neighbors_for_symbol(repo, "helper", store=store)

    assert result["success"] is True
    assert {"module": "json"} in result["imports"]
    assert any(item["path"] == "consumer.py" for item in result["calls"])


def test_impact_for_paths_returns_symbols_imports_and_likely_tests(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "thing.py").write_text(
        "import json\n\ndef helper():\n    return 1\n",
        encoding="utf-8",
    )
    (repo / "src" / "consumer.py").write_text(
        "from src.thing import helper\n\nvalue = helper()\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_thing.py").write_text(
        "def test_helper():\n    assert True\n",
        encoding="utf-8",
    )
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    index_repo(repo, store=store)

    result = impact_for_paths(repo, ["src/thing.py"], store=store)

    assert result["success"] is True
    assert result["symbols"][0]["name"] == "helper"
    assert {"path": "src/thing.py", "module": "json"} in result["imports"]
    assert "src/consumer.py" in result["referencing_files"]
    assert "tests/test_thing.py" in result["likely_tests"]


def test_context_for_goal_returns_budgeted_bundle(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def handle_function_call():\n    return 'ok'\n", encoding="utf-8")
    store = CodeGraphStore(tmp_path / "graph.sqlite")
    store.initialize()
    index_repo(repo, store=store)

    result = context_for_goal(
        repo,
        "change handle function call behavior",
        store=store,
        budget_chars=2000,
    )

    assert result["success"] is True
    assert result["symbols"]
    assert len(str(result)) < 3000

