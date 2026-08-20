import json

from tools import code_graph_tool
from tools.registry import discover_builtin_tools, registry
from toolsets import TOOLSETS, resolve_toolset


def test_code_graph_status_tool_returns_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    repo = tmp_path / "repo"
    repo.mkdir()

    payload = json.loads(code_graph_tool.code_graph_status(str(repo)))

    assert payload["success"] is True
    assert payload["graph_status"] in {"missing", "fresh", "stale"}


def test_code_graph_index_tool_indexes_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    payload = json.loads(code_graph_tool.code_graph_index(str(repo)))

    assert payload["success"] is True
    assert payload["files_seen"] == 1
    assert str(tmp_path / "hermes_home") in payload["cache_path"]


def test_code_graph_search_symbol_neighbors_impact_and_context_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("import json\n\ndef helper():\n    return 1\n", encoding="utf-8")
    (repo / "consumer.py").write_text("from sample import helper\n\nvalue = helper()\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_sample.py").write_text("def test_helper():\n    assert True\n", encoding="utf-8")
    json.loads(code_graph_tool.code_graph_index(str(repo)))

    search_payload = json.loads(code_graph_tool.code_graph_search("help", str(repo)))
    symbol_payload = json.loads(code_graph_tool.code_graph_symbol("helper", str(repo)))
    neighbors_payload = json.loads(code_graph_tool.code_graph_neighbors("helper", str(repo)))
    impact_payload = json.loads(code_graph_tool.code_graph_impact(["sample.py"], str(repo)))
    context_payload = json.loads(
        code_graph_tool.code_graph_context("change helper behavior", str(repo), budget_chars=2000)
    )

    assert search_payload["matches"][0]["name"] == "helper"
    assert symbol_payload["symbol"]["path"] == "sample.py"
    assert any(item["path"] == "consumer.py" for item in neighbors_payload["calls"])
    assert "tests/test_sample.py" in impact_payload["likely_tests"]
    assert context_payload["recommended_files"]


def test_code_graph_index_rejects_missing_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    payload = json.loads(code_graph_tool.code_graph_index(str(tmp_path / "missing")))

    assert payload["success"] is False
    assert "root does not exist" in payload["error"]


def test_code_graph_toolset_is_opt_in():
    assert "code_graph" in TOOLSETS
    tools = set(resolve_toolset("code_graph"))
    assert {
        "code_graph_index",
        "code_graph_status",
        "code_graph_search",
        "code_graph_symbol",
        "code_graph_neighbors",
        "code_graph_impact",
        "code_graph_context",
    }.issubset(tools)

    assert "code_graph_index" not in TOOLSETS["hermes-cli"]["tools"]


def test_code_graph_tool_module_registers_tools():
    discover_builtin_tools()
    names = set(registry.get_tool_names_for_toolset("code_graph"))

    assert {
        "code_graph_index",
        "code_graph_status",
        "code_graph_search",
        "code_graph_symbol",
        "code_graph_neighbors",
        "code_graph_impact",
        "code_graph_context",
    }.issubset(names)

