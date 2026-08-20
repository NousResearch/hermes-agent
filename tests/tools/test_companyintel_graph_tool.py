import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def companyintel_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "companyintel"))
    monkeypatch.setenv("HERMES_PROFILE", "companyintel")
    return Path(tmp_path / "companyintel")


def test_companyintel_graph_tool_is_profile_gated(companyintel_home):
    from tools.companyintel_graph_tool import check_companyintel_requirements

    assert check_companyintel_requirements() is True


def test_init_run_persists_canonical_graph_and_checkpoint(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    result = json.loads(companyintel_graph({
        "action": "init_run",
        "run_id": "run_test_001",
        "target_url": "https://example.com",
    }))

    assert result["ok"] is True
    graph_path = companyintel_home / "companyintel" / "runs" / "run_test_001" / "graph.json"
    checkpoint_path = companyintel_home / "companyintel" / "runs" / "run_test_001" / "checkpoints" / "latest.json"
    assert graph_path.exists()
    assert checkpoint_path.exists()
    graph = json.loads(graph_path.read_text())
    assert graph["schema_version"] == "corporate-intelligence-graph/v1"
    assert graph["target"]["domain"] == "example.com"
    assert graph["research_state"]["status"] == "RUNNING"


def test_observation_creates_durable_node_evidence_edge_and_typed_frontier(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    init = json.loads(companyintel_graph({"action": "init_run", "run_id": "run_test_002", "target_url": "https://example.com"}))
    result = json.loads(companyintel_graph({
        "action": "record_observation",
        "run_id": "run_test_002",
        "node_type": "phone",
        "value": "+380 50 123 4567",
        "source_url": "https://example.com/contact",
        "excerpt": "Call +380 50 123 4567",
        "relation_from_node_id": init["seed_node_id"],
        "relation": "publishes",
    }))

    assert result["ok"] is True
    graph_path = companyintel_home / "companyintel" / "runs" / "run_test_002" / "graph.json"
    graph = json.loads(graph_path.read_text())
    assert len(graph["nodes"]) >= 2
    assert any(node["node_type"] == "phone" for node in graph["nodes"])
    assert len(graph["evidence"]) == 1
    assert len(graph["edges"]) == 1
    assert len(graph["frontier"]) >= 1
    assert graph["search_log"] == []
    assert graph["research_state"]["round"] == 0


def test_checkpoint_is_atomic_and_summary_is_compact(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    companyintel_graph({"action": "init_run", "run_id": "run_test_003", "target_url": "https://example.com"})
    result = json.loads(companyintel_graph({"action": "summary", "run_id": "run_test_003"}))

    assert result["ok"] is True
    assert result["run_id"] == "run_test_003"
    assert result["nodes"] == 1
    assert result["edges"] == 0
    assert result["evidence"] == 0
    assert result["frontier_open"] == 1
    assert result["search_log"] == 0
    assert result["round"] == 0
    assert result["status"] == "RUNNING"
    assert result["retry_count"] == 0
    assert result["resume_count"] == 0
    assert result["worker_checkpoints"] == []


def test_non_companyintel_profile_does_not_expose_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    from tools.companyintel_graph_tool import check_companyintel_requirements

    assert check_companyintel_requirements() is False
