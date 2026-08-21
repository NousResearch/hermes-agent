"""Focused validation tests for the Graphify MCP server.

These tests prove the tools import/run deterministically without external
dependencies. We build a tiny in-memory graph, persist it to a temp
graph.json, then call the MCP tool wrappers directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.graphify.model import GraphJsonRepository, GraphModel, GraphNode, GraphEdge
from tools.graphify_mcp_server import (
    _TOOLS,
    _get_community,
    _get_neighbors,
    _get_node,
    _god_nodes,
    _graph_stats,
    _query_graph,
    _resolve_graph_path,
    _shortest_path,
    create_mcp_server,
)


def _build_sample_graph(tmp_path: Path) -> Path:
    model = GraphModel()
    model.add_node(GraphNode(label="repo", kind="module", properties={"path": "."}))
    model.add_node(GraphNode(label="auth", kind="concept", properties={"team": "platform"}))
    model.add_node(GraphNode(label="token", kind="concept", properties={"team": "platform"}))
    model.add_node(GraphNode(label="ui", kind="concept", properties={"team": "web"}))
    model.add_edge(GraphEdge(source="auth", target="token", relation="creates"))
    model.add_edge(GraphEdge(source="token", target="ui", relation="authorizes"))
    model.add_edge(GraphEdge(source="auth", target="ui", relation="authorizes"))

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    out_dir = repo_dir / "graphify-out"
    out_dir.mkdir()
    repository = GraphJsonRepository(output_dir=str(out_dir))
    repository.save(model)
    return out_dir / "graph.json"


def test_resolve_graph_path_prefers_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "custom" / "graph.json"
    explicit.parent.mkdir()
    explicit.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(explicit))
    assert _resolve_graph_path() == explicit


def test_tool_names_are_registered() -> None:
    assert set(_TOOLS) == {
        "query_graph",
        "get_node",
        "get_neighbors",
        "shortest_path",
        "get_community",
        "god_nodes",
        "graph_stats",
    }


def test_query_graph_returns_json_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = _build_sample_graph(tmp_path)
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(graph_path))

    payload = _query_graph("auth")
    data = json.loads(payload)
    assert "nodes" in data
    assert "path" in data
    assert isinstance(data["nodes"], list)
    assert data["mode"] == "bfs"


def test_get_node_by_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = _build_sample_graph(tmp_path)
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(graph_path))

    payload = _get_node("auth")
    data = json.loads(payload)
    assert data["label"] == "auth"
    assert data["kind"] == "concept"
    assert data["properties"]["team"] == "platform"


def test_get_neighbors_directed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = _build_sample_graph(tmp_path)
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(graph_path))

    payload = _get_neighbors("auth")
    neighbors = json.loads(payload)
    labels = {item["label"] for item in neighbors}
    assert labels == {"token", "ui"}
    relations = {item["relation"] for item in neighbors}
    assert relations == {"creates", "authorizes"}


def test_shortest_path_between_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = _build_sample_graph(tmp_path)
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(graph_path))

    payload = _shortest_path("auth", "ui")
    data = json.loads(payload)
    assert data["path"] == ["auth", "ui"]
    assert data["max_hops"] == 8


def test_get_community_returns_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = _build_sample_graph(tmp_path)
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(graph_path))

    payload = _get_community(0)
    data = json.loads(payload)
    assert isinstance(data, list)
    assert len(data) == 0


def test_god_nodes_sorted_by_degree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = _build_sample_graph(tmp_path)
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(graph_path))

    payload = _god_nodes(top_n=2)
    data = json.loads(payload)
    assert len(data) == 2
    assert data[0]["degree"] >= data[1]["degree"]
    assert data[0]["label"] == "auth"
    assert data[0]["degree"] == 2


def test_graph_stats_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = _build_sample_graph(tmp_path)
    monkeypatch.setenv("GRAPHIFY_GRAPH_PATH", str(graph_path))

    payload = _graph_stats()
    data = json.loads(payload)
    assert data["node_count"] == 4
    assert data["edge_count"] == 3
    assert "avg_degree" in data
    assert "density" in data


def test_create_mcp_server_lists_registered_tools() -> None:
    import asyncio

    server = create_mcp_server()
    listed = asyncio.run(server.list_tools())
    names = {tool.name for tool in listed}
    assert names == set(_TOOLS)
