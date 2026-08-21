"""Focused validation tests for the Graphify indexing bridge.

These tests verify:

* **Stable output** — the same input graph always produces identical
  ``FILE-MAP.md`` and ``map/effects/CONTEXT.md`` across repeated runs.
* **Missing-input fallback** — when ``graph.json`` is absent the bridge
  writes a clearly labeled empty artifact instead of raising.
* **Integration with existing Graphify tests** — the bridge is exercised
  from a real persisted graph built with the same patterns used in the
  existing Graphify test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.graphify.model import GraphEdge, GraphJsonRepository, GraphModel, GraphNode
from tools.graphify_indexing_bridge import (
    GraphifyIndexingBridgeResult,
    render_context_md,
    render_file_map,
    rebuild_indexing_bridge,
)


def _build_sample_graph(tmp_path: Path) -> Path:
    graph = GraphModel()
    graph.add_node(GraphNode(label="repo", kind="module", properties={"path": "."}))
    graph.add_node(GraphNode(label="auth", kind="concept", properties={"team": "platform"}))
    graph.add_node(GraphNode(label="token", kind="concept", properties={"team": "platform"}))
    graph.add_node(GraphNode(label="ui", kind="concept", properties={"team": "web"}))
    graph.add_edge(GraphEdge(source="repo", target="auth", relation="contains"))
    graph.add_edge(GraphEdge(source="repo", target="ui", relation="contains"))
    graph.add_edge(GraphEdge(source="auth", target="token", relation="creates"))
    graph.add_edge(GraphEdge(source="token", target="ui", relation="authorizes"))

    out_dir = tmp_path / "graphify-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    repository = GraphJsonRepository(output_dir=str(out_dir))
    repository.save(graph)
    return out_dir / "graph.json"


def test_file_map_stable_output(tmp_path: Path) -> None:
    nodes = [
        {"id": "repo", "label": "repo", "source_file": "src/main.py"},
        {"id": "auth", "label": "auth", "source_file": "src/auth.py"},
        {"id": "auth", "label": "auth", "source_file": "src/auth.py"},
        {"id": "ui", "label": "ui", "source_file": "ui/app.tsx"},
    ]
    first = render_file_map(nodes)
    second = render_file_map(nodes)
    assert first == second
    assert first.count("|") == 15
    assert "src/auth.py" in first


def test_context_md_stable_output(tmp_path: Path) -> None:
    nodes = [
        {"id": "repo", "label": "repo", "community": "root"},
        {"id": "auth", "label": "auth", "community": "security"},
        {"id": "token", "label": "token", "community": "security"},
    ]
    edges = [
        {"source": "repo", "target": "auth", "relation": "contains"},
        {"source": "auth", "target": "token", "relation": "creates"},
    ]
    first = render_context_md(nodes, edges)
    second = render_context_md(nodes, edges)
    assert first == second
    assert "## Communities" in first
    assert "## Adjacency" in first


def test_missing_graph_falls_back_to_empty_artifacts(tmp_path: Path) -> None:
    result = rebuild_indexing_bridge(
        tmp_path / "missing" / "graph.json",
        tmp_path / "map",
    )
    assert result.file_map_path.exists()
    assert result.context_path.exists()
    assert result.file_map_text.startswith("# FILE MAP")
    assert "missing input graph" in result.file_map_text
    assert result.context_text.startswith("# EFFECTS CONTEXT")
    assert "missing input graph" in result.context_text


def test_idempotent_on_second_run(tmp_path: Path) -> None:
    graph_path = _build_sample_graph(tmp_path)
    output_root = tmp_path / "map"
    first = rebuild_indexing_bridge(graph_path, output_root)
    second = rebuild_indexing_bridge(graph_path, output_root)

    assert first.file_map_text == second.file_map_text
    assert first.context_text == second.context_text
    assert first.file_map_path.read_text(encoding="utf-8") == second.file_map_path.read_text(encoding="utf-8")
    assert first.context_path.read_text(encoding="utf-8") == second.context_path.read_text(encoding="utf-8")


def test_integration_with_existing_graphify_tests(tmp_path: Path) -> None:
    graph_path = _build_sample_graph(tmp_path)
    output_root = tmp_path / "map"
    result = rebuild_indexing_bridge(graph_path, output_root)

    assert result.file_map_path.exists()
    assert result.context_path.exists()

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    node_count = len(payload.get("nodes") or [])
    edge_count = len(payload.get("edges") or [])

    assert "_node_count: 4_" in result.context_text
    assert "_edge_count: 4_" in result.context_text
    assert "_total source files: 0_" in result.file_map_text


def test_rebuild_indexing_bridge_result_shape() -> None:
    result = GraphifyIndexingBridgeResult()
    assert result.file_map_path is None
    assert result.context_path is None
    assert result.file_map_text == ""
    assert result.context_text == ""
