"""End-to-end tests for ``tools.graphify_postprocess``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.graphify_postprocess import (
    GraphifyPostProcessResult,
    _normalize_relation,
    _normalize_label,
    postprocess_extraction,
    write_graph_json,
)


def test_normalize_relation_strips_and_lowercases() -> None:
    assert _normalize_relation("  Caused BY  ") == "caused_by"
    assert _normalize_relation(None) is None
    assert _normalize_relation("   ") is None
    assert _normalize_relation("Uses/Calls") == "uses/calls"


def test_normalize_label_collapses_whitespace() -> None:
    assert _normalize_label("  foo\tbar\n") == "foo bar"
    assert _normalize_label(None) is None
    assert _normalize_label("   ") is None


def test_drop_malformed_nodes() -> None:
    extraction = {
        "nodes": [
            {"id": "ok", "label": "OK", "type": "concept"},
            {"id": "", "label": "bad", "type": "concept"},
            {"label": "missing-id", "type": "concept"},
        ],
        "edges": [],
    }
    result = postprocess_extraction(extraction)
    assert len(result.nodes) == 1
    assert result.nodes[0]["id"] == "ok"
    assert result.metadata["dropped_nodes"] == 2


def test_deduplicate_nodes() -> None:
    extraction = {
        "nodes": [
            {"id": "dup", "label": "A"},
            {"id": "dup", "label": "B"},
        ],
        "edges": [],
    }
    result = postprocess_extraction(extraction)
    assert len(result.nodes) == 1
    assert result.metadata["dropped_nodes"] == 1


def test_drop_bad_edges_and_self_loops() -> None:
    extraction = {
        "nodes": [
            {"id": "a", "label": "A"},
            {"id": "b", "label": "B"},
            {"id": "c", "label": "C"},
        ],
        "edges": [
            {"source": "a", "target": "b", "relation": " uses "},
            {"source": "a", "target": "a", "relation": "self"},
            {"source": "x", "target": "b", "relation": "ghost"},
            {"source": "a", "target": "b", "relation": " uses "},
        ],
    }
    result = postprocess_extraction(extraction)
    assert len(result.edges) == 1
    assert result.edges[0]["source"] == "a"
    assert result.edges[0]["target"] == "b"
    assert result.edges[0]["relation"] == "uses"
    assert result.metadata["dropped_edges"] == 3


def test_singleton_reassignment_propagates_through_component(tmp_path: Path) -> None:
    extraction = {
        "nodes": [
            {"id": "hub", "label": "Hub", "community": "networking"},
            {"id": "orphan", "label": "Orphan"},
            {"id": "leaf", "label": "Leaf"},
        ],
        "edges": [
            {"source": "hub", "target": "orphan", "relation": "connects"},
            {"source": "orphan", "target": "leaf", "relation": "relates"},
        ],
    }
    result = postprocess_extraction(extraction)
    orphan = next(n for n in result.nodes if n["id"] == "orphan")
    assert orphan["community"] == "networking"
    leaf = next(n for n in result.nodes if n["id"] == "leaf")
    assert leaf["community"] == "networking"
    assert result.metadata["assigned_communities"] == 2


def test_freshness_from_source_mtime(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("x=1", encoding="utf-8")
    extraction = {
        "nodes": [{"id": "mod", "label": "Module", "source_file": "src/module.py"}],
        "edges": [],
    }
    result = postprocess_extraction(extraction, source_root=tmp_path)
    node = result.nodes[0]
    assert node["freshness"] == int(source.stat().st_mtime)
    assert result.metadata["unknown_freshness"] == 0


def test_freshness_falls_back_to_processed_at(tmp_path: Path) -> None:
    extraction = {
        "nodes": [{"id": "mod", "label": "Module", "source_file": "missing.py"}],
        "edges": [],
    }
    before = int(Path("/proc/self").stat().st_mtime)
    result = postprocess_extraction(extraction, source_root=tmp_path)
    after = int(Path("/proc/self").stat().st_mtime) + 5
    node = result.nodes[0]
    assert before <= node["freshness"] <= after
    assert result.metadata["unknown_freshness"] == 1


def test_write_graph_json_round_trip(tmp_path: Path) -> None:
    extraction = {
        "nodes": [{"id": "a", "label": "A", "type": "concept"}],
        "edges": [],
    }
    result = postprocess_extraction(extraction)
    out = write_graph_json(result, tmp_path / "graph.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["nodes"][0]["id"] == "a"
    assert "metadata" in loaded


def test_idempotent_on_second_run(tmp_path: Path) -> None:
    extraction = {
        "nodes": [
            {"id": " a ", "label": " A ", "type": "concept"},
            {"id": "b", "label": "B"},
        ],
        "edges": [
            {"source": " a ", "target": "b", "relation": " links "},
        ],
    }
    first = postprocess_extraction(extraction)
    second = postprocess_extraction(first.to_json())
    assert first.nodes == second.nodes
    assert first.edges == second.edges
    assert first.metadata["node_count"] == second.metadata["node_count"]
    assert first.metadata["edge_count"] == second.metadata["edge_count"]
    assert first.metadata["processed_at"] == second.metadata["processed_at"]


def test_end_to_end_sample_graph(tmp_path: Path) -> None:
    extraction = {
        "nodes": [
            {"id": "auth", "label": "Authentication", "community": "security"},
            {"id": "token", "label": "AccessToken", "community": "security"},
            {"id": "session", "label": "Session"},
            {"id": "api", "label": "REST API"},
            {"id": "ui", "label": "Web UI", "source_file": "ui/app.tsx"},
        ],
        "edges": [
            {"source": "auth", "target": "token", "relation": "creates"},
            {"source": "token", "target": "session", "relation": " represents "},
            {"source": "session", "target": "api", "relation": "authorizes"},
            {"source": "api", "target": "ui", "relation": "serves"},
        ],
    }
    source = tmp_path / "ui" / "app.tsx"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("", encoding="utf-8")

    result = postprocess_extraction(extraction, source_root=tmp_path)
    assert result.metadata["node_count"] == 5
    assert result.metadata["edge_count"] == 4
    session = next(n for n in result.nodes if n["id"] == "session")
    assert session["community"] == "security"
    api = next(n for n in result.nodes if n["id"] == "api")
    assert api["community"] == "security"
    ui = next(n for n in result.nodes if n["id"] == "ui")
    assert ui["freshness"] == int(source.stat().st_mtime)
    assert result.metadata["dropped_nodes"] == 0
    assert result.metadata["dropped_edges"] == 0
    assert "processed_at" in result.metadata
