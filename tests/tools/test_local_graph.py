"""Contract tests for tools/local_graph.py.

Covers: node/edge CRUD, duplicate rejection, neighbors, FTS search fallback
when unavailable, shortest path no-route, shortest path max_hops boundary.

These tests use only stdlib sqlite3 and tmp_path. They avoid Hermes config
by supplying an explicit db_path to LocalGraph.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Isolate tests from Hermes config by stubbing hermes_constants before
# importing tools.local_graph. We never call get_hermes_home() because each
# LocalGraph instance receives an explicit db_path from pytest's tmp_path.
# ---------------------------------------------------------------------------
if "hermes_constants" not in sys.modules:
    _fake = types.ModuleType("hermes_constants")

    def _get_hermes_home() -> Path:
        return Path("/tmp/fake-hermes-home")

    _fake.get_hermes_home = _get_hermes_home  # type: ignore[attr-defined]
    sys.modules["hermes_constants"] = _fake

from tools.local_graph import (  # noqa: E402
    Edge,
    GraphError,
    LocalGraph,
    Node,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def graph(tmp_path: Path) -> LocalGraph:
    db = tmp_path / "local_graph.sqlite"
    return LocalGraph(db_path=db)


def _n(id_: str, label: str, props: dict | None = None) -> Node:
    return Node(node_id=id_, label=label, properties=props)


def _e(src: str, tgt: str, rel: str, weight: float = 1.0, props: dict | None = None) -> Edge:
    return Edge(source=src, target=tgt, rel=rel, weight=weight, properties=props)


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

class TestNodeCRUD:
    def test_add_node_then_get_node(self, graph: LocalGraph) -> None:
        graph.add_node(_n("a", "alpha", {"v": 1}))
        node = graph.get_node("a")
        assert node["id"] == "a"
        assert node["label"] == "alpha"
        assert node["properties"] == {"v": 1}
        assert "created_at" in node and isinstance(node["created_at"], str)

    def test_get_node_missing_returns_none(self, graph: LocalGraph) -> None:
        assert graph.get_node("missing") is None

    def test_delete_node_removes_edges(self, graph: LocalGraph) -> None:
        graph.add_node(_n("a", "alpha"))
        graph.add_node(_n("b", "beta"))
        graph.add_edge(_e("a", "b", "knows"))
        graph.delete_node("a")

        assert graph.get_node("a") is None
        assert graph.get_node("b")["id"] == "b"
        rows = graph.to_dict()["edges"]
        assert rows == []


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------

class TestEdgeCRUD:
    def test_add_edge_persists(self, graph: LocalGraph) -> None:
        graph.add_node(_n("x", "X"))
        graph.add_node(_n("y", "Y"))
        graph.add_edge(_e("x", "y", "links", weight=2.5, props={"k": "v"}))

        edges = graph.to_dict()["edges"]
        assert len(edges) == 1
        assert edges[0]["source"] == "x"
        assert edges[0]["target"] == "y"
        assert edges[0]["rel"] == "links"
        assert edges[0]["weight"] == 2.5
        assert edges[0]["properties"] == {"k": "v"}

    def test_neighbors_outgoing(self, graph: LocalGraph) -> None:
        graph.add_node(_n("src", "S"))
        graph.add_node(_n("tgt", "T"))
        graph.add_edge(_e("src", "tgt", "to", weight=1.0))

        nb = list(graph.neighbors("src"))
        assert len(nb) == 1
        neighbor, rel, weight = nb[0]
        assert neighbor == "tgt"
        assert rel == "to"
        assert weight == 1.0

    def test_neighbors_incoming(self, graph: LocalGraph) -> None:
        graph.add_node(_n("s", "S"))
        graph.add_node(_n("t", "T"))
        graph.add_edge(_e("s", "t", "to"))

        nb = list(graph.neighbors("t"))
        assert len(nb) == 1
        neighbor, rel, weight = nb[0]
        assert neighbor == "s"
        assert "incoming" in rel
        assert weight == 1.0

    def test_neighbors_unknown_returns_empty(self, graph: LocalGraph) -> None:
        assert list(graph.neighbors("nope")) == []


# ---------------------------------------------------------------------------
# Duplicate rejection
# ---------------------------------------------------------------------------

class TestDuplicateRejection:
    def test_duplicate_node_raises_graph_error(self, graph: LocalGraph) -> None:
        graph.add_node(_n("n", "label"))
        with pytest.raises(GraphError, match="Duplicate node id: n"):
            graph.add_node(_n("n", "label again"))

    def test_duplicate_edge_raises_graph_error(self, graph: LocalGraph) -> None:
        graph.add_node(_n("a", "A"))
        graph.add_node(_n("b", "B"))
        graph.add_edge(_e("a", "b", "r"))
        with pytest.raises(GraphError, match=r"Duplicate edge: \(a\)\-\[r\]->\(b\)"):
            graph.add_edge(_e("a", "b", "r"))


# ---------------------------------------------------------------------------
# FTS search fallback when unavailable
# ---------------------------------------------------------------------------

class TestFTSFallback:
    def test_fts_search_unavailable_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "no_fts.sqlite"
        g = LocalGraph(db_path=db)

        g.add_node(Node(node_id="n1", label="hello"))
        assert g.get_node("n1")["label"] == "hello"

        # Toggle the fts availability flag to exercise the fallback branch
        # on hosts where FTS5 is compiled-in by default.
        g._fts_available = False

        with pytest.raises(GraphError, match="not available"):
            list(g.fts_search("hello"))


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------

class TestShortestPath:
    def _build_line(self, graph: LocalGraph, nodes: list[str]) -> None:
        for idx, label in enumerate(["A", "B", "C", "D"]):
            graph.add_node(_n(nodes[idx], label))
        for idx in range(len(nodes) - 1):
            graph.add_edge(_e(nodes[idx], nodes[idx + 1], "next", weight=1.0))

    def test_no_route_returns_none(self, graph: LocalGraph) -> None:
        graph.add_node(_n("a", "A"))
        graph.add_node(_n("b", "B"))
        assert graph.shortest_path("a", "b") is None

    def test_max_hops_blocks_longer_path(self, graph: LocalGraph) -> None:
        nodes = ["n0", "n1", "n2", "n3", "n4"]
        self._build_line(graph, nodes)

        assert len(graph.shortest_path("n0", "n4", max_hops=5)) > 1
        assert graph.shortest_path("n0", "n4", max_hops=2) is None

    def test_max_hops_boundary_errors_at_limit(self, graph: LocalGraph) -> None:
        with pytest.raises(GraphError, match="max_hops"):
            graph.shortest_path("x", "y", max_hops=0)

        with pytest.raises(GraphError, match="max_hops"):
            graph.shortest_path("x", "y", max_hops=51)

    def test_same_source_target_returns_none(self, graph: LocalGraph) -> None:
        graph.add_node(_n("solo", "S"))
        path = graph.shortest_path("solo", "solo")
        assert path is None
