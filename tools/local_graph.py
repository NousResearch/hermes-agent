"""Stage 0 local graph helpers: SQLite adjacency + json1 + FTS5.

This module is the Hermes-native graph layer for constrained Windows hosts
and aligns with graph-contract-risk-stage0 v0.1.0. It is intentionally
stdlib-only; no extra dependencies beyond Python's sqlite3/json.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


from hermes_constants import get_hermes_home

DB_FILENAME = "local_graph.sqlite"


class GraphError(Exception):
    pass


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    rel: str
    weight: float = 1.0
    properties: Dict[str, Any] | None = None


@dataclass(frozen=True)
class Node:
    node_id: str
    label: str
    properties: Dict[str, Any] | None = None


def _now() -> str:
    return "CURRENT_TIMESTAMP"


class LocalGraph:
    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            base = get_hermes_home()
            base.mkdir(parents=True, exist_ok=True)
            db_path = base / DB_FILENAME
        self.db_path = Path(db_path)
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    rel TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    properties_json TEXT NOT NULL,
                    PRIMARY KEY (source, target, rel)
                );

                CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
                CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel);
                """
            )
            try:
                conn.executescript(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                        label,
                        properties_json,
                        content=nodes,
                        content_rowid=rowid,
                        tokenize='porter unicode61'
                    );

                    CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                        INSERT INTO nodes_fts(rowid, label, properties_json)
                        VALUES (new.rowid, new.label, new.properties_json);
                    END;

                    CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                        INSERT INTO nodes_fts(nodes_fts, rowid, label, properties_json)
                        VALUES ('delete', old.rowid, old.label, old.properties_json);
                    END;

                    CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                        INSERT INTO nodes_fts(nodes_fts, rowid, label, properties_json)
                        VALUES ('delete', old.rowid, old.label, old.properties_json);
                        INSERT INTO nodes_fts(rowid, label, properties_json)
                        VALUES (new.rowid, new.label, new.properties_json);
                    END;
                    """
                )
            except sqlite3.OperationalError as exc:
                # If FTS5 is unavailable on the host sqlite build, keep the core graph working.
                if "fts5" not in str(exc).lower():
                    raise
                self._fts_available = False
            else:
                self._fts_available = True

    # --- node helpers ---

    def add_node(self, node: Node) -> None:
        with self._connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO nodes (id, label, properties_json) VALUES (?, ?, ?)",
                    (node.node_id, node.label, json.dumps(node.properties or {})),
                )
            except sqlite3.IntegrityError as exc:
                raise GraphError(f"Duplicate node id: {node.node_id}") from exc

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "label": row["label"],
                "properties": json.loads(row["properties_json"]),
                "created_at": row["created_at"],
            }

    def delete_node(self, node_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))

    # --- edge helpers ---

    def add_edge(self, edge: Edge) -> None:
        with self._connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO edges (source, target, rel, weight, properties_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        edge.source,
                        edge.target,
                        edge.rel,
                        float(edge.weight),
                        json.dumps(edge.properties or {}),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise GraphError(
                    f"Duplicate edge: ({edge.source})-[{edge.rel}]->({edge.target})"
                ) from exc

    def neighbors(self, node_id: str) -> Iterable[Tuple[str, str, float]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT source, target, rel, weight FROM edges WHERE source = ? OR target = ?",
                (node_id, node_id),
            ).fetchall()
            for row in rows:
                outgoing = row["source"] == node_id
                neighbor = row["target"] if outgoing else row["source"]
                rel = f"{row['rel']}{' (incoming)' if not outgoing else ''}"
                yield neighbor, rel, row["weight"]

    # --- search / traversal ---

    def fts_search(self, query: str, limit: int = 25) -> Iterable[Dict[str, Any]]:
        if not getattr(self, "_fts_available", False):
            raise GraphError("FTS5 is not available in this sqlite build")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT nodes.rowid, nodes.id, nodes.label, nodes.properties_json, rank FROM nodes_fts JOIN nodes ON nodes.rowid = nodes_fts.rowid WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, int(limit)),
            ).fetchall()
            for row in rows:
                yield {
                    "id": row["id"],
                    "label": row["label"],
                    "properties": json.loads(row["properties_json"]),
                    "rank": row["rank"],
                }

    def shortest_path(self, source: str, target: str, max_hops: int = 12) -> Optional[list[str]]:
        if max_hops < 1 or max_hops > 50:
            raise GraphError("max_hops must be in [1, 50] for bounded traversal")
        with self._connection() as conn:
            rows = conn.execute(
                """
                WITH RECURSIVE path(source, target, rel, seq, path_str) AS (
                    SELECT source, target, rel, 1, source || '/' || rel || '/' || target
                    FROM edges
                    WHERE source = ?
                    UNION ALL
                    SELECT edges.source, edges.target, edges.rel, path.seq + 1,
                           path.path_str || '/' || edges.rel || '/' || edges.target
                    FROM edges
                    JOIN path ON edges.source = path.target
                    WHERE path.target <> ? AND path.seq < ?
                )
                SELECT path_str, seq FROM path WHERE target = ? ORDER BY seq ASC LIMIT 1
                """,
                (source, target, max_hops, target),
            ).fetchone()
            if rows is None:
                return None
            return rows["path_str"].split("/")

    def to_dict(self) -> Dict[str, Any]:
        with self._connection() as conn:
            nodes = [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "properties": json.loads(row["properties_json"]),
                }
                for row in conn.execute("SELECT id, label, properties_json FROM nodes").fetchall()
            ]
            edges = [
                {
                    "source": row["source"],
                    "target": row["target"],
                    "rel": row["rel"],
                    "weight": row["weight"],
                    "properties": json.loads(row["properties_json"]),
                }
                for row in conn.execute("SELECT source, target, rel, weight, properties_json FROM edges").fetchall()
            ]
        return {"nodes": nodes, "edges": edges}


def json1_property_filter(rows: Iterable[Dict[str, Any]], dotpath: str, value: Any) -> Iterable[Dict[str, Any]]:
    """Filter graph rows by a json1 property path using JSON extraction semantics."""
    for row in rows:
        props = dict(row)
        obj = props.pop("properties", {}) or {}
        parts = dotpath.split(".")
        current = obj
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current == value:
            yield props


__all__ = [
    "DB_FILENAME",
    "Edge",
    "GraphError",
    "LocalGraph",
    "Node",
    "json1_property_filter",
    "json1_filter",
]


class LocalGraphToolError(Exception):
    pass


def _instance() -> LocalGraph:
    return LocalGraph()


def local_graph_add_node(node_id: str, label: str, properties: str = "{}") -> str:
    try:
        props = json.loads(properties)
    except json.JSONDecodeError as exc:
        raise LocalGraphToolError(f"Invalid properties JSON: {exc}")
    g = _instance()
    g.add_node(Node(node_id=node_id, label=label, properties=props))
    return json.dumps({"ok": True, "node_id": node_id})


def local_graph_add_edge(source: str, target: str, rel: str, weight: float = 1.0, properties: str = "{}") -> str:
    try:
        props = json.loads(properties)
    except json.JSONDecodeError as exc:
        raise LocalGraphToolError(f"Invalid properties JSON: {exc}")
    g = _instance()
    g.add_edge(Edge(source=source, target=target, rel=rel, weight=float(weight), properties=props))
    return json.dumps({"ok": True, "source": source, "target": target, "rel": rel})


def local_graph_neighbors(node_id: str) -> str:
    g = _instance()
    items = [{"neighbor": n, "rel": r, "weight": w} for n, r, w in g.neighbors(node_id)]
    return json.dumps({"node_id": node_id, "neighbors": items})


def local_graph_get_node(node_id: str) -> str:
    g = _instance()
    node = g.get_node(node_id)
    if node is None:
        return json.dumps({"node_id": node_id, "found": False})
    return json.dumps({"found": True, "node": node})


def local_graph_search(query: str, limit: int = 25) -> str:
    g = _instance()
    items = list(g.fts_search(query, limit=int(limit)))
    return json.dumps({"query": query, "results": items})


def local_graph_path(source: str, target: str, max_hops: int = 12) -> str:
    g = _instance()
    path = g.shortest_path(source, target, max_hops=int(max_hops))
    return json.dumps({"source": source, "target": target, "path": path})


try:
    from tools.registry import registry, tool_error

    registry.register(
        name="local_graph_add_node",
        toolset="local_graph",
        schema={
            "name": "local_graph_add_node",
            "description": "Add a node to the Stage 0 local graph store (SQLite adjacency + FTS5).",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Unique node id"},
                    "label": {"type": "string", "description": "Node label or type"},
                    "properties": {"type": "string", "description": "JSON object string for node metadata"},
                },
                "required": ["node_id", "label"],
            },
        },
        handler=lambda args, **kw: local_graph_add_node(
            args.get("node_id", ""), args.get("label", ""), str(args.get("properties", "{}"))
        ),
        check_fn=lambda: True,
        emoji="🕸",
    )

    registry.register(
        name="local_graph_add_edge",
        toolset="local_graph",
        schema={
            "name": "local_graph_add_edge",
            "description": "Add an edge to the Stage 0 local graph store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "rel": {"type": "string"},
                    "weight": {"type": "number", "default": 1.0},
                    "properties": {"type": "string", "description": "JSON object string for edge metadata", "default": "{}"},
                },
                "required": ["source", "target", "rel"],
            },
        },
        handler=lambda args, **kw: local_graph_add_edge(
            args.get("source", ""), args.get("target", ""), args.get("rel", ""), float(args.get("weight", 1.0)), str(args.get("properties", "{}"))
        ),
        check_fn=lambda: True,
        emoji="🔗",
    )

    registry.register(
        name="local_graph_neighbors",
        toolset="local_graph",
        schema={
            "name": "local_graph_neighbors",
            "description": "List neighbors for a node in the local graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Source node id"}
                },
                "required": ["node_id"],
            },
        },
        handler=lambda args, **kw: local_graph_neighbors(args.get("node_id", "")),
        check_fn=lambda: True,
        emoji="🧲",
    )

    registry.register(
        name="local_graph_get_node",
        toolset="local_graph",
        schema={
            "name": "local_graph_get_node",
            "description": "Get a node from the local graph store by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node id"}
                },
                "required": ["node_id"],
            },
        },
        handler=lambda args, **kw: local_graph_get_node(args.get("node_id", "")),
        check_fn=lambda: True,
        emoji="🔍",
    )

    registry.register(
        name="local_graph_search",
        toolset="local_graph",
        schema={
            "name": "local_graph_search",
            "description": "FTS5 search over local graph node labels and properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "FTS5 query string"},
                    "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 200},
                },
                "required": ["query"],
            },
        },
        handler=lambda args, **kw: local_graph_search(args.get("query", ""), int(args.get("limit", 25))),
        check_fn=lambda: True,
        emoji="🔎",
    )

    registry.register(
        name="local_graph_path",
        toolset="local_graph",
        schema={
            "name": "local_graph_path",
            "description": "Compute a shortest graph path between two nodes in the local graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "max_hops": {"type": "integer", "default": 12, "minimum": 1, "maximum": 50},
                },
                "required": ["source", "target"],
            },
        },
        handler=lambda args, **kw: local_graph_path(args.get("source", ""), args.get("target", ""), int(args.get("max_hops", 12))),
        check_fn=lambda: True,
        emoji="📐",
    )
except Exception as exc:
    logger.debug("local_graph tool registration skipped: %s", exc)

def json1_filter(rows_iterable, dotpath: str, value: Any):
    return list(json1_property_filter(rows_iterable, dotpath, value))
