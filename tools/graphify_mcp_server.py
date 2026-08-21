"""Graphify MCP server — deterministic, local-only.

Exposes the local graphify implementation as MCP tools over stdio. No
network dependencies. Graph path resolution order:

1. ``GRAPHIFY_GRAPH_PATH`` env var (absolute/relative JSON file).
2. ``GRAPHIFY_PROJECT_DIR`` env var (repo root, default cwd), resolved
   to ``<project_dir>/graphify-out/graph.json``.
3. CWD, resolved to ``./graphify-out/graph.json``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MCP SDK import — keep import lazy so the module can be imported
# in environments without `mcp` installed (tests, tooling).
# ---------------------------------------------------------------------------
_MCP_SERVER_AVAILABLE = False
_MCP_SERVER_CLASS: Any = None
try:  # pragma: no branch - import shim
    from mcp.server import MCPServer as _MCP_SERVER_CLASS

    _MCP_SERVER_AVAILABLE = True
except Exception:
    try:  # pragma: no cover - legacy fallback
        from mcp.server.fastmcp import FastMCP as _MCP_SERVER_CLASS

        _MCP_SERVER_AVAILABLE = True
    except Exception:
        _MCP_SERVER_CLASS = None


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------


def _resolve_graph_path() -> Optional[Path]:
    graph_path = os.environ.get("GRAPHIFY_GRAPH_PATH")
    if graph_path:
        return Path(graph_path)

    project_dir = os.environ.get("GRAPHIFY_PROJECT_DIR") or os.getcwd()
    return Path(project_dir) / "graphify-out" / "graph.json"


def _load_graph():
    path = _resolve_graph_path()
    if path is None or not path.exists():
        return None, f"graph file not found: {path}"

    try:
        # Local implementation only — no external graphify package.
        from src.graphify.model import GraphJsonRepository

        repo = GraphJsonRepository(output_dir=str(path.parent))
        model = repo.load()
        return model, None
    except Exception as exc:  # pragma: no cover - fail-fast boundary
        logger.debug("graph load failed", exc_info=True)
        return None, f"graph load failed: {exc}"


def _get_query_engine():
    model, err = _load_graph()
    if model is None:
        raise RuntimeError(err)

    # Local query implementation only.
    from src.graphify.query import GraphifyQueryEngine

    return GraphifyQueryEngine(model)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _query_graph(question: str, mode: str = "bfs", depth: int = 3, token_budget: int = 2000) -> str:
    engine = _get_query_engine()
    result = engine.query(question, mode=mode, depth=depth, token_budget=token_budget)
    import json

    return json.dumps(result.to_dict(), ensure_ascii=False)


def _get_node(label: str) -> str:
    engine = _get_query_engine()
    node = engine.get_node(label)
    import json

    return json.dumps(
        {"label": node.label, "kind": node.kind, "properties": node.properties},
        ensure_ascii=False,
    )


def _get_neighbors(label: str, relation: Optional[str] = None) -> str:
    engine = _get_query_engine()
    neighbors = engine.get_neighbors(label, relation=relation)
    out = []
    for neighbor in neighbors:
        node = neighbor["node"]
        out.append(
            {
                "label": node.label,
                "kind": node.kind,
                "relation": neighbor["relation"],
                "direction": neighbor["direction"],
                "properties": node.properties,
            }
        )
    import json

    return json.dumps(out, ensure_ascii=False)


def _shortest_path(source: str, target: str, max_hops: int = 8) -> str:
    engine = _get_query_engine()
    path = engine.shortest_path(source, target, max_hops=max_hops)
    import json

    return json.dumps({"source": source, "target": target, "path": path, "max_hops": max_hops}, ensure_ascii=False)


def _get_community(community_id: int) -> str:
    engine = _get_query_engine()
    nodes = engine.get_community(community_id)
    import json

    return json.dumps(
        [
            {"label": node.label, "kind": node.kind, "properties": node.properties}
            for node in nodes
        ],
        ensure_ascii=False,
    )


def _god_nodes(top_n: int = 10) -> str:
    engine = _get_query_engine()
    ranked = engine.god_nodes(top_n=top_n)
    out = []
    for item in ranked:
        node = item["node"]
        out.append(
            {
                "label": item["label"],
                "degree": item["degree"],
                "kind": node.kind,
                "properties": node.properties,
            }
        )
    import json

    return json.dumps(out, ensure_ascii=False)


def _graph_stats() -> str:
    engine = _get_query_engine()
    stats = engine.graph_stats()
    import json

    return json.dumps(stats, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

_TOOLS = {
    "query_graph": (_query_graph, "Search the repo graph from a natural-language question."),
    "get_node": (_get_node, "Return a single graph node by label."),
    "get_neighbors": (_get_neighbors, "Return adjacent nodes for a label, optionally filtered by relation."),
    "shortest_path": (_shortest_path, "Return the shortest path between two node labels."),
    "get_community": (_get_community, "Return all nodes in a community by id."),
    "god_nodes": (_god_nodes, "Return the highest-degree nodes in the graph."),
    "graph_stats": (_graph_stats, "Return aggregate graph statistics."),
}


def create_mcp_server() -> Any:
    if not _MCP_SERVER_AVAILABLE or _MCP_SERVER_CLASS is None:
        raise RuntimeError(
            "mcp package is not installed. Install it to run the Graphify MCP server."
        )

    mcp = _MCP_SERVER_CLASS(
        "graphify",
        instructions="Graphify MCP server. Query the local repo knowledge graph.",
    )

    for name, (fn, description) in _TOOLS.items():
        try:
            mcp.add_tool(fn, name=name, description=description)
        except TypeError:
            try:
                decorated = mcp.tool(name=name, description=description)(fn)
                decorated  # registered as side-effect
            except Exception as exc:
                raise RuntimeError(f"failed to register graphify tool {name}: {exc}") from exc

    logger.info("graphify MCP server registered %d tools", len(_TOOLS))
    return mcp


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_graphify_mcp_server(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,
    )

    if not _MCP_SERVER_AVAILABLE:
        print(
            "Error: Graphify MCP server requires the 'mcp' package.",
            file=sys.stderr,
        )
        sys.exit(2)

    server = create_mcp_server()
    try:
        server.run()
    except KeyboardInterrupt:
        raise SystemExit(0)
