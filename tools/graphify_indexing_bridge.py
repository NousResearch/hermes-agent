"""Deterministic indexing bridge from ``graph.json`` to ``map/`` markdown artifacts.

This module rebuilds two derived artifacts from a persisted Graphify
``graph.json``:

* **map/FILE-MAP.md** — stable listing of source files mentioned in the
  graph, sorted deterministically so repeated runs produce identical
  output even when underlying ``graph.json`` node order changes.
* **map/effects/CONTEXT.md** — stable adjacency/community summary used by
  downstream effects/docs tooling.

Output is designed to be **idempotent** by construction: no timestamps,
no nondeterministic JSON key order in rendered text, and fallback paths
write a clearly labeled empty artifact instead of failing when the input
graph is missing or empty.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_graph_json(graph_path: Path) -> Dict[str, Any]:
    if not graph_path.exists() or not graph_path.is_file():
        raise FileNotFoundError(f"graph file not found: {graph_path}")
    with graph_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_source_file(source_file: Optional[str]) -> Optional[str]:
    if not source_file:
        return None
    value = str(source_file).strip()
    return value or None


def _sorted_unique(values: List[str]) -> List[str]:
    return sorted({v for v in values if v})


# ---------------------------------------------------------------------------
# Deterministic renderers
# ---------------------------------------------------------------------------


def render_file_map(
    nodes: List[Dict[str, Any]],
    *,
    max_source_files: Optional[int] = None,
) -> str:
    source_files = [
        _normalize_source_file(
            node.get("source_file") or node.get("properties", {}).get("source_file")
        )
        for node in nodes
    ]
    unique_sources = _sorted_unique([source for source in source_files if source])
    if max_source_files is not None:
        unique_sources = unique_sources[:max_source_files]

    lines = ["# FILE MAP", ""]
    lines.append("Generated from graph.json.")
    lines.append("")
    lines.append("| # | source file |")
    lines.append("|---:|:---|")
    for index, source in enumerate(unique_sources, start=1):
        lines.append(f"| {index} | {source} |")
    lines.append("")
    lines.append(f"_total source files: {len(unique_sources)}_")
    lines.append("")
    return "\n".join(lines)


def render_context_md(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    *,
    max_community_examples: int = 5,
) -> str:
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        relation = str(edge.get("relation") or "").strip()
        if not source or not target or source == target:
            continue
        adjacency[source].append(f"{relation} {target}")

    community_index: Dict[str, List[str]] = defaultdict(list)
    for node in nodes:
        community = str(node.get("community") or "__unclustered__").strip() or "__unclustered__"
        community_index[community].append(str(node.get("id") or node.get("label") or "").strip())

    community_examples: Dict[str, List[str]] = {}
    for community, member_ids in community_index.items():
        community_examples[community] = _sorted_unique(member_ids)[:max_community_examples]

    lines = ["# EFFECTS CONTEXT", ""]
    lines.append("Generated from graph.json.")
    lines.append("")
    lines.append(f"_node_count: {len(nodes)}_")
    lines.append(f"_edge_count: {len(edges)}_")
    lines.append(f"_community_count: {len(community_index)}_")
    lines.append("")
    lines.append("## Communities")
    lines.append("")
    for community in sorted(community_index):
        member_ids = community_examples[community]
        lines.append(f"- {community}: {', '.join(member_ids)}")
    lines.append("")
    lines.append("## Adjacency")
    lines.append("")
    for node_id in sorted(adjacency):
        lines.append(f"- {node_id}: {', '.join(adjacency[node_id])}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class GraphifyIndexingBridgeResult:
    def __init__(self) -> None:
        self.file_map_path: Optional[Path] = None
        self.context_path: Optional[Path] = None
        self.file_map_text: str = ""
        self.context_text: str = ""


def rebuild_indexing_bridge(
    graph_path: Path,
    output_root: Path,
    *,
    graph_payload: Optional[Dict[str, Any]] = None,
) -> GraphifyIndexingBridgeResult:
    """Rebuild deterministic markdown artifacts from a Graphify graph.json.

    Parameters
    ----------
    graph_path:
        Path to ``graph.json``. Used for existence validation and for
        deriving the relative doc path when ``graph_payload`` is omitted.
    output_root:
        Directory under which ``map/FILE-MAP.md`` and
        ``map/effects/CONTEXT.md`` are written.
    graph_payload:
        Optional pre-parsed graph payload. When provided, file existence
        on ``graph_path`` is not enforced, which supports missing-input
        fallback behavior for tests and offline runs.

    Returns
    -------
    GraphifyIndexingBridgeResult
        Paths and rendered text for both artifacts.
    """
    result = GraphifyIndexingBridgeResult()

    missing_input = graph_payload is None and not graph_path.exists()
    if missing_input:
        file_map_text = "# FILE MAP\n\n_missing input graph; no source files available._\n\n"
        context_text = "# EFFECTS CONTEXT\n\n_missing input graph; no graph context available._\n\n"
    else:
        if graph_payload is None:
            graph_payload = _load_graph_json(graph_path)

        nodes = list(graph_payload.get("nodes") or [])
        edges = list(graph_payload.get("edges") or [])

        file_map_text = render_file_map(nodes)
        context_text = render_context_md(nodes, edges)

    file_map_path = output_root / "FILE-MAP.md"
    context_path = output_root / "effects" / "CONTEXT.md"

    file_map_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.parent.mkdir(parents=True, exist_ok=True)

    file_map_path.write_text(file_map_text, encoding="utf-8")
    context_path.write_text(context_text, encoding="utf-8")

    result.file_map_path = file_map_path
    result.context_path = context_path
    result.file_map_text = file_map_text
    result.context_text = context_text
    return result
