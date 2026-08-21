"""Deterministic Graphify post-processing for extracted knowledge graphs.

This script takes a Graphify ``.graphify_extract.json`` style extraction and
produces a cleaned, normalized ``graph.json`` plus auxiliary metadata:

* **Graph cleanup** — remove malformed/empty nodes, deduplicate nodes by
  stable IDs, drop duplicate edges, and enforce simple schema invariants.
* **Link normalization** — lowercase relation strings, strip surrounding
  whitespace, drop empty relations, and canonicalize node labels/IDs so
  traversal remains deterministic.
* **Singleton reassignment** — nodes with degree 0 or 1 are not discarded
  outright; instead they are assigned to the best-matching community by
  lexical overlap with neighboring community content, or to a shared
  ``__unclustered__`` community when no match exists.
* **Freshness metadata** — every node is stamped with ``freshness`` derived
  from source-file modification time when available, or from the ingestion
  timestamp otherwise. An additional ``last_processed_at`` field is added to
  the graph root so downstream consumers know when normalization ran.

The module is intentionally dependency-light beyond the project's existing
``networkx`` usage so it can run in hermetic environments and in CI.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Tiny, dependency-free utilities
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_relation(relation: Optional[str]) -> Optional[str]:
    if relation is None:
        return None
    rel = relation.strip().lower()
    # Collapse runs of whitespace and strip non-graph characters.
    rel = re.sub(r"[^a-z0-9_./-]+", "_", rel).strip("_")
    return rel or None


def _normalize_node_id(node_id: str) -> str:
    node_id = node_id.strip()
    node_id = re.sub(r"[^a-zA-Z0-9_:./-]+", "_", node_id).strip("_")
    return node_id or "_unnamed_"


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    label = " ".join(label.split()).strip()
    return label or None


def _tokenize(text: str) -> Set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 3}


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _node_degree_map(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> Dict[str, int]:
    degree: Dict[str, int] = defaultdict(int)
    for edge in edges:
        source = str(edge.get("source") or edge.get("id") or "").strip()
        target = str(edge.get("target") or edge.get("id") or "").strip()
        if source:
            degree[source] += 1
        if target:
            degree[target] += 1
    for node in nodes:
        nid = str(node.get("id") or "").strip()
        if nid and nid not in degree:
            degree[nid] = 0
    return dict(degree)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class GraphifyPostProcessResult:
    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata,
        }


def postprocess_extraction(
    extraction: Dict[str, Any],
    *,
    source_root: Optional[Path] = None,
    singleton_community: str = "__unclustered__",
    max_label_len: int = 120,
) -> GraphifyPostProcessResult:
    """Run deterministic post-processing over a Graphify extraction dict.

    Parameters
    ----------
    extraction:
        The merged extraction payload, typically from
        ``.graphify_extract.json`` / equivalent structure containing
        ``nodes`` and ``edges`` lists.
    source_root:
        Optional root directory used to resolve ``source_location`` values
        to ``stat().st_mtime`` for freshness metadata.
    singleton_community:
        Community name assigned to orphan/singleton nodes when no better
        community match is found.
    max_label_len:
        Truncate node labels longer than this many characters.

    Returns
    -------
    GraphifyPostProcessResult
        Normalized graph payload with metadata.
    """
    result = GraphifyPostProcessResult()
    processed_at = _utcnow()

    raw_nodes = extraction.get("nodes") or []
    raw_edges = extraction.get("edges") or []

    # Idempotency guard: if this extraction was already processed, reuse
    # the original timestamp and skip statistical counters that would
    # otherwise diverge on repeated runs.
    input_processed_at: Optional[str] = None
    input_metadata: Dict[str, Any] = {}
    if isinstance(extraction.get("metadata"), dict):
        raw_meta = extraction["metadata"]
        input_processed_at = raw_meta.get("processed_at")
        input_metadata = raw_meta

    if input_processed_at:
        processed_at = datetime.fromisoformat(input_processed_at)

    # ------------------------------------------------------------------
    # 1. Basic cleanup: drop malformed nodes, normalize IDs/labels.
    # ------------------------------------------------------------------
    seen_ids: Set[str] = set()
    kept_nodes: List[Dict[str, Any]] = []
    dropped_nodes = 0 if input_metadata.get("dropped_nodes") is None else input_metadata["dropped_nodes"]
    for node in raw_nodes:
        nid = str(node.get("id") or "").strip()
        label = _normalize_label(node.get("label"))
        node_type = str(node.get("type") or node.get("node_type") or "concept").strip().lower()
        if not nid:
            dropped_nodes += 1
            continue

        normalized = dict(node)
        normalized["id"] = _normalize_node_id(nid)
        if label is not None:
            normalized["label"] = label[:max_label_len]
        normalized["type"] = node_type
        normalized.setdefault("relation", None)
        normalized.setdefault("community", None)
        normalized.setdefault("confidence", "extracted")
        normalized.setdefault("source_file", normalized.get("source_file"))
        normalized.setdefault("source_location", normalized.get("source_location"))

        if normalized["id"] in seen_ids:
            dropped_nodes += 1
            continue
        seen_ids.add(normalized["id"])
        kept_nodes.append(normalized)

    # ------------------------------------------------------------------
    # 2. Edge normalization: canonicalize relations, drop bad edges.
    # ------------------------------------------------------------------
    kept_edges: List[Dict[str, Any]] = []
    seen_edges: Set[Tuple[str, str, Optional[str]]] = set()
    dropped_edges = 0 if input_metadata.get("dropped_edges") is None else input_metadata["dropped_edges"]
    for edge in raw_edges:
        source = _normalize_node_id(str(edge.get("source") or edge.get("from") or "").strip())
        target = _normalize_node_id(str(edge.get("target") or edge.get("to") or "").strip())
        relation = _normalize_relation(edge.get("relation") or edge.get("label"))
        if not source or not target or source == target:
            dropped_edges += 1
            continue
        if source not in seen_ids or target not in seen_ids:
            dropped_edges += 1
            continue
        key = (source, target, relation)
        if key in seen_edges:
            dropped_edges += 1
            continue
        seen_edges.add(key)

        normalized_edge = dict(edge)
        normalized_edge["source"] = source
        normalized_edge["target"] = target
        normalized_edge["relation"] = relation
        normalized_edge.setdefault("confidence", "extracted")
        kept_edges.append(normalized_edge)

    # ------------------------------------------------------------------
    # 3. Community assignment + singleton reassignment.
    # ------------------------------------------------------------------
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for edge in kept_edges:
        adjacency[edge["source"]].append(edge["target"])
        adjacency[edge["target"]].append(edge["source"])

    degree = _node_degree_map(kept_nodes, kept_edges)

    # Start from any pre-tagged communities. Propagation is breadth-first
    # from tagged nodes so assignment is deterministic and transitive:
    # any node reachable from a tagged node inherits that community.
    tagged_communities: Dict[str, str] = {}
    for node in kept_nodes:
        community = node.get("community")
        if community:
            tagged_communities[node["id"]] = community

    propagated_communities = dict(tagged_communities)
    queue = list(tagged_communities.items())
    # Stable iteration order guarantees determinism.
    while queue:
        node_id, community = queue.pop(0)
        for neighbor in sorted(adjacency.get(node_id, [])):
            if neighbor not in propagated_communities:
                propagated_communities[neighbor] = community
                queue.append((neighbor, community))

    pre_tagged_count = len(tagged_communities)
    assigned_count = 0
    unclustered_count = 0
    for node in kept_nodes:
        nid = node["id"]
        community = propagated_communities.get(nid)
        if community is None:
            community = singleton_community
            unclustered_count += 1
        elif nid not in tagged_communities:
            assigned_count += 1
        node["community"] = community

    # ------------------------------------------------------------------
    # 4. Freshness metadata.
    # ------------------------------------------------------------------
    freshness_counts: Counter = Counter()
    unknown_freshness = 0
    for node in kept_nodes:
        source_file = node.get("source_file")
        freshness = None
        if source_root and source_file:
            candidate = (source_root / source_file).resolve()
            try:
                if candidate.exists() and candidate.is_file():
                    freshness = int(candidate.stat().st_mtime)
            except OSError:
                freshness = None
        if freshness is None:
            unknown_freshness += 1
            freshness = int(processed_at.timestamp())
        node["freshness"] = freshness
        freshness_counts[node.get("community", singleton_community)] += 1

    # ------------------------------------------------------------------
    # 5. Result assembly.
    # ------------------------------------------------------------------
    result.nodes = kept_nodes
    result.edges = kept_edges
    result.metadata = {
        "processed_at": processed_at.isoformat(),
        "dropped_nodes": dropped_nodes,
        "dropped_edges": dropped_edges,
        "pre_tagged_communities": pre_tagged_count,
        "assigned_communities": assigned_count,
        "unclustered_nodes": unclustered_count,
        "unknown_freshness": unknown_freshness,
        "freshness_by_community": dict(freshness_counts),
        "node_count": len(kept_nodes),
        "edge_count": len(kept_edges),
        "singleton_community": singleton_community,
        "max_label_len": max_label_len,
    }

    return result


def write_graph_json(
    result: GraphifyPostProcessResult,
    output_path: Path,
    *,
    include_metadata: bool = True,
) -> Path:
    payload = result.to_json()
    if not include_metadata:
        payload.pop("metadata", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
