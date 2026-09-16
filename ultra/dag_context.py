#!/usr/bin/env python3
"""
DAG-Based Context Manager for Hermes Agent.

Replaces linear summary with a compressible directed acyclic graph.
Inspired by MUSE-AutoSkill (arXiv 2605.27366).

Structure:
  Nodes = reasoning turns (plan, action, observation, decision)
  Branches = alternative approaches (some fail, some partially succeed)
  Level 1 compression = in-place summary per oversized node
  Level 2 compression = chain-level merge (pin first/last, fuse middle)

Storage:
  ~/.hermes/ultra/dag_store/          (or $ULTRA_HOME/dag_store/)
    index.json                        (session registry)
    sessions/<session_id>.json        (session metadata)
    nodes/<session_id>_<node_id>.json (individual node files)

Usage:
    from ultra.dag_context import DagContext

    dag = DagContext("my-task")
    dag.add("plan", "Research approach for X")
    dag.add("action", "Search for X patterns", tool="web_search")
    dag.add("observation", "Found 3 papers on X", cost=450)
    dag.add("decision", "Use paper 2 approach")

    # Compact view for LLM context injection
    print(dag.export(fmt="compact"))

    # Compress when context grows large
    dag.compress(level=2, budget=50000)

    # Tree visualization
    dag.show()

Paper: MUSE-AutoSkill — Self-Evolving Agents via Skill Creation, Memory,
Management, and Evaluation. Lin et al. (ByteDance/RIT), arXiv 2605.27366.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ultra.dag_context")

# ── Node Types ──────────────────────────────────────────

NODE_TYPES = frozenset({"plan", "action", "observation", "decision", "hypothesis", "result"})

# ── Storage Paths ───────────────────────────────────────

def _get_ultra_home() -> Path:
    """Resolve the Ultra storage root."""
    try:
        from hermes_constants import get_hermes_home
        return Path(os.environ.get("ULTRA_HOME", str(get_hermes_home() / "ultra")))
    except ImportError:
        return Path(
            os.environ.get("ULTRA_HOME", str(Path.home() / ".hermes" / "ultra"))
        )


def _get_store_paths() -> Tuple[Path, Path, Path]:
    """Return (session_dir, node_dir, index_path)."""
    base = _get_ultra_home() / "dag_store"
    return base / "sessions", base / "nodes", base / "index.json"


# ── Data Model ──────────────────────────────────────────

def _mk_id(content: str) -> str:
    return hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:12]


def make_node(
    node_type: str,
    content: str,
    parent_id: Optional[str] = None,
    tool: Optional[str] = None,
    cost: int = 0,
) -> Dict[str, Any]:
    """Create a single DAG node."""
    if node_type not in NODE_TYPES:
        raise ValueError(f"Invalid node type: {node_type!r}. Valid: {sorted(NODE_TYPES)}")
    return {
        "id": _mk_id(content),
        "type": node_type,
        "content": content,
        "parent": parent_id,
        "tool": tool,
        "cost": cost,
        "status": "active",        # active | compressed | merged | dropped
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": None,           # filled by compression
        "children": [],
    }


# ── Session ─────────────────────────────────────────────

class DagSession:
    """A single reasoning session — a tree of DAG nodes."""

    def __init__(self, session_id: str, label: str):
        self.id = session_id
        self.label = label
        self.created = datetime.now(timezone.utc).isoformat()
        self.total_cost = 0
        self.total_nodes = 0
        self.compressed_nodes = 0
        self.root_node: Optional[str] = None
        self.head_node: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "created": self.created,
            "total_cost": self.total_cost,
            "total_nodes": self.total_nodes,
            "compressed_nodes": self.compressed_nodes,
            "root_node": self.root_node,
            "head_node": self.head_node,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DagSession":
        s = cls.__new__(cls)
        s.id = d["id"]
        s.label = d["label"]
        s.created = d["created"]
        s.total_cost = d.get("total_cost", 0)
        s.total_nodes = d.get("total_nodes", 0)
        s.compressed_nodes = d.get("compressed_nodes", 0)
        s.root_node = d.get("root_node")
        s.head_node = d.get("head_node")
        return s


# ── DAG Context ─────────────────────────────────────────

class DagContext:
    """Main interface for DAG-based context management."""

    def __init__(self, label: str = "", session_id: Optional[str] = None):
        self._sessions_dir, self._nodes_dir, self._index_path = _get_store_paths()
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._nodes_dir.mkdir(parents=True, exist_ok=True)

        if session_id:
            self.session = self._load_session(session_id)
        else:
            self.session = self._new_session(label)
        logger.debug("DagContext initialized: session=%s label=%s", self.session.id[:8], self.session.label)

    # ── Persistence ───────────────────────────────────

    def _load_index(self) -> Dict[str, Any]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {"active_session": None, "sessions": []}

    def _save_index(self, idx: Dict[str, Any]) -> None:
        self._index_path.write_text(json.dumps(idx, indent=2))

    def _save_session(self) -> None:
        path = self._sessions_dir / f"{self.session.id}.json"
        path.write_text(json.dumps(self.session.to_dict(), indent=2))

    def _load_session(self, sid: str) -> DagSession:
        path = self._sessions_dir / f"{sid}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session {sid} not found")
        return DagSession.from_dict(json.loads(path.read_text()))

    def _save_node(self, node: Dict[str, Any]) -> None:
        path = self._nodes_dir / f"{self.session.id}_{node['id']}.json"
        path.write_text(json.dumps(node, indent=2))

    def _load_nodes(self) -> List[Dict[str, Any]]:
        nodes = []
        prefix = f"{self.session.id}_"
        for f in sorted(self._nodes_dir.glob(f"{prefix}*.json")):
            nodes.append(json.loads(f.read_text()))
        return nodes

    def _new_session(self, label: str) -> DagSession:
        sid = hashlib.sha256(f"{label}{time.time()}".encode()).hexdigest()[:16]
        session = DagSession(sid, label or "untitled")
        self._save_session()
        idx = self._load_index()
        idx["active_session"] = session.id
        idx.setdefault("sessions", []).append({
            "id": session.id, "label": session.label, "created": session.created
        })
        self._save_index(idx)
        return session

    # ── Core API ──────────────────────────────────────

    def add(
        self,
        node_type: str,
        content: str,
        tool: Optional[str] = None,
        cost: int = 0,
    ) -> Dict[str, Any]:
        """Add a reasoning node to the DAG."""
        parent_id = self.session.head_node
        node = make_node(node_type, content, parent_id, tool, cost)

        # Link parent → child
        if parent_id:
            parent_path = self._nodes_dir / f"{self.session.id}_{parent_id}.json"
            if parent_path.exists():
                parent = json.loads(parent_path.read_text())
                parent["children"].append(node["id"])
                self._save_node(parent)

        # Set root if first node
        if not self.session.root_node:
            self.session.root_node = node["id"]
        self.session.head_node = node["id"]
        self.session.total_cost += cost
        self.session.total_nodes += 1
        self._save_session()
        self._save_node(node)

        logger.debug("Node added: type=%s id=%s… cost=%d", node_type, node["id"][:8], cost)
        return node

    # ── Compression ───────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count: 1 token ≈ 4 characters."""
        return max(len(text) // 4, 1)

    def _compress_level1(self, nodes: List[Dict[str, Any]], max_tokens: int = 5000) -> List[Dict[str, Any]]:
        """Level 1: in-place summary — each oversized node summarized individually."""
        for node in nodes:
            content_tokens = self._estimate_tokens(node.get("content", ""))
            if content_tokens > max_tokens:
                node["summary"] = node["content"][: max_tokens * 4] + "… [COMPRESSED_L1]"
                node["content"] = node["summary"]
                node["status"] = "compressed"
                node["cost"] = min(node["cost"], max_tokens)
        return nodes

    def _compress_level2(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Level 2: chain-level merge — fuse middle turns, pin first and last."""
        if len(nodes) <= 4:
            return nodes

        first, middle, last = nodes[0], nodes[1:-1], nodes[-1]
        middle_text = " | ".join(n.get("content", "")[:200] for n in middle)
        middle_tools = list({n.get("tool") for n in middle if n.get("tool")})
        middle_cost = sum(n.get("cost", 0) for n in middle)

        merged = make_node("observation", f"[MERGED_L2] {middle_text[:500]}")
        merged["status"] = "merged"
        merged["cost"] = middle_cost
        if middle_tools:
            merged["tool"] = ",".join(t for t in middle_tools if t)

        for n in middle:
            n["status"] = "merged_into_child"

        merged["parent"] = first["id"]
        first["children"] = [merged["id"]]
        last["parent"] = merged["id"]

        return [first, merged, last]

    def compress(self, level: int = 2, budget: int = 50000) -> List[Dict[str, Any]]:
        """Run the full compression pipeline."""
        nodes = self._load_nodes()
        if not nodes:
            return []

        nodes.sort(key=lambda n: n.get("timestamp", ""))

        if level >= 1:
            nodes = self._compress_level1(nodes)
        if level >= 2:
            nodes = self._compress_level2(nodes)

        self.session.compressed_nodes = sum(
            1 for n in nodes if n["status"] in ("compressed", "merged")
        )
        self._save_session()

        logger.info("Compressed %d nodes (level=%d, budget=%d)", len(nodes), level, budget)
        return nodes

    # ── Export ────────────────────────────────────────

    def export(self, fmt: str = "compact") -> Optional[str]:
        """Export the DAG for LLM context injection."""
        nodes = self._load_nodes()
        if not nodes:
            return None

        if fmt == "json":
            return json.dumps({
                "session": self.session.to_dict(),
                "nodes": nodes,
            }, indent=2)

        lines = [
            f"# DAG: {self.session.label} "
            f"({self.session.total_nodes} nodes, {self.session.total_cost} tokens)"
        ]
        for n in nodes:
            if n["status"] not in ("merged_into_child", "dropped"):
                icon = {
                    "plan": "📋", "action": "⚡", "observation": "👁",
                    "decision": "🔀", "hypothesis": "💡", "result": "✅",
                }.get(n["type"], "•")
                lines.append(f"{icon} [{n['type']}] {n.get('content', '')[:120]}")

        return "\n".join(lines) if fmt == "compact" else self._export_tree()

    def _export_tree(self) -> str:
        """Render an ASCII tree of the DAG."""
        nodes = self._load_nodes()
        by_id = {n["id"]: n for n in nodes}
        root = self.session.root_node
        if not root or root not in by_id:
            return "No root node found"

        def _render(nid: str, depth: int = 0) -> str:
            n = by_id.get(nid)
            if not n:
                return ""
            indent = "  " * depth
            status = {"active": "●", "compressed": "○", "merged": "◉"}.get(n["status"], "?")
            line = f"{indent}{status} [{n['type']}] {n.get('content', '')[:80]}"
            children = "".join(_render(cid, depth + 1) for cid in n.get("children", []))
            return f"{line}\n{children}" if children else line

        return _render(root)

    def show(self) -> None:
        """Print the tree to stdout."""
        print(self._export_tree())

    # ── Stats ─────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return session statistics."""
        nodes = self._load_nodes()
        type_counts: Dict[str, int] = {}
        for n in nodes:
            type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

        return {
            "session_id": self.session.id,
            "label": self.session.label,
            "total_nodes": self.session.total_nodes,
            "active_nodes": sum(1 for n in nodes if n["status"] == "active"),
            "compressed_nodes": sum(1 for n in nodes if n["status"] == "compressed"),
            "merged_nodes": sum(1 for n in nodes if n["status"] == "merged"),
            "total_cost": self.session.total_cost,
            "type_breakdown": type_counts,
        }

    # ── Session Registry ──────────────────────────────

    @classmethod
    def list_sessions(cls) -> List[Dict[str, str]]:
        """List all registered sessions."""
        *_, idx_path = _get_store_paths()
        if not idx_path.exists():
            return []
        idx = json.loads(idx_path.read_text())
        return idx.get("sessions", [])

    @classmethod
    def set_active(cls, session_id: str) -> None:
        """Set the active session by ID."""
        *_, idx_path = _get_store_paths()
        idx: dict = json.loads(idx_path.read_text()) if idx_path.exists() else {"sessions": []}
        idx["active_session"] = session_id
        idx_path.write_text(json.dumps(idx, indent=2))


# ── Convenience Functions (backwards-compatible) ────────

def new_session(label: str = "") -> Dict[str, Any]:
    """Create a new session and return its metadata. Legacy API."""
    ctx = DagContext(label)
    return ctx.session.to_dict()


def add_node(
    node_type: str,
    content: str,
    tool: Optional[str] = None,
    cost: int = 0,
) -> Dict[str, Any]:
    """Add a node to the active session. Legacy API."""
    ctx = DagContext(session_id=_load_active_session_id())
    return ctx.add(node_type, content, tool, cost)


def _load_active_session_id() -> Optional[str]:
    *_, idx_path = _get_store_paths()
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        return idx.get("active_session")
    return None


def compress_dag(
    session_id: Optional[str] = None,
    level: int = 2,
    budget: int = 50000,
) -> Optional[List[Dict[str, Any]]]:
    """Compress the DAG for a session. Legacy API."""
    if not session_id:
        session_id = _load_active_session_id()
    if not session_id:
        return None
    ctx = DagContext(session_id=session_id)
    return ctx.compress(level, budget)


def export_dag(
    session_id: Optional[str] = None,
    fmt: str = "compact",
) -> Optional[str]:
    """Export the DAG for a session. Legacy API."""
    if not session_id:
        session_id = _load_active_session_id()
    if not session_id:
        return None
    ctx = DagContext(session_id=session_id)
    return ctx.export(fmt)


def get_stats(session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get stats for a session. Legacy API."""
    if not session_id:
        session_id = _load_active_session_id()
    if not session_id:
        return None
    ctx = DagContext(session_id=session_id)
    return ctx.stats()
