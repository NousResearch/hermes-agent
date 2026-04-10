#!/usr/bin/env python3
"""Project-scoped persistent memory, decisions, graph relations, and drift baselines.

This operationalizes the recommended mex/iwe/context-graph/audit-log pattern
inside Hermes Agent without replacing the existing global MEMORY.md / USER.md
stores. Data is scoped to a project root (default: TERMINAL_CWD or cwd) and
persists across sessions under ~/.hermes/project_memory/<namespace>.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from tools.memory_tool import _scan_memory_content
from tools.registry import registry

PROJECT_MEMORY_DIR = get_hermes_home() / "project_memory"
_DEFAULT_CHAR_LIMIT = 4000
_DEFAULT_GRAPH_EDGE_LIMIT = 200


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "project"


def _derive_project_root(project_root: Optional[str] = None) -> Path:
    root = project_root or os.getenv("TERMINAL_CWD") or os.getcwd()
    return Path(root).expanduser().resolve()


def _namespace_for_root(root: Path) -> str:
    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(root.name)}-{digest}"


def _signature_for_file(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return {
        "sha256": sha.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class ProjectMemoryStore:
    """File-backed project memory store with notes, decisions, graph, and drift baselines."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        char_limit: int = _DEFAULT_CHAR_LIMIT,
        graph_edge_limit: int = _DEFAULT_GRAPH_EDGE_LIMIT,
    ):
        self.project_root = _derive_project_root(project_root)
        self.namespace = _namespace_for_root(self.project_root)
        self.char_limit = int(char_limit or _DEFAULT_CHAR_LIMIT)
        self.graph_edge_limit = int(graph_edge_limit or _DEFAULT_GRAPH_EDGE_LIMIT)
        self.filepath = PROJECT_MEMORY_DIR / f"{self.namespace}.json"
        self.data: Dict[str, Any] = {}
        self._system_prompt_snapshot = ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_from_disk(self) -> None:
        PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if self.filepath.exists():
            try:
                self.data = json.loads(self.filepath.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        if not self.data:
            self.data = {
                "version": 1,
                "namespace": self.namespace,
                "project_root": str(self.project_root),
                "notes": [],
                "decisions": [],
                "graph": {"edges": []},
                "tracked_files": {},
                "metadata": {"created_at": _utc_now(), "updated_at": _utc_now()},
            }
        self.data.setdefault("notes", [])
        self.data.setdefault("decisions", [])
        self.data.setdefault("graph", {}).setdefault("edges", [])
        self.data.setdefault("tracked_files", {})
        self.data.setdefault("metadata", {})
        self.data["namespace"] = self.namespace
        self.data["project_root"] = str(self.project_root)
        self._system_prompt_snapshot = self._render_snapshot()

    def save_to_disk(self) -> None:
        PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.data.setdefault("metadata", {})["updated_at"] = _utc_now()
        fd, tmp_path = tempfile.mkstemp(dir=str(PROJECT_MEMORY_DIR), prefix=".project_mem_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.filepath)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def format_for_system_prompt(self) -> Optional[str]:
        return self._system_prompt_snapshot or None

    # ------------------------------------------------------------------
    # Notes / decisions / graph
    # ------------------------------------------------------------------

    def add_note(self, content: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        content = (content or "").strip()
        if not content:
            return {"success": False, "error": "content is required"}
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}
        note_id = f"note-{len(self.data['notes']) + 1:04d}"
        self.data["notes"].append({
            "id": note_id,
            "content": content,
            "tags": [str(t).strip() for t in (tags or []) if str(t).strip()],
            "created_at": _utc_now(),
        })
        self._enforce_limits()
        self.save_to_disk()
        return {
            "success": True,
            "message": "Project note added.",
            "note_id": note_id,
            "notes": self.list_notes(limit=10)["notes"],
            "namespace": self.namespace,
            "project_root": str(self.project_root),
        }

    def list_notes(self, limit: int = 10) -> Dict[str, Any]:
        notes = self.data.get("notes", [])[-max(1, int(limit or 10)):]
        return {
            "success": True,
            "notes": notes,
            "count": len(self.data.get("notes", [])),
            "namespace": self.namespace,
            "project_root": str(self.project_root),
        }

    def record_decision(
        self,
        summary: str,
        rationale: Optional[str] = None,
        related_files: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: str = "open",
    ) -> Dict[str, Any]:
        summary = (summary or "").strip()
        if not summary:
            return {"success": False, "error": "summary is required"}
        scan_error = _scan_memory_content(summary)
        if scan_error:
            return {"success": False, "error": scan_error}
        if rationale:
            rationale = rationale.strip()
            scan_error = _scan_memory_content(rationale)
            if scan_error:
                return {"success": False, "error": scan_error}
        decision_id = f"dec-{len(self.data['decisions']) + 1:04d}"
        cleaned_files = [self._normalize_project_path(p) for p in (related_files or []) if str(p).strip()]
        self.data["decisions"].append({
            "id": decision_id,
            "summary": summary,
            "rationale": rationale or "",
            "related_files": cleaned_files,
            "tags": [str(t).strip() for t in (tags or []) if str(t).strip()],
            "status": status or "open",
            "outcome": "",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        })
        self.save_to_disk()
        return {
            "success": True,
            "message": "Decision recorded.",
            "decision_id": decision_id,
            "decisions": self.list_decisions(limit=10)["decisions"],
            "namespace": self.namespace,
        }

    def update_decision(self, decision_id: str, outcome: str, status: Optional[str] = None) -> Dict[str, Any]:
        decision_id = (decision_id or "").strip()
        outcome = (outcome or "").strip()
        if not decision_id or not outcome:
            return {"success": False, "error": "decision_id and outcome are required"}
        scan_error = _scan_memory_content(outcome)
        if scan_error:
            return {"success": False, "error": scan_error}
        for decision in self.data.get("decisions", []):
            if decision.get("id") == decision_id:
                decision["outcome"] = outcome
                if status:
                    decision["status"] = status
                decision["updated_at"] = _utc_now()
                self.save_to_disk()
                return {"success": True, "message": "Decision updated.", "decision": decision}
        return {"success": False, "error": f"No decision matched '{decision_id}'."}

    def list_decisions(self, limit: int = 10) -> Dict[str, Any]:
        decisions = self.data.get("decisions", [])[-max(1, int(limit or 10)):]
        return {
            "success": True,
            "decisions": decisions,
            "count": len(self.data.get("decisions", [])),
            "namespace": self.namespace,
        }

    def relate(self, subject: str, predicate: str, object_: str, weight: float = 1.0) -> Dict[str, Any]:
        subject = (subject or "").strip()
        predicate = (predicate or "").strip()
        object_ = (object_ or "").strip()
        if not subject or not predicate or not object_:
            return {"success": False, "error": "subject, predicate, and object are required"}
        for field in (subject, predicate, object_):
            scan_error = _scan_memory_content(field)
            if scan_error:
                return {"success": False, "error": scan_error}
        edge = {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "weight": float(weight or 1.0),
            "created_at": _utc_now(),
        }
        edges = self.data.setdefault("graph", {}).setdefault("edges", [])
        if edge not in edges:
            edges.append(edge)
        if len(edges) > self.graph_edge_limit:
            self.data["graph"]["edges"] = edges[-self.graph_edge_limit :]
        self.save_to_disk()
        return {
            "success": True,
            "message": "Relation added.",
            "edge": edge,
            "edge_count": len(self.data["graph"]["edges"]),
            "namespace": self.namespace,
        }

    def view_graph(self, limit: int = 25) -> Dict[str, Any]:
        edges = self.data.get("graph", {}).get("edges", [])[-max(1, int(limit or 25)):]
        nodes = sorted({edge["subject"] for edge in edges} | {edge["object"] for edge in edges})
        return {
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "edge_count": len(self.data.get("graph", {}).get("edges", [])),
            "namespace": self.namespace,
        }

    # ------------------------------------------------------------------
    # Drift tracking
    # ------------------------------------------------------------------

    def track_paths(self, paths: Iterable[str]) -> Dict[str, Any]:
        expanded = self._expand_paths(paths)
        if not expanded:
            return {"success": False, "error": "No files matched the provided paths."}
        tracked = self.data.setdefault("tracked_files", {})
        updated = []
        for path in expanded:
            rel = self._normalize_project_path(str(path))
            tracked[rel] = {
                **_signature_for_file(path),
                "updated_at": _utc_now(),
            }
            updated.append(rel)
        self.save_to_disk()
        return {
            "success": True,
            "message": f"Tracked {len(updated)} file(s).",
            "tracked": updated,
            "tracked_count": len(tracked),
            "namespace": self.namespace,
        }

    def drift_check(self, paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        tracked = self.data.setdefault("tracked_files", {})
        if paths:
            candidate_paths = self._expand_paths(paths)
            keys = [self._normalize_project_path(str(path)) for path in candidate_paths]
        else:
            keys = list(tracked.keys())
        changes = []
        for rel in keys:
            absolute = (self.project_root / rel).resolve()
            baseline = tracked.get(rel)
            if baseline is None:
                if absolute.exists() and absolute.is_file():
                    changes.append({"path": rel, "status": "untracked"})
                continue
            if not absolute.exists():
                changes.append({"path": rel, "status": "deleted"})
                continue
            current = _signature_for_file(absolute)
            if current["sha256"] != baseline.get("sha256"):
                changes.append({
                    "path": rel,
                    "status": "modified",
                    "before": baseline.get("sha256"),
                    "after": current["sha256"],
                })
        return {
            "success": True,
            "changes": changes,
            "tracked_count": len(tracked),
            "namespace": self.namespace,
            "project_root": str(self.project_root),
        }

    def overview(self, limit: int = 5) -> Dict[str, Any]:
        return {
            "success": True,
            "namespace": self.namespace,
            "project_root": str(self.project_root),
            "notes": self.data.get("notes", [])[-max(1, int(limit or 5)):],
            "decisions": self.data.get("decisions", [])[-max(1, int(limit or 5)):],
            "graph_edge_count": len(self.data.get("graph", {}).get("edges", [])),
            "tracked_file_count": len(self.data.get("tracked_files", {})),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_project_path(self, path: str) -> str:
        try:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = (self.project_root / candidate).resolve()
            return str(candidate.relative_to(self.project_root))
        except Exception:
            return path

    def _expand_paths(self, paths: Iterable[str]) -> List[Path]:
        expanded: List[Path] = []
        seen = set()
        for raw in paths or []:
            pattern = str(raw or "").strip()
            if not pattern:
                continue
            has_glob = any(ch in pattern for ch in "*?[")
            if has_glob:
                matches = sorted(self.project_root.glob(pattern))
            else:
                matches = [Path(pattern)]
            for match in matches:
                path = match.expanduser()
                if not path.is_absolute():
                    path = (self.project_root / path).resolve()
                if path.exists() and path.is_file() and path not in seen:
                    expanded.append(path)
                    seen.add(path)
        return expanded

    def _render_snapshot(self) -> str:
        notes = self.data.get("notes", [])[-3:]
        decisions = self.data.get("decisions", [])[-3:]
        edges = self.data.get("graph", {}).get("edges", [])[-5:]
        tracked_count = len(self.data.get("tracked_files", {}))
        parts = [
            "══════════════════════════════════════════════",
            f"PROJECT MEMORY ({self.project_root.name})",
            "══════════════════════════════════════════════",
            f"Root: {self.project_root}",
        ]
        if notes:
            parts.append("Recent project notes:")
            for note in notes:
                parts.append(f"- {note['content']}")
        if decisions:
            parts.append("Recent decisions:")
            for decision in decisions:
                status = decision.get("status", "open")
                outcome = f" | outcome: {decision['outcome']}" if decision.get("outcome") else ""
                parts.append(f"- [{decision['id']}] ({status}) {decision['summary']}{outcome}")
        if edges:
            parts.append("Context graph:")
            for edge in edges:
                parts.append(f"- {edge['subject']} --{edge['predicate']}--> {edge['object']}")
        parts.append(f"Tracked files: {tracked_count}")
        block = "\n".join(parts)
        if len(block) <= self.char_limit:
            return block
        return block[: self.char_limit - 24] + "\n...[truncated project memory]"

    def _enforce_limits(self) -> None:
        while len(self._render_snapshot()) > self.char_limit and self.data.get("notes"):
            self.data["notes"].pop(0)


PROJECT_MEMORY_SCHEMA = {
    "name": "project_memory",
    "description": (
        "Manage project-scoped persistent memory for the current working directory. "
        "Use this for repo-specific conventions, architecture decisions, context-graph "
        "relationships, and file drift baselines across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "read",
                    "add_note",
                    "list_notes",
                    "record_decision",
                    "update_decision",
                    "list_decisions",
                    "relate",
                    "view_graph",
                    "track_paths",
                    "drift_check",
                ],
            },
            "content": {"type": "string", "description": "Project note content for add_note."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string", "description": "Decision summary for record_decision."},
            "rationale": {"type": "string"},
            "related_files": {"type": "array", "items": {"type": "string"}},
            "decision_id": {"type": "string"},
            "outcome": {"type": "string"},
            "status": {"type": "string"},
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "project_root": {"type": "string", "description": "Optional override for the project root path."},
        },
        "required": ["action"],
    },
}


def project_memory_tool(
    action: str,
    content: Optional[str] = None,
    tags: Optional[List[str]] = None,
    summary: Optional[str] = None,
    rationale: Optional[str] = None,
    related_files: Optional[List[str]] = None,
    decision_id: Optional[str] = None,
    outcome: Optional[str] = None,
    status: Optional[str] = None,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_: Optional[str] = None,
    paths: Optional[List[str]] = None,
    limit: int = 10,
    project_root: Optional[str] = None,
    store: Optional[ProjectMemoryStore] = None,
) -> str:
    if store is None:
        store = ProjectMemoryStore(project_root=project_root)
        store.load_from_disk()
    if action == "read":
        result = store.overview(limit=limit)
    elif action == "add_note":
        result = store.add_note(content or "", tags=tags)
    elif action == "list_notes":
        result = store.list_notes(limit=limit)
    elif action == "record_decision":
        result = store.record_decision(summary or "", rationale=rationale, related_files=related_files, tags=tags, status=status or "open")
    elif action == "update_decision":
        result = store.update_decision(decision_id or "", outcome or "", status=status)
    elif action == "list_decisions":
        result = store.list_decisions(limit=limit)
    elif action == "relate":
        result = store.relate(subject or "", predicate or "", object_ or "")
    elif action == "view_graph":
        result = store.view_graph(limit=limit)
    elif action == "track_paths":
        result = store.track_paths(paths or [])
    elif action == "drift_check":
        result = store.drift_check(paths=paths)
    else:
        result = {"success": False, "error": f"Unknown action '{action}'."}
    return json.dumps(result, ensure_ascii=False)


def check_project_memory_requirements() -> bool:
    return True


registry.register(
    name="project_memory",
    toolset="project_memory",
    schema=PROJECT_MEMORY_SCHEMA,
    handler=lambda args, **kw: project_memory_tool(
        action=args.get("action", "read"),
        content=args.get("content"),
        tags=args.get("tags"),
        summary=args.get("summary"),
        rationale=args.get("rationale"),
        related_files=args.get("related_files"),
        decision_id=args.get("decision_id"),
        outcome=args.get("outcome"),
        status=args.get("status"),
        subject=args.get("subject"),
        predicate=args.get("predicate"),
        object_=args.get("object"),
        paths=args.get("paths"),
        limit=args.get("limit", 10),
        project_root=args.get("project_root"),
        store=kw.get("store"),
    ),
    check_fn=check_project_memory_requirements,
    emoji="🗂️",
)
