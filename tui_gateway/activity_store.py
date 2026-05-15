"""Durable activity inbox + artifact registry for gateway clients.

The activity inbox is intentionally small and transport-neutral: gateway code can
record high-signal events once, then native/web/TUI clients can list and inspect
them without scraping chat history or reading arbitrary files.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_ALLOWED_ARTIFACT_BYTES = 10 * 1024 * 1024
_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _now() -> float:
    return time.time()


def _safe_name(name: str, fallback: str = "artifact") -> str:
    cleaned = _FILENAME_RE.sub("_", (name or fallback).strip()).strip(". ")
    return cleaned[:160] or fallback


def _preview(text: str, limit: int = 240) -> str:
    single = " ".join((text or "").split())
    if len(single) <= limit:
        return single
    return single[: limit - 1].rstrip() + "…"


class ActivityStore:
    """JSONL-backed activity store with file-backed artifacts.

    V0 keeps the storage deliberately simple and append-friendly. Mutating read
    state rewrites the compact JSONL index atomically; artifacts are addressed by
    generated IDs and never by caller-supplied paths.
    """

    def __init__(self, root: str | Path | None = None):
        base = Path(root) if root is not None else Path(get_hermes_home()) / "activity"
        self.root = base
        self.artifacts_root = base / "artifacts"
        self.index_path = base / "activity.jsonl"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        kind: str,
        title: str,
        summary: str = "",
        severity: str = "info",
        source: str = "gateway",
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
        actions: list[dict[str, Any]] | None = None,
        external_refs: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        activity_id: str | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        now = _now() if created_at is None else float(created_at)
        item_id = activity_id or f"act_{uuid.uuid4().hex[:16]}"
        item_artifacts = [self._store_artifact(item_id, art) for art in artifacts or []]
        item = {
            "id": item_id,
            "created_at": now,
            "updated_at": now,
            "kind": kind or "activity",
            "severity": severity or "info",
            "source": source or "gateway",
            "title": title or kind or "Activity",
            "summary": summary or "",
            "session_id": session_id or "",
            "read": False,
            "dismissed": False,
            "actions": actions or [],
            "external_refs": external_refs or [],
            "artifacts": item_artifacts,
            "payload": payload or {},
        }
        with self._lock:
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        return item

    def list(self, *, limit: int = 100, include_read: bool = True, include_dismissed: bool = False) -> list[dict[str, Any]]:
        items = self._read_all()
        if not include_read:
            items = [i for i in items if not i.get("read")]
        if not include_dismissed:
            items = [i for i in items if not i.get("dismissed")]
        items.sort(key=lambda i: float(i.get("created_at") or 0), reverse=True)
        return items[: max(1, min(int(limit or 100), 500))]

    def get(self, activity_id: str) -> dict[str, Any] | None:
        for item in self._read_all():
            if item.get("id") == activity_id:
                return item
        return None

    def mark_read(self, activity_id: str, read: bool = True) -> dict[str, Any] | None:
        return self._update(activity_id, {"read": bool(read), "updated_at": _now()})

    def dismiss(self, activity_id: str, dismissed: bool = True) -> dict[str, Any] | None:
        return self._update(activity_id, {"dismissed": bool(dismissed), "updated_at": _now()})

    def list_artifacts(self, activity_id: str) -> list[dict[str, Any]]:
        item = self.get(activity_id)
        return list((item or {}).get("artifacts") or [])

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        artifact_id = str(artifact_id or "")
        if not artifact_id.startswith("art_"):
            return None
        for item in self._read_all():
            for meta in item.get("artifacts") or []:
                if meta.get("id") == artifact_id:
                    path = self._artifact_path(meta)
                    if path is None or not path.exists() or not path.is_file():
                        return None
                    data = path.read_bytes()
                    result = dict(meta)
                    result["activity_id"] = item.get("id", "")
                    if self._is_text(meta.get("mime_type", ""), path.name):
                        result["content"] = data.decode("utf-8", errors="replace")
                        result["encoding"] = "utf-8"
                    else:
                        result["content_base64"] = base64.b64encode(data).decode("ascii")
                        result["encoding"] = "base64"
                    return result
        return None

    def _read_all(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.index_path.exists():
                return []
            items: list[dict[str, Any]] = []
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except Exception:
                    continue
                if isinstance(value, dict):
                    items.append(value)
            return items

    def _update(self, activity_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            items = self._read_all()
            found: dict[str, Any] | None = None
            for item in items:
                if item.get("id") == activity_id:
                    item.update(updates)
                    found = item
                    break
            if found is None:
                return None
            tmp = self.index_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for item in items:
                    fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(tmp, self.index_path)
            return found

    def _store_artifact(self, activity_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
        art_id = f"art_{uuid.uuid4().hex[:16]}"
        name = _safe_name(str(artifact.get("name") or art_id))
        mime_type = str(artifact.get("mime_type") or mimetypes.guess_type(name)[0] or "application/octet-stream")
        data = self._artifact_bytes(artifact)
        if len(data) > _ALLOWED_ARTIFACT_BYTES:
            raise ValueError(f"artifact {name} exceeds {_ALLOWED_ARTIFACT_BYTES} bytes")
        directory = self.artifacts_root / art_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / name
        path.write_bytes(data)
        rel = path.relative_to(self.artifacts_root).as_posix()
        return {
            "id": art_id,
            "name": name,
            "mime_type": mime_type,
            "size": len(data),
            "created_at": _now(),
            "preview": artifact.get("preview") or _preview(data.decode("utf-8", errors="ignore")) if self._is_text(mime_type, name) else "",
            "storage": rel,
        }

    def _artifact_bytes(self, artifact: dict[str, Any]) -> bytes:
        if "content_base64" in artifact:
            return base64.b64decode(str(artifact.get("content_base64") or ""), validate=True)
        content = artifact.get("content", artifact.get("text", ""))
        if isinstance(content, bytes):
            return content
        return str(content).encode("utf-8")

    def _artifact_path(self, meta: dict[str, Any]) -> Path | None:
        storage = str(meta.get("storage") or "")
        if not storage or storage.startswith("/") or ".." in Path(storage).parts:
            return None
        path = (self.artifacts_root / storage).resolve()
        try:
            path.relative_to(self.artifacts_root.resolve())
        except ValueError:
            return None
        return path

    @staticmethod
    def _is_text(mime_type: str, name: str) -> bool:
        mt = (mime_type or "").lower()
        if mt.startswith("text/") or mt in {"application/json", "application/xml", "application/javascript"}:
            return True
        return Path(name).suffix.lower() in {".md", ".txt", ".log", ".json", ".csv", ".html", ".htm", ".xml"}


def activity_from_gateway_event(kind: str, session_id: str, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map noisy live gateway events into durable high-signal inbox items."""
    payload = payload or {}
    if kind == "message.complete":
        status = str(payload.get("status") or "complete")
        if status == "interrupted":
            return None
        text = str(payload.get("text") or payload.get("rendered") or "")
        return {
            "kind": kind,
            "title": "Session response completed" if status == "complete" else "Session response failed",
            "summary": _preview(text or status),
            "severity": "error" if status == "error" else "info",
            "source": "session",
        }
    if kind == "approval.request":
        command = str(payload.get("command") or payload.get("raw_args") or "Approval requested")
        return {
            "kind": kind,
            "title": "Approval required",
            "summary": _preview(command),
            "severity": "warning",
            "source": "approval",
        }
    if kind == "clarify.request":
        return {
            "kind": kind,
            "title": "Clarification needed",
            "summary": _preview(str(payload.get("question") or "Hermes needs input")),
            "severity": "warning",
            "source": "clarify",
        }
    if kind == "background.complete":
        return {
            "kind": kind,
            "title": "Background task completed",
            "summary": _preview(str(payload.get("text") or "Done")),
            "severity": "info",
            "source": "background",
        }
    if kind == "error":
        return {
            "kind": kind,
            "title": "Gateway error",
            "summary": _preview(str(payload.get("message") or "Unknown error")),
            "severity": "error",
            "source": "gateway",
        }
    return None
