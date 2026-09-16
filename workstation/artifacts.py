"""Durable Artifact Store for Hermes Workstation.

Separates the Data Plane (raw scrapes, full terminal outputs, datasets, captures)
from the Reasoning Plane (compact LLM context). Large outputs are persisted by
reference (``artifact://...``) and read on demand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ArtifactRef:
    ref: str
    name: str
    task_id: str
    local_path: str
    size_bytes: int
    sha256: str
    schema: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_model_reference(self, signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compact representation for model context."""
        payload = {
            "artifact_ref": self.ref,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256[:12] + "...",
            "summary": self.summary,
        }
        if self.schema:
            payload["schema"] = self.schema
        if signals:
            payload["signals"] = signals
        return payload


class ArtifactStore:
    """Thread-safe disk store for raw data payloads."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        if root_dir is not None:
            self.root = Path(root_dir)
        else:
            self.root = get_hermes_home() / "workstation" / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _task_dir(self, task_id: str) -> Path:
        safe_task = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in task_id)
        d = self.root / safe_task
        d.mkdir(parents=True, exist_ok=True)
        return d

    def store(
        self,
        task_id: str,
        name: str,
        content: Union[str, bytes, Dict[str, Any], List[Any]],
        *,
        schema: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        """Store content atomically and return an ArtifactRef."""
        task_dir = self._task_dir(task_id)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
        if safe_name in {"", ".", ".."}:
            raise ValueError("artifact name must identify a file")
        target_path = task_dir / safe_name

        if isinstance(content, (dict, list)):
            raw_bytes = json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")
        elif isinstance(content, str):
            raw_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            raw_bytes = content
        else:
            raw_bytes = str(content).encode("utf-8")

        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        size_bytes = len(raw_bytes)

        # Atomic write
        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{threading.get_ident()}")
        with self._lock:
            temp_path.write_bytes(raw_bytes)
            temp_path.replace(target_path)

        # Meta descriptor
        safe_task = task_dir.name
        ref_uri = f"artifact://tasks/{safe_task}/{safe_name}"
        computed_summary = summary or {}
        if not computed_summary and isinstance(content, (dict, list)):
            computed_summary = {
                "item_count": len(content),
                "type": type(content).__name__,
            }

        ref = ArtifactRef(
            ref=ref_uri,
            name=safe_name,
            task_id=task_id,
            local_path=str(target_path),
            size_bytes=size_bytes,
            sha256=sha256,
            schema=schema,
            summary=computed_summary,
        )

        meta_path = target_path.with_suffix(target_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(ref.to_dict(), indent=2), encoding="utf-8")
        return ref

    def store_json(
        self,
        content: Union[Dict[str, Any], List[Any]],
        *,
        task_id: str = "default",
        name: str = "data.json",
        schema: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store json dictionary/list content directly and return artifact_ref string."""
        ref = self.store(task_id=task_id, name=name, content=content, schema=schema, summary=summary)
        return ref.ref

    def resolve_ref(self, ref_uri: str) -> Optional[Path]:
        """Resolve only canonical artifact:// references and enforce root containment."""
        prefix = "artifact://tasks/"
        if not isinstance(ref_uri, str) or not ref_uri.startswith(prefix):
            return None
        tail = ref_uri[len(prefix):]
        if "\x00" in tail or "\\" in tail:
            return None
        parts = tail.split("/")
        if len(parts) != 2:
            return None
        task_component, name = parts
        if not task_component or not name or task_component in {".", ".."} or name in {".", ".."}:
            return None
        safe_task = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in task_component)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
        if safe_task != task_component or safe_name != name:
            return None
        root = self.root.resolve()
        candidate = (root / safe_task / safe_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        if candidate.parent != (root / safe_task).resolve():
            return None
        return candidate if candidate.is_file() else None


    def resolve_path(self, ref_uri: str) -> Optional[Path]:
        """Alias for resolve_ref returning Path or None."""
        return self.resolve_ref(ref_uri)

    def read(self, ref_uri: str) -> str:
        path = self.resolve_ref(ref_uri)
        if not path or not path.exists():
            raise FileNotFoundError(f"Artifact {ref_uri} not found")
        return path.read_text(encoding="utf-8", errors="replace")

    def read_json(self, ref_uri: str) -> Any:
        raw = self.read(ref_uri)
        return json.loads(raw)

    def list_artifacts(self, task_id: str) -> List[ArtifactRef]:
        task_dir = self._task_dir(task_id)
        results: List[ArtifactRef] = []
        for meta_file in task_dir.glob("*.meta.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                results.append(
                    ArtifactRef(
                        ref=data["ref"],
                        name=data["name"],
                        task_id=data["task_id"],
                        local_path=data["local_path"],
                        size_bytes=data["size_bytes"],
                        sha256=data["sha256"],
                        schema=data.get("schema"),
                        summary=data.get("summary", {}),
                        created_at=data.get("created_at", _utc_now()),
                    )
                )
            except Exception:
                continue
        return results
