"""Session ownership and migration metadata over Hermes' canonical sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import shutil
from typing import Any, Callable
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionTemperature(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    model_id: str | None = None
    provider: str | None = None
    toolset_fingerprint: str | None = None
    worker_ids: list[str] = field(default_factory=list)
    temperature: SessionTemperature = SessionTemperature.HOT
    schema_version: int = 1
    context_tokens: int = 0
    context_token_limit: int | None = None
    last_compacted_at: str | None = None
    compaction_count: int = 0
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["temperature"] = self.temperature.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=str(data["session_id"]),
            model_id=data.get("model_id"),
            provider=data.get("provider"),
            toolset_fingerprint=data.get("toolset_fingerprint"),
            worker_ids=[str(item) for item in data.get("worker_ids", [])],
            temperature=SessionTemperature(data.get("temperature", SessionTemperature.HOT.value)),
            schema_version=int(data.get("schema_version", 1)),
            context_tokens=max(0, int(data.get("context_tokens", 0))),
            context_token_limit=(
                None if data.get("context_token_limit") is None else max(1, int(data["context_token_limit"]))
            ),
            last_compacted_at=data.get("last_compacted_at"),
            compaction_count=max(0, int(data.get("compaction_count", 0))),
            updated_at=str(data.get("updated_at", _utc_now())),
        )


class SessionLease:
    """Cross-process exclusive lease for a writable session projection."""

    def __init__(self, path: Path, *, owner_id: str) -> None:
        self.path = Path(path)
        self.owner_id = owner_id
        self.lease_id = f"lease-{uuid4().hex}"
        self.acquired = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"owner_id": self.owner_id, "lease_id": self.lease_id, "acquired_at": _utc_now()}
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream)
        except FileExistsError as exc:
            raise RuntimeError(f"session lease is already owned: {self.path}") from exc
        self.acquired = True
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            self.acquired = False
            return
        if data.get("lease_id") != self.lease_id:
            raise RuntimeError("session lease ownership changed before release")
        self.path.unlink(missing_ok=True)
        self.acquired = False


class SessionLifecycleStore:
    """Durable model/binding/temperature projection, not a replacement SessionDB."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._sessions: dict[str, SessionRecord] = {}
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            self.load_error = str(exc)
            return
        for item in data.get("sessions", []) if isinstance(data, dict) else []:
            try:
                record = SessionRecord.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
            self._sessions[record.session_id] = record

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps({"schema_version": 1, "sessions": [item.to_dict() for item in self._sessions.values()]}, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def register(
        self,
        session_id: str,
        *,
        model_id: str | None = None,
        provider: str | None = None,
        toolset_fingerprint: str | None = None,
        worker_ids: list[str] | None = None,
    ) -> SessionRecord:
        record = self._sessions.get(session_id) or SessionRecord(session_id=session_id)
        record.model_id = model_id if model_id is not None else record.model_id
        record.provider = provider if provider is not None else record.provider
        record.toolset_fingerprint = toolset_fingerprint if toolset_fingerprint is not None else record.toolset_fingerprint
        if worker_ids is not None:
            record.worker_ids = list(worker_ids)
        record.updated_at = _utc_now()
        self._sessions[session_id] = record
        self._persist()
        return record

    def get(self, session_id: str) -> SessionRecord:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "session_count": len(self._sessions),
            "load_error": self.load_error,
        }

    def set_temperature(self, session_id: str, temperature: SessionTemperature) -> SessionRecord:
        record = self.get(session_id)
        record.temperature = temperature
        record.updated_at = _utc_now()
        self._persist()
        return record

    def update_context_usage(
        self,
        session_id: str,
        *,
        context_tokens: int,
        context_token_limit: int | None = None,
    ) -> SessionRecord:
        record = self.get(session_id)
        record.context_tokens = max(0, int(context_tokens))
        if context_token_limit is not None:
            record.context_token_limit = max(1, int(context_token_limit))
        record.updated_at = _utc_now()
        self._persist()
        return record

    def should_compact(self, session_id: str, *, reserve_tokens: int = 0) -> bool:
        record = self.get(session_id)
        if record.context_token_limit is None:
            return False
        return record.context_tokens + max(0, reserve_tokens) >= record.context_token_limit

    def mark_compacted(self, session_id: str, *, resulting_tokens: int) -> SessionRecord:
        record = self.get(session_id)
        record.context_tokens = max(0, int(resulting_tokens))
        record.compaction_count += 1
        record.last_compacted_at = _utc_now()
        record.updated_at = record.last_compacted_at
        self._persist()
        return record

    def migrate(
        self,
        session_id: str,
        *,
        target_version: int,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        validate: Callable[[dict[str, Any]], bool],
    ) -> SessionRecord:
        record = self.get(session_id)
        if target_version <= record.schema_version:
            return record
        original = record.to_dict()
        backup = self.path.with_suffix(self.path.suffix + f".{record.schema_version}.bak")
        shutil.copy2(self.path, backup)
        try:
            migrated = transform(dict(original))
            migrated["schema_version"] = target_version
            if not validate(migrated):
                raise ValueError("session migration validation failed")
            updated = SessionRecord.from_dict(migrated)
            self._sessions[session_id] = updated
            self._persist()
            return updated
        except Exception:
            self._sessions[session_id] = SessionRecord.from_dict(original)
            self._persist()
            raise
