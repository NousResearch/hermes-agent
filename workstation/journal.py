from __future__ import annotations

import json
import hashlib
import os
from contextlib import contextmanager
from uuid import uuid4
from functools import lru_cache
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from hermes_constants import get_hermes_home
from workstation.contracts import EvidenceRef, ExecutionEvent, ExecutionEventKind, RiskLevel, utc_now


class JournalIntegrityError(RuntimeError):
    """Evidence cannot be trusted after a malformed or altered journal line."""


@contextmanager
def _writer_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    if not lock_path.exists() or lock_path.stat().st_size == 0:
        try:
            with lock_path.open("a+b") as init_f:
                if init_f.tell() == 0:
                    init_f.write(b"0")
                    init_f.flush()
        except OSError:
            pass
    with lock_path.open("r+b" if os.name == "nt" else "a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _event_hash(data: dict) -> str:
    body = {k: v for k, v in data.items() if k != "event_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _build_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=2)
        value = result.stdout.strip()
        return value if result.returncode == 0 and len(value) == 40 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def execution_provenance() -> dict[str, Any]:
    from workstation import __version__
    return {"environment": "test" if os.environ.get("HERMES_TEST_ISOLATION") or os.environ.get("PYTEST_CURRENT_TEST") else "production",
            "build_sha": _build_sha(), "workstation_version": __version__}


def get_journal_dir() -> Path:
    """Resolve the directory for durable workstation execution journals."""
    d = get_hermes_home() / "workstation" / "journals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_journal_path(task_id: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in task_id)
    return get_journal_dir() / f"{safe_name}.jsonl"


class ExecutionJournal:
    """Durable append-only execution and evidence journal for Workstation tasks."""

    def __init__(self, task_id: str, session_id: str, *, file_path: Path | None = None,
                 environment: str | None = None, build_sha: str | None = None,
                 workstation_version: str | None = None, test_case_id: str | None = None,
                 evaluation_run_id: str | None = None) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self.file_path = file_path or get_journal_path(task_id)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer_id = str(uuid4())
        self.environment = environment or ("test" if os.environ.get("HERMES_TEST_ISOLATION") or os.environ.get("PYTEST_CURRENT_TEST") else "production")
        if self.environment not in {"production", "dogfood", "benchmark", "test", "e2e", "replay"}:
            raise ValueError("invalid execution environment")
        from workstation import __version__
        self.provenance = {"build_sha": build_sha or _build_sha(), "workstation_version": workstation_version or __version__,
                           "test_case_id": test_case_id, "evaluation_run_id": evaluation_run_id}

    def append(self, event: ExecutionEvent) -> None:
        if event.task_id != self.task_id or event.session_id != self.session_id:
            raise ValueError("journal event lineage mismatch")
        if event.kind == ExecutionEventKind.TASK_COMPLETED and event.metadata.get("completed") is False:
            raise ValueError("incomplete outcome cannot emit TASK_COMPLETED")
        with _writer_lock(self.file_path):
            records = self._read_records()
            data = event.to_dict()
            data.update(schema_version=2, sequence_number=len(records) + 1,
                        previous_event_hash=_event_hash(records[-1]) if records else None,
                        writer_id=self.writer_id, environment=self.environment, **self.provenance)
            data["event_hash"] = _event_hash(data)
            for key in ("schema_version", "sequence_number", "previous_event_hash", "event_hash",
                        "writer_id", "environment", *self.provenance):
                setattr(event, key, data[key])
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                if event.metadata.get("boundary") == "acceptance" or event.kind in {ExecutionEventKind.TASK_COMPLETED, ExecutionEventKind.APPROVAL_RESOLVED,
                                  ExecutionEventKind.RECOVERY, ExecutionEventKind.DELIVERABLE}:
                    os.fsync(f.fileno())

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []
        records = []
        with self.file_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError("event is not an object")
                    ExecutionEventKind(data["kind"])
                    RiskLevel(data.get("risk", "low"))
                    for key in ("task_id", "session_id", "message"):
                        if not isinstance(data[key], str):
                            raise ValueError(f"invalid {key}")
                    if data.get("event_hash"):
                        if data.get("schema_version") != 2 or data.get("sequence_number") != len(records) + 1:
                            raise ValueError("sequence/schema mismatch")
                        previous = _event_hash(records[-1]) if records else None
                        if data.get("previous_event_hash") != previous or data["event_hash"] != _event_hash(data):
                            raise ValueError("hash chain break")
                    elif records and records[-1].get("event_hash"):
                        raise ValueError("legacy event after verified chain")
                    records.append(data)
                except (ValueError, TypeError, KeyError) as exc:
                    raise JournalIntegrityError(f"JOURNAL_DEGRADED {self.file_path}:{line_number}: {exc}") from exc
        return records

    def integrity(self) -> dict[str, Any]:
        try:
            records = self._read_records()
            return {"status": "verified" if all(r.get("event_hash") for r in records) else "legacy_unverified",
                    "events": len(records)}
        except JournalIntegrityError as exc:
            return {"status": "JOURNAL_DEGRADED", "diagnostic": str(exc)}

    def record(
        self,
        kind: ExecutionEventKind,
        message: str,
        *,
        url: str | None = None,
        risk: RiskLevel = RiskLevel.LOW,
        evidence: list[EvidenceRef] | None = None,
        metadata: dict[str, Any] | None = None,
        browser_tab_id: str | None = None,
    ) -> ExecutionEvent:
        event = ExecutionEvent(
            kind=kind,
            task_id=self.task_id,
            session_id=self.session_id,
            message=message,
            url=url,
            risk=risk,
            evidence=evidence or [],
            metadata=metadata or {},
            browser_tab_id=browser_tab_id,
        )
        self.append(event)
        return event

    def read_events(self) -> list[ExecutionEvent]:
        if not self.file_path.exists():
            return []
        events: list[ExecutionEvent] = []
        for data in self._read_records():
            try:
                ev_refs = [
                    EvidenceRef(**ref) if isinstance(ref, dict) else ref
                    for ref in data.get("evidence", [])
                ]
                event = ExecutionEvent(
                    kind=ExecutionEventKind(data["kind"]),
                    task_id=data["task_id"],
                    session_id=data["session_id"],
                    message=data["message"],
                    timestamp=data.get("timestamp", utc_now()),
                    event_id=data.get("event_id", ""),
                    browser_tab_id=data.get("browser_tab_id"),
                    url=data.get("url"),
                    risk=RiskLevel(data.get("risk", "low")),
                    evidence=ev_refs,
                    metadata=data.get("metadata", {}),
                    **{k: data.get(k) for k in ("schema_version", "sequence_number", "previous_event_hash",
                        "event_hash", "writer_id", "environment", "build_sha", "workstation_version",
                        "test_case_id", "evaluation_run_id")},
                )
                events.append(event)
            except (ValueError, TypeError, KeyError) as exc:
                raise JournalIntegrityError(f"JOURNAL_DEGRADED {self.file_path}: invalid event: {exc}") from exc
        return events

    def read_timeline(self) -> list[dict[str, Any]]:
        """Return chronological timeline events enriched with elapsed durations."""
        events = self.read_events()
        events.sort(key=lambda e: e.timestamp)
        timeline: list[dict[str, Any]] = []
        prev_dt = None

        from datetime import datetime
        for ev in events:
            d = ev.to_dict()
            try:
                cur_dt = datetime.fromisoformat(ev.timestamp)
                if prev_dt:
                    d["elapsed_seconds"] = max(0.0, round((cur_dt - prev_dt).total_seconds(), 2))
                else:
                    d["elapsed_seconds"] = 0.0
                prev_dt = cur_dt
            except Exception:
                d["elapsed_seconds"] = 0.0
            timeline.append(d)

        return timeline


def get_task_timeline(task_id: str) -> list[dict[str, Any]]:
    """Retrieve formatted execution timeline for a specific task."""
    j = ExecutionJournal(task_id, "")
    return j.read_timeline()
