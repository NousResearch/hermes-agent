"""Bounded persistence/reconnect soak over canonical Workstation owners."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from workstation.contracts import ExecutionEventKind
from workstation.evaluation import SoakResult, SoakRunner
from workstation.journal import ExecutionJournal
from workstation.memory import MemoryKind, ProceduralMemory
from workstation.session_lifecycle import SessionLease, SessionLifecycleStore, SessionTemperature
from workstation.workers import WorkerRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]
_CHILD_TIMEOUT_SECONDS = 30.0
_TASK_CONTEXT_RECORD_LIMIT = 8


@dataclass(slots=True)
class WorkstationSoakReport:
    """Serializable report for the contract-level reconnect scenario."""

    scenario: str
    result: SoakResult
    session_count: int
    process_restarts: int
    journal_events: int
    memory_records: int
    max_live_memory_records: int
    memory_snapshots: int
    migrated_sessions: int
    model_changes: int
    cold_reloads: int
    reconstructed_workers: int
    worker_storage: str
    retained_root: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["result"] = asdict(self.result)
        return data


def run_workstation_soak(
    *,
    duration_seconds: float,
    max_iterations: int,
    root: Path | None = None,
    session_count: int = 3,
    child_timeout_seconds: float = _CHILD_TIMEOUT_SECONDS,
) -> WorkstationSoakReport:
    """Exercise durable sessions, memory and worker reconnect for a duration.

    Every iteration is executed in a fresh Python child process. The child
    reconstructs all owner objects from the same durable paths, which makes a
    successful iteration an actual process-restart boundary rather than only a
    new set of in-memory objects. The scenario remains model-, browser- and
    service-independent; it is contract evidence, not Desktop production
    evidence.
    """

    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if session_count <= 0:
        raise ValueError("session_count must be positive")
    if child_timeout_seconds <= 0:
        raise ValueError("child_timeout_seconds must be positive")

    if root is not None:
        target = Path(root).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return _run_in_root(
            target,
            duration_seconds,
            max_iterations,
            session_count,
            child_timeout_seconds,
            retained_root=True,
        )

    with tempfile.TemporaryDirectory(prefix="hermes-workstation-soak-") as raw_root:
        return _run_in_root(
            Path(raw_root),
            duration_seconds,
            max_iterations,
            session_count,
            child_timeout_seconds,
            retained_root=False,
        )


def _run_in_root(
    root: Path,
    duration_seconds: float,
    max_iterations: int,
    session_count: int,
    child_timeout_seconds: float,
    *,
    retained_root: bool,
) -> WorkstationSoakReport:
    root.mkdir(parents=True, exist_ok=True)
    workers_path = root / "workers.json"
    counters = {
        "memory_records": 0,
        "max_live_memory_records": 0,
        "memory_snapshots": 0,
        "migrated_sessions": 0,
        "model_changes": 0,
        "cold_reloads": 0,
        "reconstructed_workers": 0,
    }

    def iteration(index: int) -> dict[str, Any]:
        started = time.monotonic()
        command = [
            sys.executable,
            "-m",
            "workstation.soak",
            "--child-root",
            str(root),
            "--child-iteration",
            str(index),
            "--child-session-count",
            str(session_count),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=child_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "success": False,
                "actions": 0,
                "latency_seconds": time.monotonic() - started,
                "metadata": {"iteration": index, "error": str(exc)},
            }
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        metadata = payload.get("metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        if completed.returncode != 0:
            metadata.setdefault("error", completed.stderr.strip() or "soak child failed")
            payload["success"] = False
        for key in counters:
            if key == "max_live_memory_records":
                continue
            counters[key] += max(0, int(metadata.get(key, 0)))
        counters["max_live_memory_records"] = max(
            counters["max_live_memory_records"],
            max(0, int(metadata.get("max_live_memory_records", 0))),
        )
        return {
            "success": bool(payload.get("success", False)),
            "actions": max(0, int(payload.get("actions", 0))),
            "latency_seconds": max(0.0, float(payload.get("latency_seconds", time.monotonic() - started))),
            "metadata": metadata,
        }

    result = SoakRunner().run_for_duration(
        duration_seconds,
        iteration,
        max_iterations=max_iterations,
    )
    journal_events = 0
    for session_index in range(session_count):
        journal = ExecutionJournal(
            f"soak-task-{session_index}",
            f"soak-session-{session_index}",
            file_path=root / f"execution-{session_index}.jsonl",
        )
        journal_events += len(journal.read_events())
    final_store = SessionLifecycleStore(root / "sessions.json")
    return WorkstationSoakReport(
        scenario="canonical-session-memory-worker-process-restart",
        result=result,
        session_count=final_store.diagnostics()["session_count"],
        process_restarts=result.completed_iterations,
        journal_events=journal_events,
        memory_records=counters["memory_records"],
        max_live_memory_records=counters["max_live_memory_records"],
        memory_snapshots=counters["memory_snapshots"],
        migrated_sessions=counters["migrated_sessions"],
        model_changes=counters["model_changes"],
        cold_reloads=counters["cold_reloads"],
        reconstructed_workers=counters["reconstructed_workers"],
        worker_storage=str(workers_path),
        retained_root=retained_root,
    )


def _run_child_iteration(root: Path, index: int, session_count: int) -> dict[str, Any]:
    if session_count <= 0:
        raise ValueError("session_count must be positive")
    started = time.monotonic()
    sessions_path = root / "sessions.json"
    workers_path = root / "workers.json"
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "iteration": index,
        "memory_records": 0,
        "max_live_memory_records": 0,
        "memory_snapshots": 0,
        "migrated_sessions": 0,
        "model_changes": 0,
        "cold_reloads": 0,
        "reconstructed_workers": 0,
    }
    try:
        for session_index in range(session_count):
            session_id = f"soak-session-{session_index}"
            task_id = f"soak-task-{session_index}"
            worker_id = f"soak-worker-{session_index}"
            journal = ExecutionJournal(
                task_id,
                session_id,
                file_path=root / f"execution-{session_index}.jsonl",
            )
            lease = SessionLease(root / f"{session_id}.lock", owner_id=f"soak-process-{index}")
            registry = WorkerRegistry(storage_path=workers_path)
            worker = None
            try:
                lease.acquire()
                store = SessionLifecycleStore(sessions_path)
                try:
                    existing = store.get(session_id)
                except KeyError:
                    existing = None
                model_id = f"soak-model-{index % 2}"
                if existing is not None and existing.temperature == SessionTemperature.COLD:
                    store.set_temperature(session_id, SessionTemperature.HOT)
                    metadata["cold_reloads"] += 1
                if existing is not None and existing.model_id != model_id:
                    metadata["model_changes"] += 1
                store.register(
                    session_id,
                    model_id=model_id,
                    provider="soak",
                    worker_ids=[worker_id],
                )
                if existing is None:
                    store.migrate(
                        session_id,
                        target_version=2,
                        transform=lambda data: {**data, "toolset_fingerprint": "soak-v2"},
                        validate=lambda data: data.get("toolset_fingerprint") == "soak-v2",
                    )
                    metadata["migrated_sessions"] += 1
                store.update_context_usage(
                    session_id,
                    context_tokens=index + session_index + 1,
                    context_token_limit=10_000,
                )
                journal.record(
                    ExecutionEventKind.ACTION,
                    f"soak iteration {index} started",
                    metadata={"iteration": index, "process_restart": True},
                )

                memory = ProceduralMemory(root / "memory.json")
                memory.record_memory(
                    MemoryKind.TASK_CONTEXT,
                    f"soak iteration {index} session {session_index}",
                    workspace_id=session_id,
                    source="workstation.soak",
                )
                snapshot_path = memory.snapshot(root / "memory.snapshot.json")
                reloaded_memory = ProceduralMemory(root / "memory.json")
                reloaded_memory.restore_snapshot(snapshot_path)
                reloaded_memory.compact_memory(
                    max_records=_TASK_CONTEXT_RECORD_LIMIT,
                    workspace_id=session_id,
                    kind=MemoryKind.TASK_CONTEXT,
                )
                live_memory_records = len(reloaded_memory.list_memory(include_stale=True))
                metadata["memory_records"] += live_memory_records
                metadata["max_live_memory_records"] = max(
                    metadata["max_live_memory_records"], live_memory_records
                )
                metadata["memory_snapshots"] += 1

                def execute(content: str) -> dict[str, Any]:
                    return {"deliverables": [content], "usage": {"actions": 1}}

                try:
                    worker = registry.reconstruct_worker(worker_id, executor_fn=execute, journal=journal)
                    metadata["reconstructed_workers"] += 1
                except KeyError:
                    worker = registry.start_persistent_worker(
                        worker_id,
                        task_id,
                        session_id,
                        executor_fn=execute,
                        journal=journal,
                    )
                registry.send_message(worker_id, f"iteration-{index}")
                result = registry.wait(worker_id, timeout=5.0)
                if result is None or result.status != "completed" or result.parent_task_id != task_id:
                    return _child_failure(metadata, started, f"worker result failed for {session_id}")
                journal.record(
                    ExecutionEventKind.ACTION,
                    f"soak iteration {index} completed",
                    metadata={
                        "iteration": index,
                        "sequence": result.sequence,
                        "model_id": store.get(session_id).model_id,
                    },
                )
                store.set_temperature(session_id, SessionTemperature.COLD)
            finally:
                if worker is not None:
                    try:
                        registry.stop_worker(worker_id)
                    except Exception:
                        pass
                lease.release()
    except Exception as exc:
        return _child_failure(metadata, started, str(exc))
    return {
        "success": True,
        "actions": session_count * 5,
        "latency_seconds": time.monotonic() - started,
        "metadata": metadata,
    }


def _child_failure(metadata: dict[str, Any], started: float, error: str) -> dict[str, Any]:
    metadata = dict(metadata)
    metadata["error"] = error
    return {
        "success": False,
        "actions": 1,
        "latency_seconds": time.monotonic() - started,
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Workstation persistence/reconnect soak")
    parser.add_argument("--duration", type=float, default=60.0, help="maximum duration in seconds")
    parser.add_argument("--iterations", type=int, default=1000, help="hard iteration cap")
    parser.add_argument("--sessions", type=int, default=3, help="number of independent sessions per iteration")
    parser.add_argument("--root", type=Path, help="retain durable soak files under this directory")
    parser.add_argument("--report", type=Path, help="also write the JSON report to this path")
    parser.add_argument("--child-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-iteration", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-session-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child_root is not None:
        if args.child_iteration is None or args.child_session_count is None:
            parser.error("child mode requires iteration and session count")
        payload = _run_child_iteration(args.child_root.resolve(), args.child_iteration, args.child_session_count)
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if payload["success"] else 1

    report = run_workstation_soak(
        duration_seconds=args.duration,
        max_iterations=args.iterations,
        root=args.root,
        session_count=args.sessions,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.result.failures == 0 and not report.result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
