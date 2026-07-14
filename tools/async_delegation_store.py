"""Profile-scoped durable state for gateway background delegations.

The JSON registry is an intent/outbox store, not a second delivery consumer.
Live events continue to use ``process_registry.completion_queue``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_REDISPATCH_ATTEMPTS = 2
MAX_REGISTRY_RECORDS = 256
TERMINAL_RETENTION_SECONDS = 7 * 24 * 60 * 60
ABSOLUTE_RETENTION_SECONDS = 30 * 24 * 60 * 60
ACTIVE_STALE_SECONDS = 30 * 24 * 60 * 60
# Lifecycle breadcrumb trail cap per record (oldest entries dropped first).
MAX_LIFECYCLE_EVENTS = 50
# Caller-stack frames captured for cancel attribution (innermost frames).
MAX_ATTRIBUTION_FRAMES = 12

class RegistryError(RuntimeError):
    """Registry cannot be safely read or mutated."""


def _capture_caller_stack(*, skip_frames: int = 3) -> list[str]:
    """Innermost caller frames as compact ``file:line:func`` strings.

    ``skip_frames`` drops this helper, ``_stamp_cancel_attribution``, and
    the store-internal cancel writer (``cancel_matching`` /
    ``transition_owned``) so the trail starts at the runtime code that
    requested the cancellation — the WHO for Phase-0 cancel forensics.
    """
    try:
        frames = traceback.extract_stack()
        if skip_frames:
            frames = frames[:-skip_frames]
        return [
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
            for frame in frames[-MAX_ATTRIBUTION_FRAMES:]
        ]
    except Exception:  # pragma: no cover - forensics must never break a cancel
        return []


def append_lifecycle_event(
    record: dict[str, Any],
    event: str,
    detail: str = "",
    *,
    now: float | None = None,
) -> None:
    """Append a breadcrumb to the record's additive ``lifecycle`` trail.

    Purely observational (Phase-0 cancel forensics): older readers ignore the
    key, and records written before the upgrade simply have no trail.
    """
    try:
        trail = record.setdefault("lifecycle", [])
        if not isinstance(trail, list):  # defensive against hand-edited state
            return
        entry: dict[str, Any] = {"event": str(event), "ts": now if now is not None else time.time()}
        if detail:
            entry["detail"] = str(detail)[:240]
        trail.append(entry)
        if len(trail) > MAX_LIFECYCLE_EVENTS:
            del trail[: len(trail) - MAX_LIFECYCLE_EVENTS]
    except Exception:  # pragma: no cover - forensics must never break a write
        pass


def _stamp_cancel_attribution(
    record: dict[str, Any],
    *,
    now: float,
    reason: str,
    caller: str,
    via: str,
    selector: dict[str, Any] | None = None,
) -> None:
    """Stamp WHO/WHY/WHEN onto a record entering ``state=cancelled``.

    Additive schema (Phase-0 of the parent-interrupt spec): every writer of
    ``state=cancelled`` records which terminal path asked for the cancel, the
    caller stack that reached it, and the wall-clock time — so the next
    "silent death" in ``async-delegations.json`` is attributable instead of
    mysterious. Never overwrites an earlier stamp (first cancel wins).
    """
    try:
        if isinstance(record.get("cancel_attribution"), dict):
            return
        attribution: dict[str, Any] = {
            "reason": str(reason or "unspecified"),
            "caller": str(caller or "unknown"),
            "via": str(via),
            "cancelled_at": now,
            "stack": _capture_caller_stack(skip_frames=3),
        }
        if selector:
            attribution["selector"] = {
                k: v for k, v in selector.items() if v not in ("", None, False)
            }
        record["cancel_attribution"] = attribution
        append_lifecycle_event(
            record,
            "cancelled",
            f"by={attribution['caller']} reason={attribution['reason']} via={via}",
            now=now,
        )
        logger.info(
            "async_delegation_cancelled delegation_id=%s reason=%s caller=%s via=%s",
            record.get("delegation_id"),
            attribution["reason"],
            attribution["caller"],
            via,
        )
    except Exception:  # pragma: no cover - forensics must never break a cancel
        pass


def registry_path(profile_home: Path | None = None) -> Path:
    home = Path(profile_home) if profile_home is not None else get_hermes_home()
    return home / "state" / "async-delegations.json"


def lock_path(profile_home: Path | None = None) -> Path:
    home = Path(profile_home) if profile_home is not None else get_hermes_home()
    return home / "state" / "async-delegations.lock"


def empty_registry() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": 0.0, "records": {}}


def _record_checksum(record: dict[str, Any]) -> str:
    canonical = dict(record)
    canonical.pop("integrity", None)
    raw = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path, *, allow_invalid_records: bool = False) -> dict[str, Any]:
    if not path.exists():
        return empty_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("async_delegation_registry_invalid path=%s reason=%s", path, exc)
        raise RegistryError(f"invalid async-delegation registry: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        reason = f"unsupported schema_version={data.get('schema_version') if isinstance(data, dict) else None}"
        logger.error("async_delegation_registry_invalid path=%s reason=%s", path, reason)
        raise RegistryError(reason)
    if not isinstance(data.get("records"), dict):
        raise RegistryError("registry records must be an object")
    invalid_record_ids: list[str] = []
    for delegation_id, record in data["records"].items():
        if not isinstance(record, dict):
            if not allow_invalid_records:
                raise RegistryError(f"record {delegation_id} must be an object")
            invalid_record_ids.append(str(delegation_id))
            logger.error(
                "async_delegation_registry_invalid delegation_id=%s reason=record_not_object",
                delegation_id,
            )
            continue
        expected = record.get("integrity")
        if not expected or expected != _record_checksum(record):
            if not allow_invalid_records:
                raise RegistryError(f"record {delegation_id} failed integrity validation")
            invalid_record_ids.append(str(delegation_id))
            logger.error(
                "async_delegation_registry_invalid delegation_id=%s reason=integrity",
                delegation_id,
            )
    if invalid_record_ids:
        data["_invalid_record_ids"] = invalid_record_ids
    return data


def read_registry(profile_home: Path | None = None) -> dict[str, Any]:
    return _load(registry_path(profile_home))


def _prepare_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass


def _write(path: Path, registry: dict[str, Any]) -> None:
    _prepare_directory(path)
    registry["schema_version"] = SCHEMA_VERSION
    registry["updated_at"] = time.time()
    invalid_record_ids = set(registry.pop("_invalid_record_ids", []))
    for delegation_id, record in registry.get("records", {}).items():
        if str(delegation_id) in invalid_record_ids:
            continue
        if isinstance(record, dict):
            record["integrity"] = _record_checksum(record)
    atomic_json_write(path, registry, mode=0o600)


@contextmanager
def locked_registry(
    profile_home: Path | None = None,
    *,
    timeout: float = 5.0,
    write: bool = True,
) -> Iterator[dict[str, Any]]:
    """Lock, load, and optionally atomically persist one profile registry."""
    path = registry_path(profile_home)
    lpath = lock_path(profile_home)
    _prepare_directory(path)
    deadline = time.monotonic() + max(0.0, timeout)
    handle = lpath.open("a+b")
    acquired = False
    try:
        try:
            import fcntl  # type: ignore
        except ImportError:  # pragma: no cover - Windows
            import msvcrt  # type: ignore

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while not acquired:
                try:
                    handle.seek(0)
                    getattr(msvcrt, "locking")(
                        handle.fileno(), getattr(msvcrt, "LK_NBLCK"), 1
                    )
                    acquired = True
                    break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring {lpath}")
                time.sleep(0.01)
        else:
            while not acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out acquiring {lpath}")
                    time.sleep(0.01)
        registry = _load(path, allow_invalid_records=True)
        yield registry
        if write:
            _write(path, registry)
    finally:
        if acquired:
            try:
                import fcntl  # type: ignore
            except ImportError:  # pragma: no cover - Windows
                import msvcrt  # type: ignore

                try:
                    handle.seek(0)
                    getattr(msvcrt, "locking")(
                        handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                    )
                except OSError:
                    pass
            else:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        handle.close()


def write_for_tests(registry: dict[str, Any], profile_home: Path | None = None) -> None:
    with locked_registry(profile_home) as current:
        current.clear()
        current.update(copy.deepcopy(registry))


def _terminal_pending(record: dict[str, Any]) -> bool:
    return any(
        event.get("state") == "pending"
        for event in record.get("outbox", [])
        if isinstance(event, dict)
    )


def _prune(registry: dict[str, Any], now: float) -> None:
    records = registry["records"]
    invalid_record_ids = set(registry.get("_invalid_record_ids", []))
    removable: list[tuple[float, str]] = []
    for delegation_id, record in records.items():
        if str(delegation_id) in invalid_record_ids:
            continue
        if not isinstance(record, dict):
            continue
        state = record.get("state")
        if state not in {"done", "failed", "cancelled"}:
            continue
        updated = float(record.get("updated_at") or record.get("created_at") or 0.0)
        age = now - updated
        if (not _terminal_pending(record) and age >= TERMINAL_RETENTION_SECONDS) or age >= ABSOLUTE_RETENTION_SECONDS:
            removable.append((updated, delegation_id))
    for _, delegation_id in sorted(removable):
        records.pop(delegation_id, None)
    if len(records) <= MAX_REGISTRY_RECORDS:
        return
    eligible = sorted(
        (float(r.get("updated_at") or 0.0), rid)
        for rid, r in records.items()
        if str(rid) not in invalid_record_ids
        and isinstance(r, dict)
        and r.get("state") in {"done", "failed", "cancelled"}
        and not _terminal_pending(r)
    )
    for _, delegation_id in eligible:
        if len(records) <= MAX_REGISTRY_RECORDS:
            break
        records.pop(delegation_id, None)


def _attempt_id(delegation_id: str, generation: int) -> str:
    import uuid

    return f"{delegation_id}:g{generation}:{uuid.uuid4().hex[:8]}"


_SECRET_FIELD_PARTS = (
    "api_key",
    "apikey",
    "password",
    "secret",
    "authorization",
    "access_token",
    "refresh_token",
)


def _assert_nonsecret_settings(durable_spec: dict[str, Any]) -> None:
    """Fail closed if execution metadata contains credential-shaped fields."""
    execution = durable_spec.get("execution") or {}

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).lower().replace("-", "_")
                if any(part in normalized for part in _SECRET_FIELD_PARTS):
                    raise RegistryError(
                        f"secret-shaped field cannot be persisted: {path}.{key}"
                    )
                _walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _walk(child, f"{path}[{index}]")

    _walk(execution, "execution")
    args = [str(item).lower() for item in execution.get("acp_args") or []]
    if any(
        item.startswith(("--api-key", "--token", "--password"))
        or "authorization=" in item
        for item in args
    ):
        raise RegistryError("credential-shaped ACP argument cannot be persisted")
    base_url = str(execution.get("base_url") or "")
    if "://" in base_url:
        from urllib.parse import parse_qsl, urlsplit

        parsed = urlsplit(base_url)
        sensitive_query = any(
            any(
                part in key.lower().replace("-", "_")
                for part in _SECRET_FIELD_PARTS
            )
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        )
        if parsed.username or parsed.password or sensitive_query:
            raise RegistryError("credential-bearing base_url cannot be persisted")


def persist_dispatch(
    *,
    delegation_id: str,
    durable_spec: dict[str, Any],
    boot_id: str,
    dispatched_at: float,
    profile_home: Path | None = None,
    max_records: int | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Persist generation-zero intent. Returns (record, rejection_reason)."""
    now = time.time()
    record_limit = MAX_REGISTRY_RECORDS if max_records is None else max_records
    path = registry_path(profile_home)
    try:
        _assert_nonsecret_settings(durable_spec)
        with locked_registry(profile_home) as registry:
            _prune(registry, now)
            if len(registry["records"]) >= record_limit:
                logger.error(
                    "async_delegation_sync_fallback reason=registry_cap cap=%d path=%s",
                    record_limit,
                    path,
                )
                return None, "registry_cap"
            record = {
                "delegation_id": delegation_id,
                "state": "running",
                "created_at": dispatched_at,
                "updated_at": now,
                "profile": str(durable_spec.get("profile") or "default"),
                "source": copy.deepcopy(durable_spec.get("source")),
                "execution": copy.deepcopy(durable_spec.get("execution")),
                "route": copy.deepcopy(durable_spec.get("route")),
                "attempt": {
                    "attempt_id": _attempt_id(delegation_id, 0),
                    "generation": 0,
                    "redispatch_count": 0,
                    "owner_boot_id": boot_id,
                    "started_at": dispatched_at,
                    "submitted_at": None,
                    "last_interrupted_at": None,
                    "last_error": None,
                },
                "terminal": None,
                "outbox": [],
            }
            append_lifecycle_event(
                record,
                "spawned",
                f"boot={boot_id} profile={record['profile']}",
                now=dispatched_at,
            )
            registry["records"][delegation_id] = record
        logger.info(
            "async_delegation_persisted delegation_id=%s attempt_id=%s owner_boot_id=%s profile=%s",
            delegation_id,
            record["attempt"]["attempt_id"],
            boot_id,
            record["profile"],
        )
        return record, None
    except Exception as exc:
        logger.error(
            "async_delegation_persist_failed delegation_id=%s path=%s reason=%s",
            delegation_id,
            path,
            exc,
        )
        return None, "registry_error"


def mark_submitted(
    delegation_id: str,
    attempt_id: str,
    *,
    profile_home: Path | None = None,
) -> bool:
    with locked_registry(profile_home) as registry:
        if delegation_id in set(registry.get("_invalid_record_ids", [])):
            return False
        record = registry["records"].get(delegation_id)
        if not isinstance(record, dict) or record.get("state") != "running":
            return False
        if record.get("attempt", {}).get("attempt_id") != attempt_id:
            return False
        now = time.time()
        record["attempt"]["submitted_at"] = now
        record["updated_at"] = now
        append_lifecycle_event(record, "running", f"attempt={attempt_id}", now=now)
        return True


def _terminal_payload(record: dict[str, Any], result: dict[str, Any], status: str) -> dict[str, Any]:
    route = record.get("route") or {}
    source = record.get("source") or {}
    execution = record.get("execution") or {}
    tasks = source.get("tasks") or []
    goals = [str(task.get("goal") or "") for task in tasks if isinstance(task, dict)]
    dispatched_at = record.get("created_at") or time.time()
    completed_at = time.time()
    payload = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        "attempt_id": record.get("attempt", {}).get("attempt_id"),
        "attempt_generation": record.get("attempt", {}).get("generation"),
        "redispatch_count": record.get("attempt", {}).get("redispatch_count", 0),
        "session_key": route.get("session_key", ""),
        "origin_ui_session_id": route.get("origin_ui_session_id", ""),
        "parent_session_id": route.get("parent_session_id"),
        "platform": route.get("platform"),
        "chat_type": route.get("chat_type"),
        "chat_id": route.get("chat_id"),
        "thread_id": route.get("thread_id"),
        "user_id": route.get("user_id"),
        "user_name": route.get("user_name"),
        "profile": route.get("profile") or record.get("profile"),
        "goal": goals[0] if len(goals) == 1 else f"{len(goals)} parallel subagents",
        "goals": goals,
        "context": source.get("shared_context"),
        "toolsets": execution.get("toolsets"),
        "role": (tasks[0].get("role") if tasks else "leaf"),
        "model": result.get("model") or execution.get("model"),
        "status": status,
        "summary": result.get("summary"),
        "error": result.get("error"),
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": result.get("duration_seconds", round(completed_at - dispatched_at, 2)),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "exit_reason": result.get("exit_reason"),
    }
    if source.get("kind") == "batch" or len(goals) > 1 or "results" in result:
        payload.update({
            "is_batch": True,
            "results": result.get("results") or [],
            "total_duration_seconds": result.get("total_duration_seconds"),
        })
    return payload


def append_terminal(
    delegation_id: str,
    attempt_id: str,
    result: dict[str, Any],
    status: str,
    *,
    profile_home: Path | None = None,
) -> dict[str, Any] | None:
    with locked_registry(profile_home) as registry:
        if delegation_id in set(registry.get("_invalid_record_ids", [])):
            return None
        record = registry["records"].get(delegation_id)
        if not isinstance(record, dict):
            return None
        if record.get("state") != "running" or record.get("attempt", {}).get("attempt_id") != attempt_id:
            logger.info(
                "async_delegation_stale_finalize delegation_id=%s attempt_id=%s current_attempt_id=%s state=%s",
                delegation_id,
                attempt_id,
                record.get("attempt", {}).get("attempt_id"),
                record.get("state"),
            )
            return None
        terminal_state = "done" if status in {"completed", "success"} else "failed"
        payload = _terminal_payload(record, result, status)
        event_id = f"{delegation_id}:terminal:g{record['attempt']['generation']}"
        payload["event_id"] = event_id
        record["state"] = terminal_state
        record["terminal"] = {
            "status": status,
            "completed_at": payload["completed_at"],
            "error": result.get("error"),
        }
        record["updated_at"] = payload["completed_at"]
        append_lifecycle_event(
            record,
            terminal_state,
            f"status={status} attempt={attempt_id}",
            now=payload["completed_at"],
        )
        if not any(event.get("event_id") == event_id for event in record.get("outbox", [])):
            record.setdefault("outbox", []).append({
                "event_id": event_id,
                "type": "async_delegation",
                "state": "pending",
                "queued_boot_id": record["attempt"].get("owner_boot_id"),
                "created_at": payload["completed_at"],
                "delivered_at": None,
                "drop_reason": None,
                "payload": payload,
            })
        return payload


def fail_attempt(
    delegation_id: str,
    attempt_id: str,
    error: str,
    *,
    profile_home: Path | None = None,
) -> dict[str, Any] | None:
    return append_terminal(
        delegation_id,
        attempt_id,
        {"status": "error", "error": error, "summary": None},
        "error",
        profile_home=profile_home,
    )


def transition_owned(
    delegation_ids: list[str],
    state: str,
    *,
    profile_home: Path | None = None,
    timeout: float = 5.0,
    reason: str = "unspecified",
    caller: str = "",
) -> bool:
    try:
        with locked_registry(profile_home, timeout=timeout) as registry:
            now = time.time()
            invalid_record_ids = set(registry.get("_invalid_record_ids", []))
            for delegation_id in delegation_ids:
                if delegation_id in invalid_record_ids:
                    continue
                record = registry["records"].get(delegation_id)
                if isinstance(record, dict) and record.get("state") == "running":
                    record["state"] = state
                    record["updated_at"] = now
                    record["attempt"]["last_interrupted_at"] = now
                    if state == "cancelled":
                        _stamp_cancel_attribution(
                            record,
                            now=now,
                            reason=reason,
                            caller=caller,
                            via="transition_owned",
                            selector={"delegation_ids": list(delegation_ids)},
                        )
                    else:
                        append_lifecycle_event(
                            record, state, f"by={caller or 'unknown'} reason={reason}", now=now
                        )
                    if state == "cancelled":
                        for event in record.get("outbox", []):
                            if (
                                isinstance(event, dict)
                                and event.get("state") == "pending"
                                and event.get("type") == "async_delegation_restarted"
                            ):
                                event["state"] = "dropped"
                                event["drop_reason"] = "cancelled"
                                event["delivered_at"] = now
            return True
    except (TimeoutError, RegistryError) as exc:
        logger.warning(
            "async_delegation_shutdown_persist_skipped state=%s count=%d reason=%s; "
            "running+dead-boot recovery is equivalent",
            state,
            len(delegation_ids),
            exc,
        )
        return False


def cancel_matching(
    *,
    session_key: str = "",
    parent_session_id: str = "",
    origin_ui_session_id: str = "",
    all_active: bool = False,
    profile_home: Path | None = None,
    reason: str = "unspecified",
    caller: str = "",
) -> int:
    """Durably cancel active records, including resume-disabled records.

    ``reason``/``caller`` are Phase-0 cancel forensics: they are stamped into
    each cancelled record's additive ``cancel_attribution`` block (WHO/WHY/
    WHEN) and change no cancellation behavior.
    """
    if not registry_path(profile_home).exists():
        return 0
    count = 0
    with locked_registry(profile_home) as registry:
        now = time.time()
        invalid_record_ids = set(registry.get("_invalid_record_ids", []))
        for delegation_id, record in registry["records"].items():
            if str(delegation_id) in invalid_record_ids or not isinstance(record, dict):
                continue
            if record.get("state") not in {"running", "recoverable"}:
                continue
            route = record.get("route") or {}
            # Scoped-pair semantics mirror interrupt_for_session (Greptile P1
            # 2026-07-11): key+parent supplied together => a record recorded
            # under a DIFFERENT parent_session_id is a prior /new-reset
            # session's job and must not be durably cancelled by key alone.
            _scoped = bool(session_key and parent_session_id)
            _rec_parent = str(route.get("parent_session_id") or "")
            _key_ok = bool(
                session_key
                and str(route.get("session_key") or "") == session_key
                and not (_scoped and _rec_parent and _rec_parent != parent_session_id)
            )
            matches = all_active or (
                _key_ok
                or (
                    parent_session_id
                    and _rec_parent == parent_session_id
                )
                or (
                    origin_ui_session_id
                    and str(route.get("origin_ui_session_id") or "")
                    == origin_ui_session_id
                )
            )
            if not matches:
                continue
            record["state"] = "cancelled"
            record["updated_at"] = now
            _stamp_cancel_attribution(
                record,
                now=now,
                reason=reason,
                caller=caller,
                via="cancel_matching",
                selector={
                    "session_key": session_key,
                    "parent_session_id": parent_session_id,
                    "origin_ui_session_id": origin_ui_session_id,
                    "all_active": all_active,
                },
            )
            attempt = record.get("attempt")
            if isinstance(attempt, dict):
                attempt["last_interrupted_at"] = now
            for event in record.get("outbox", []):
                if (
                    isinstance(event, dict)
                    and event.get("state") == "pending"
                    and event.get("type") == "async_delegation_restarted"
                ):
                    event["state"] = "dropped"
                    event["drop_reason"] = "cancelled"
                    event["delivered_at"] = now
            count += 1
    return count


def is_boot_id_alive(boot_id: str) -> bool:
    """Use the gateway's canonical boot-id producer and liveness contract."""
    from gateway.status import is_boot_id_alive as _gateway_boot_is_alive

    return _gateway_boot_is_alive(boot_id)


def _restart_payload(record: dict[str, Any], interrupted_at: float, restarted_at: float) -> dict[str, Any]:
    route = record.get("route") or {}
    tasks = (record.get("source") or {}).get("tasks") or []
    goals = [str(task.get("goal") or "") for task in tasks if isinstance(task, dict)]
    return {
        "type": "async_delegation_restarted",
        "delegation_id": record["delegation_id"],
        "attempt_id": record["attempt"]["attempt_id"],
        "attempt_generation": record["attempt"]["generation"],
        "redispatch_count": record["attempt"]["redispatch_count"],
        "interrupted_at": interrupted_at,
        "restarted_at": restarted_at,
        "session_key": route.get("session_key", ""),
        "parent_session_id": route.get("parent_session_id"),
        "origin_ui_session_id": route.get("origin_ui_session_id", ""),
        "platform": route.get("platform"),
        "chat_type": route.get("chat_type"),
        "chat_id": route.get("chat_id"),
        "thread_id": route.get("thread_id"),
        "user_id": route.get("user_id"),
        "user_name": route.get("user_name"),
        "profile": route.get("profile") or record.get("profile"),
        "goal": goals[0] if len(goals) == 1 else f"{len(goals)} parallel subagents",
        "goals": goals,
        "is_batch": len(goals) > 1 or (record.get("source") or {}).get("kind") == "batch",
    }


def _fail_recovery_record(
    record: dict[str, Any], error: str, now: float, *, emit_event: bool
) -> None:
    record["state"] = "failed"
    record["updated_at"] = now
    record["terminal"] = {"status": "error", "error": error, "completed_at": now}
    if not emit_event:
        return
    payload = _terminal_payload(record, {"error": error}, "error")
    event_id = f"{record.get('delegation_id')}:terminal:{error}"
    payload["event_id"] = event_id
    if not any(event.get("event_id") == event_id for event in record.get("outbox", [])):
        record.setdefault("outbox", []).append({
            "event_id": event_id,
            "type": "async_delegation",
            "state": "pending",
            "queued_boot_id": None,
            "created_at": now,
            "delivered_at": None,
            "drop_reason": None,
            "payload": payload,
        })


def claim_recoveries(
    *,
    current_boot_id: str,
    resume_enabled: bool,
    profile_home: Path | None = None,
    owner_alive=is_boot_id_alive,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    summary = {"scanned": 0, "eligible": 0, "claimed": 0, "exhausted": 0, "failed_validation": 0}
    if not resume_enabled:
        return [], summary
    claimed: list[dict[str, Any]] = []
    now = time.time()
    with locked_registry(profile_home) as registry:
        invalid_record_ids = set(registry.get("_invalid_record_ids", []))
        for delegation_id, record in list(registry["records"].items()):
            summary["scanned"] += 1
            if str(delegation_id) in invalid_record_ids:
                summary["failed_validation"] += 1
                continue
            if not isinstance(record, dict):
                summary["failed_validation"] += 1
                continue
            state = record.get("state")
            if state not in {"running", "recoverable"}:
                continue
            attempt = record.get("attempt")
            source = record.get("source")
            execution = record.get("execution")
            route = record.get("route")
            if (
                not isinstance(attempt, dict)
                or not isinstance(source, dict)
                or not isinstance(execution, dict)
                or not isinstance(route, dict)
            ):
                summary["failed_validation"] += 1
                continue
            route_ready = all(
                str(route.get(key) or "").strip()
                for key in ("platform", "session_key", "parent_session_id")
            )
            tasks = source.get("tasks")
            tasks_valid = (
                isinstance(tasks, list)
                and bool(tasks)
                and all(
                    isinstance(task, dict)
                    and bool(str(task.get("goal") or "").strip())
                    for task in tasks
                )
            )
            profile_matches = str(
                route.get("profile") or record.get("profile") or ""
            ) == str(record.get("profile") or "")
            if not route_ready or not tasks_valid or not profile_matches:
                error = "unsupported_route" if not route_ready else "invalid_record"
                _fail_recovery_record(
                    record,
                    error,
                    now,
                    emit_event=bool(route.get("platform") and route.get("session_key")),
                )
                summary["failed_validation"] += 1
                continue
            owner = str(attempt.get("owner_boot_id") or "")
            if state == "running" and owner == current_boot_id:
                continue
            if state == "running" and owner_alive(owner):
                continue
            summary["eligible"] += 1
            if now - float(record.get("created_at") or now) > ACTIVE_STALE_SECONDS:
                payload = _terminal_payload(record, {"error": "stale_record"}, "error")
                event_id = f"{delegation_id}:terminal:stale"
                payload["event_id"] = event_id
                record["state"] = "failed"
                record["updated_at"] = now
                record["terminal"] = {"status": "error", "error": "stale_record", "completed_at": now}
                record.setdefault("outbox", []).append({
                    "event_id": event_id, "type": "async_delegation", "state": "pending",
                    "queued_boot_id": None, "created_at": now, "delivered_at": None,
                    "drop_reason": None, "payload": payload,
                })
                continue
            # RC-1: a dead boot's claim with no executor-submission telemetry did
            # not launch replacement work and therefore does not consume retry budget.
            redispatch_count = int(attempt.get("redispatch_count") or 0)
            if int(attempt.get("generation") or 0) > 0 and attempt.get("submitted_at") is None:
                redispatch_count = max(0, redispatch_count - 1)
            if redispatch_count >= MAX_REDISPATCH_ATTEMPTS:
                payload = _terminal_payload(record, {"error": "restart_attempts_exhausted"}, "error")
                event_id = f"{delegation_id}:terminal:exhausted"
                payload["event_id"] = event_id
                record["state"] = "failed"
                record["updated_at"] = now
                record["terminal"] = {
                    "status": "error", "error": "restart_attempts_exhausted", "completed_at": now,
                }
                if not any(event.get("event_id") == event_id for event in record.get("outbox", [])):
                    record.setdefault("outbox", []).append({
                        "event_id": event_id, "type": "async_delegation", "state": "pending",
                        "queued_boot_id": None, "created_at": now, "delivered_at": None,
                        "drop_reason": None, "payload": payload,
                    })
                summary["exhausted"] += 1
                logger.warning(
                    "async_delegation_retry_exhausted delegation_id=%s redispatch_count=%d",
                    delegation_id,
                    redispatch_count,
                )
                continue
            generation = int(attempt.get("generation") or 0) + 1
            interrupted_at = float(attempt.get("started_at") or record.get("updated_at") or now)
            for event in record.get("outbox", []):
                if (
                    isinstance(event, dict)
                    and event.get("type") == "async_delegation_restarted"
                    and event.get("state") == "pending"
                    and int((event.get("payload") or {}).get("attempt_generation") or -1) < generation
                ):
                    event["state"] = "dropped"
                    event["drop_reason"] = "superseded"
                    event["delivered_at"] = now
            attempt.update({
                "attempt_id": _attempt_id(delegation_id, generation),
                "generation": generation,
                "redispatch_count": redispatch_count + 1,
                "owner_boot_id": current_boot_id,
                "started_at": now,
                "submitted_at": None,
                "last_interrupted_at": interrupted_at,
                "last_error": None,
            })
            record["state"] = "running"
            record["updated_at"] = now
            payload = _restart_payload(record, interrupted_at, now)
            event_id = f"{delegation_id}:restart:g{generation}"
            payload["event_id"] = event_id
            record.setdefault("outbox", []).append({
                "event_id": event_id,
                "type": "async_delegation_restarted",
                "state": "pending",
                "queued_boot_id": None,
                "created_at": now,
                "delivered_at": None,
                "drop_reason": None,
                "payload": payload,
            })
            claimed.append(copy.deepcopy(record))
            summary["claimed"] += 1
            logger.info(
                "async_delegation_recovery_claimed delegation_id=%s attempt_id=%s owner_boot_id=%s redispatch_count=%d",
                delegation_id,
                attempt["attempt_id"],
                current_boot_id,
                attempt["redispatch_count"],
            )
    return claimed, summary


def enqueue_pending_outbox(
    *,
    current_boot_id: str,
    profile_home: Path | None = None,
) -> list[dict[str, Any]]:
    queued: list[dict[str, Any]] = []
    with locked_registry(profile_home) as registry:
        invalid_record_ids = set(registry.get("_invalid_record_ids", []))
        for delegation_id, record in registry["records"].items():
            if str(delegation_id) in invalid_record_ids:
                continue
            if not isinstance(record, dict):
                continue
            generation = int((record.get("attempt") or {}).get("generation") or 0)
            for event in record.get("outbox", []):
                if not isinstance(event, dict) or event.get("state") != "pending":
                    continue
                payload = event.get("payload") or {}
                if event.get("type") == "async_delegation_restarted" and int(payload.get("attempt_generation") or -1) < generation:
                    event["state"] = "dropped"
                    event["drop_reason"] = "superseded"
                    event["delivered_at"] = time.time()
                    continue
                if event.get("queued_boot_id") == current_boot_id:
                    continue
                event["queued_boot_id"] = current_boot_id
                queued.append(copy.deepcopy(payload))
                logger.info(
                    "async_delegation_outbox_queued delegation_id=%s event_id=%s current_boot_id=%s",
                    delegation_id,
                    event.get("event_id"),
                    current_boot_id,
                )
    return queued


def acknowledge(
    event_id: str,
    *,
    outcome: str,
    reason: str | None = None,
    profile_home: Path | None = None,
) -> bool:
    if outcome not in {"delivered", "dropped"}:
        raise ValueError("outcome must be delivered or dropped")
    with locked_registry(profile_home) as registry:
        invalid_record_ids = set(registry.get("_invalid_record_ids", []))
        for delegation_id, record in registry["records"].items():
            if str(delegation_id) in invalid_record_ids:
                continue
            if not isinstance(record, dict):
                continue
            for event in record.get("outbox", []):
                if isinstance(event, dict) and event.get("event_id") == event_id:
                    event["state"] = outcome
                    event["delivered_at"] = time.time()
                    event["drop_reason"] = reason if outcome == "dropped" else None
                    logger.info(
                        "async_delegation_outbox_%s event_id=%s reason=%s",
                        outcome,
                        event_id,
                        reason,
                    )
                    return True
    return False
