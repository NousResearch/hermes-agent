"""Regression coverage for process notifications that finish while their owning session is reaped."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from tui_gateway import server


def _session(key: str, *, finalized: bool = False) -> dict:
    return {
        "session_key": key,
        "history_lock": threading.RLock(),
        "running": False,
        "history": [],
        "agent": None,
        "profile_home": "",
        "_finalized": finalized,
    }


def _event(key: str, number: int = 0) -> dict:
    return {
        "type": "completion",
        "session_id": f"proc_{key}_{number}",
        "session_key": key,
        "task_id": f"session:{key}",
        "owner_task_id": key,
        "command": f"job-{number}",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": f"done-{number}",
    }


def _registry() -> SimpleNamespace:
    return SimpleNamespace(
        completion_queue=queue.Queue(),
        is_completion_consumed=lambda _sid: False,
        restore_completions=lambda: 0,
    )


@pytest.fixture(autouse=True)
def _isolated_notification_state(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_reaped_notification_backlog", [], raising=False)

    @contextlib.contextmanager
    def _db(_session):
        yield SimpleNamespace(resolve_resume_session_id=lambda key: key)

    monkeypatch.setattr(server, "_session_db", _db)
    yield


def _drain_from_foreign_session(monkeypatch, registry, event) -> None:
    foreign = _session("foreign-live")
    monkeypatch.setattr(server, "_sessions", {"foreign-sid": foreign})
    server._notif_handle_ready(
        "foreign-sid", foreign, [event], set(), registry, lambda _evt: "notification", None
    )


def _run_owner_poller(monkeypatch, registry, owner_key: str) -> list[dict]:
    owner = _session(owner_key)
    monkeypatch.setattr(server, "_sessions", {"owner-sid": owner})
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr("tools.async_delegation.maybe_sweep_orphaned_completions", lambda _queue: None)
    for name in (
        "_poll_bot_live_delivery_guarded",
        "_maybe_fire_tui_loop_tick",
        "_maybe_fire_tui_heartbeat_tick",
        "_notif_poll_kanban",
    ):
        monkeypatch.setattr(server, name, lambda *args, **kwargs: None)

    seen: list[dict] = []
    stop = threading.Event()

    def _handle(_sid, _session, events, _emitted, _registry, _fmt, _deferred, **_kwargs):
        seen.extend(events)
        if events:
            stop.set()

    monkeypatch.setattr(server, "_notif_handle_ready", _handle)
    worker = threading.Thread(
        target=server._notification_poller_scoped_loop,
        args=(stop, "owner-sid", owner),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=1.5)
    if worker.is_alive():
        stop.set()
        worker.join(timeout=2)
    return seen


def test_shutdown_drain_preserves_completion_for_recoverable_resume(monkeypatch):
    registry = _registry()
    event = _event("resume-me")
    closing = _session("resume-me", finalized=True)
    deferred: list = []

    server._notif_handle_ready(
        "old-sid", closing, [event], set(), registry, lambda _evt: "notification", deferred
    )

    assert deferred == []
    assert registry.completion_queue.empty()
    seen = _run_owner_poller(monkeypatch, registry, "resume-me")
    assert event in seen


def test_one_absent_owner_cannot_fill_the_mailbox_and_drop_another(monkeypatch):
    registry = _registry()

    for number in range(16):
        _drain_from_foreign_session(monkeypatch, registry, _event("noisy-owner", number))
    wanted = _event("quiet-owner", 99)
    _drain_from_foreign_session(monkeypatch, registry, wanted)

    assert registry.completion_queue.empty()
    seen = _run_owner_poller(monkeypatch, registry, "quiet-owner")
    assert wanted in seen


def test_foreign_session_cannot_claim_retained_notification():
    event = _event("owner")
    assert server._retain_reaped_notification(event)
    assert server._claim_reaped_notifications("other-sid", _session("other")) == []
    assert server._claim_reaped_notifications("owner-sid", _session("owner")) == [event]


def test_per_owner_pressure_evicts_only_that_owners_oldest(monkeypatch):
    monkeypatch.setattr(server, "_REAPED_NOTIFICATION_MAX", 3)
    monkeypatch.setattr(server, "_REAPED_NOTIFICATION_MAX_PER_OWNER", 1)

    quiet = _event("quiet", 1)
    first = _event("noisy", 1)
    newest = _event("noisy", 2)
    assert server._retain_reaped_notification(quiet)
    assert server._retain_reaped_notification(first)
    assert server._retain_reaped_notification(newest)

    assert server._claim_reaped_notifications("quiet-sid", _session("quiet")) == [quiet]
    assert server._claim_reaped_notifications("noisy-sid", _session("noisy")) == [newest]


def test_total_cap_admits_a_new_owner_by_evicting_oldest(monkeypatch):
    monkeypatch.setattr(server, "_REAPED_NOTIFICATION_MAX", 2)
    monkeypatch.setattr(server, "_REAPED_NOTIFICATION_MAX_PER_OWNER", 2)

    oldest = _event("a", 1)
    newer = _event("a", 2)
    incoming = _event("b", 1)
    assert server._retain_reaped_notification(oldest)
    assert server._retain_reaped_notification(newer)
    assert server._retain_reaped_notification(incoming)

    assert server._claim_reaped_notifications("a-sid", _session("a")) == [newer]
    assert server._claim_reaped_notifications("b-sid", _session("b")) == [incoming]


def test_expired_retention_is_not_delivered(monkeypatch):
    monkeypatch.setattr(server, "_REAPED_NOTIFICATION_TTL_SECONDS", 0.0)
    event = _event("owner")
    assert server._retain_reaped_notification(event)
    time.sleep(0)
    assert server._claim_reaped_notifications("owner-sid", _session("owner")) == []
