"""Behavior contract for stable, recoverable gateway Approval identities."""

from __future__ import annotations

import gc
import threading
import json
import math
import weakref


def _queue_entry(approval, session_key: str, **kwargs):
    entry = approval._ApprovalEntry({"command": "dangerous"}, **kwargs)
    approval._gateway_queues.setdefault(session_key, []).append(entry)
    return entry


class _ManualTimer:
    """Deterministic stand-in for the single tombstone cleanup timer."""

    created: list["_ManualTimer"] = []

    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.daemon = False
        self.started = False
        self.cancelled = False
        self.created.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        assert self.started
        assert not self.cancelled
        self.function(*self.args, **self.kwargs)


def _detach_tombstone_scheduler(approval):
    """Isolate the process-global scheduler and return state for restoration."""
    with approval._lock:
        timer = approval._gateway_tombstone_cleanup_timer
        if timer is not None:
            timer.cancel()
        state = (
            approval._gateway_tombstones,
            approval._gateway_tombstone_cleanup_generation,
        )
        approval._gateway_tombstones = {}
        approval._gateway_tombstone_cleanup_timer = None
        approval._gateway_tombstone_cleanup_deadline = None
        approval._gateway_tombstone_cleanup_generation += 1
    return state


def _restore_tombstone_scheduler(approval, state):
    with approval._lock:
        timer = approval._gateway_tombstone_cleanup_timer
        if timer is not None:
            timer.cancel()
        approval._gateway_tombstones = state[0]
        approval._gateway_tombstone_cleanup_timer = None
        approval._gateway_tombstone_cleanup_deadline = None
        approval._gateway_tombstone_cleanup_generation = state[1] + 1
        approval._schedule_gateway_tombstone_cleanup_locked()


def test_terminal_entry_is_released_by_scheduled_ttl_cleanup(monkeypatch):
    from tools import approval

    state = _detach_tombstone_scheduler(approval)
    now = [100.0]
    _ManualTimer.created = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(approval.threading, "Timer", _ManualTimer)
            patch.setattr(approval.time, "monotonic", lambda: now[0])
            patch.setattr(
                approval,
                "_GATEWAY_APPROVAL_TOMBSTONE_TTL_SECONDS",
                5,
            )
            entry = _queue_entry(
                approval,
                "scheduled-tombstone-cleanup",
                approval_id="a" * 32,
                timeout_seconds=30,
                now_monotonic=now[0],
            )
            entry_ref = weakref.ref(entry)

            outcome = approval.resolve_gateway_approval_by_id(
                "scheduled-tombstone-cleanup",
                entry.approval_id,
                "deny",
            )
            assert outcome["outcome"] == "resolved"
            assert len(_ManualTimer.created) == 1
            timer = _ManualTimer.created[0]
            assert timer.interval == 5
            assert timer.daemon is True

            del entry
            gc.collect()
            assert entry_ref() is not None

            now[0] = 105.0
            timer.fire()
            gc.collect()

            assert "scheduled-tombstone-cleanup" not in (
                approval._gateway_tombstones
            )
            assert entry_ref() is None
    finally:
        _restore_tombstone_scheduler(approval, state)


def test_unregister_and_clear_tombstones_share_scheduled_cleanup(monkeypatch):
    from tools import approval

    state = _detach_tombstone_scheduler(approval)
    now = [200.0]
    _ManualTimer.created = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(approval.threading, "Timer", _ManualTimer)
            patch.setattr(approval.time, "monotonic", lambda: now[0])
            patch.setattr(
                approval,
                "_GATEWAY_APPROVAL_TOMBSTONE_TTL_SECONDS",
                5,
            )

            stale_session = "scheduled-unregister-cleanup"
            approval.register_gateway_notify(stale_session, lambda _: None)
            stale = _queue_entry(
                approval,
                stale_session,
                approval_id="b" * 32,
                timeout_seconds=30,
                now_monotonic=now[0],
            )
            stale_ref = weakref.ref(stale)
            approval.unregister_gateway_notify(stale_session)

            cleared_session = "scheduled-clear-cleanup"
            cleared = _queue_entry(
                approval,
                cleared_session,
                approval_id="c" * 32,
                timeout_seconds=30,
                now_monotonic=now[0],
            )
            cleared_ref = weakref.ref(cleared)
            approval.clear_session(cleared_session)

            assert set(approval._gateway_tombstones) == {
                stale_session,
                cleared_session,
            }
            assert len(_ManualTimer.created) == 1
            timer = _ManualTimer.created[0]

            del stale, cleared
            gc.collect()
            assert stale_ref() is not None
            assert cleared_ref() is not None

            now[0] = 205.0
            timer.fire()
            gc.collect()

            assert approval._gateway_tombstones == {}
            assert stale_ref() is None
            assert cleared_ref() is None
    finally:
        _restore_tombstone_scheduler(approval, state)


def test_pending_approval_exposes_stable_sanitized_public_descriptor(monkeypatch):
    from tools import approval

    session_key = "mobile-approval-public-descriptor"
    fake_secret = "sk-proj-" + "X" * 40
    command = (
        "rm -rf /tmp/hermes-approval-preview && "
        f"export OPENAI_API_KEY={fake_secret}"
    )
    notified: list[dict] = []
    notification = threading.Event()
    result: dict = {}

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 30},
    )

    def notify(descriptor: dict) -> None:
        notified.append(descriptor)
        notification.set()

    approval.register_gateway_notify(session_key, notify)

    def run_guard() -> None:
        token = approval.set_current_session_key(session_key)
        try:
            result.update(approval.check_all_command_guards(command, "local"))
        finally:
            approval.reset_current_session_key(token)

    worker = threading.Thread(target=run_guard, daemon=True)
    worker.start()

    try:
        assert notification.wait(timeout=5), "Approval request was not emitted"
        assert len(notified) == 1

        descriptor = notified[0]
        assert descriptor["approval_id"]
        assert descriptor["state"] == "pending"
        assert descriptor["resolution"] is None
        assert descriptor["created_at"] < descriptor["expires_at"]
        assert fake_secret not in descriptor["command"]

        pending = approval.list_pending_gateway_approvals(session_key)
        assert pending == [descriptor]
    finally:
        approval.resolve_gateway_approval(
            session_key,
            "deny",
            resolve_all=True,
        )
        worker.join(timeout=5)
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)

    assert not worker.is_alive()
    assert result["approved"] is False


def test_targeted_resolution_is_out_of_order_and_duplicate_safe():
    from tools import approval

    session_key = "mobile-approval-targeted"
    first = _queue_entry(
        approval,
        session_key,
        approval_id="approval-first",
        timeout_seconds=30,
    )
    second = _queue_entry(
        approval,
        session_key,
        approval_id="approval-second",
        timeout_seconds=30,
    )

    try:
        outcome = approval.resolve_gateway_approval_by_id(
            session_key,
            second.approval_id,
            "deny",
            reason="not this one",
            resolution_metadata={"source": "mobile"},
        )

        assert outcome["outcome"] == "resolved"
        assert outcome["approval"]["approval_id"] == second.approval_id
        assert outcome["approval"]["state"] == "resolved"
        assert outcome["approval"]["resolution"]["choice"] == "deny"
        assert outcome["approval"]["resolution"]["reason"] == "not this one"
        assert outcome["approval"]["resolution"]["metadata"] == {
            "source": "mobile"
        }
        assert outcome["approval"]["resolution"]["resolved_at"] >= outcome[
            "approval"
        ]["created_at"]
        assert not first.event.is_set()
        assert second.event.is_set()
        assert [item["approval_id"] for item in approval.list_pending_gateway_approvals(session_key)] == [
            first.approval_id
        ]

        duplicate = approval.resolve_gateway_approval_by_id(
            session_key,
            second.approval_id,
            "once",
        )
        assert duplicate["outcome"] == "already_resolved"
        assert duplicate["approval"] == outcome["approval"]
        assert first.event.is_set() is False
    finally:
        approval.clear_session(session_key)


def test_targeted_resolution_notifies_once_with_terminal_descriptor():
    from tools import approval

    session_key = "mobile-approval-terminal-notify"
    terminal_descriptors: list[dict] = []
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-notify",
        timeout_seconds=30,
    )
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )

    try:
        outcome = approval.resolve_gateway_approval_by_id(
            session_key,
            entry.approval_id,
            "deny",
            reason="not now",
        )
        duplicate = approval.resolve_gateway_approval_by_id(
            session_key,
            entry.approval_id,
            "once",
        )

        assert outcome["outcome"] == "resolved"
        assert terminal_descriptors == [outcome["approval"]]
        assert duplicate["outcome"] == "already_resolved"
        assert terminal_descriptors == [outcome["approval"]]
    finally:
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_gateway_timeout_notifies_once_with_expired_descriptor(monkeypatch):
    from tools import approval

    session_key = "mobile-approval-terminal-timeout"
    requests: list[dict] = []
    terminal_descriptors: list[dict] = []
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 0},
    )
    approval.register_gateway_notify(session_key, requests.append)
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )

    token = approval.set_current_session_key(session_key)
    try:
        result = approval.check_all_command_guards(
            "rm -rf /tmp/hermes-approval-timeout",
            "local",
        )
        assert result["approved"] is False
        assert len(requests) == 1
        assert terminal_descriptors[0]["approval_id"] == requests[0]["approval_id"]
        assert terminal_descriptors[0]["state"] == "expired"
        assert terminal_descriptors[0]["resolution"]["metadata"] == {
            "source": "timeout"
        }

        late = approval.resolve_gateway_approval_by_id(
            session_key,
            requests[0]["approval_id"],
            "once",
        )
        assert late == {
            "outcome": "expired",
            "approval": terminal_descriptors[0],
        }
        assert len(terminal_descriptors) == 1
    finally:
        approval.reset_current_session_key(token)
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_unregister_notifies_stale_once_and_clears_both_callbacks():
    from tools import approval

    session_key = "mobile-approval-terminal-unregister"
    terminal_descriptors: list[dict] = []
    approval.register_gateway_notify(session_key, lambda _: None)
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )
    stale = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-stale",
        timeout_seconds=30,
    )

    try:
        approval.unregister_gateway_notify(session_key)

        assert len(terminal_descriptors) == 1
        assert terminal_descriptors[0]["approval_id"] == stale.approval_id
        assert terminal_descriptors[0]["state"] == "stale"
        assert terminal_descriptors[0]["resolution"]["metadata"] == {
            "source": "callback_unregistered"
        }

        late = approval.resolve_gateway_approval_by_id(
            session_key,
            stale.approval_id,
            "once",
        )
        assert late == {
            "outcome": "stale",
            "approval": terminal_descriptors[0],
        }

        after_unregister = _queue_entry(
            approval,
            session_key,
            approval_id="approval-after-unregister",
            timeout_seconds=30,
        )
        approval.resolve_gateway_approval_by_id(
            session_key,
            after_unregister.approval_id,
            "deny",
        )
        assert len(terminal_descriptors) == 1
    finally:
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_resolution_notification_failure_does_not_change_approval_outcome():
    from tools import approval

    session_key = "mobile-approval-terminal-callback-failure"
    callback_calls = 0

    def broken_callback(_: dict) -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("terminal transport unavailable")

    approval.register_gateway_resolution_notify(session_key, broken_callback)
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-callback-failure",
        timeout_seconds=30,
    )

    try:
        outcome = approval.resolve_gateway_approval_by_id(
            session_key,
            entry.approval_id,
            "once",
        )
        duplicate = approval.resolve_gateway_approval_by_id(
            session_key,
            entry.approval_id,
            "once",
        )

        assert outcome["outcome"] == "resolved"
        assert duplicate["outcome"] == "already_resolved"
        assert callback_calls == 1
    finally:
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_terminal_notification_happens_before_waiter_is_released():
    from tools import approval

    session_key = "mobile-approval-terminal-ordering"
    callback_started = threading.Event()
    release_callback = threading.Event()
    outcome: dict = {}
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-ordering",
        timeout_seconds=30,
    )

    def blocking_callback(_: dict) -> None:
        callback_started.set()
        assert release_callback.wait(timeout=5)

    approval.register_gateway_resolution_notify(session_key, blocking_callback)

    def resolve() -> None:
        outcome.update(
            approval.resolve_gateway_approval_by_id(
                session_key,
                entry.approval_id,
                "deny",
            )
        )

    resolver = threading.Thread(target=resolve, daemon=True)
    resolver.start()
    try:
        assert callback_started.wait(timeout=5)
        assert entry.event.is_set() is False
        assert resolver.is_alive()
    finally:
        release_callback.set()
        resolver.join(timeout=5)
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)

    assert resolver.is_alive() is False
    assert entry.event.is_set()
    assert outcome["outcome"] == "resolved"


def test_legacy_fifo_resolution_notifies_in_queue_order():
    from tools import approval

    session_key = "mobile-approval-terminal-fifo"
    terminal_descriptors: list[dict] = []
    approval.register_gateway_notify(session_key, lambda _: None)
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )
    first = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-fifo-first",
        timeout_seconds=30,
    )
    second = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-fifo-second",
        timeout_seconds=30,
    )

    try:
        resolved = approval.resolve_gateway_approval(
            session_key,
            "deny",
            resolve_all=True,
        )

        assert resolved == 2
        assert [
            descriptor["approval_id"] for descriptor in terminal_descriptors
        ] == [first.approval_id, second.approval_id]
        assert all(
            descriptor["state"] == "resolved"
            for descriptor in terminal_descriptors
        )
        assert all(
            descriptor["resolution"]["metadata"] == {
                "source": "legacy_fifo"
            }
            for descriptor in terminal_descriptors
        )
    finally:
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_pending_snapshot_notifies_when_it_expires_an_approval():
    from tools import approval

    session_key = "mobile-approval-terminal-snapshot-expiry"
    terminal_descriptors: list[dict] = []
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-snapshot-expiry",
        timeout_seconds=0,
    )

    try:
        assert approval.list_pending_gateway_approvals(session_key) == []
        assert len(terminal_descriptors) == 1
        assert terminal_descriptors[0]["approval_id"] == entry.approval_id
        assert terminal_descriptors[0]["state"] == "expired"

        assert approval.list_pending_gateway_approvals(session_key) == []
        assert len(terminal_descriptors) == 1
    finally:
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_failed_request_notification_emits_stale_terminal_descriptor(monkeypatch):
    from tools import approval

    session_key = "mobile-approval-terminal-request-failure"
    terminal_descriptors: list[dict] = []
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 30},
    )

    def broken_request_callback(_: dict) -> None:
        raise RuntimeError("request transport unavailable")

    approval.register_gateway_notify(session_key, broken_request_callback)
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )
    token = approval.set_current_session_key(session_key)

    try:
        result = approval.check_all_command_guards(
            "rm -rf /tmp/hermes-approval-request-failure",
            "local",
        )

        assert result["approved"] is False
        assert len(terminal_descriptors) == 1
        descriptor = terminal_descriptors[0]
        assert descriptor["state"] == "stale"
        assert descriptor["resolution"]["metadata"] == {
            "source": "notify_failed"
        }

        late = approval.resolve_gateway_approval_by_id(
            session_key,
            descriptor["approval_id"],
            "once",
        )
        assert late == {"outcome": "stale", "approval": descriptor}
        assert len(terminal_descriptors) == 1
    finally:
        approval.reset_current_session_key(token)
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_session_cleanup_notifies_resolved_denial_once():
    from tools import approval

    session_key = "mobile-approval-terminal-session-cleanup"
    terminal_descriptors: list[dict] = []
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-terminal-session-cleanup",
        timeout_seconds=30,
    )

    try:
        approval.clear_session(session_key)

        assert len(terminal_descriptors) == 1
        descriptor = terminal_descriptors[0]
        assert descriptor["approval_id"] == entry.approval_id
        assert descriptor["state"] == "resolved"
        assert descriptor["resolution"]["choice"] == "deny"
        assert descriptor["resolution"]["metadata"] == {
            "source": "session_cleanup"
        }

        late = approval.resolve_gateway_approval_by_id(
            session_key,
            entry.approval_id,
            "once",
        )
        assert late == {"outcome": "already_resolved", "approval": descriptor}
        assert len(terminal_descriptors) == 1
    finally:
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_legacy_direct_request_callback_still_emits_terminal_descriptor(
    monkeypatch,
):
    from tools import approval

    session_key = "mobile-approval-terminal-legacy-direct"
    terminal_descriptors: list[dict] = []
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 30},
    )

    def resolve_directly(_: dict) -> None:
        entry = approval._gateway_queues[session_key][0]
        entry.result = "once"
        entry.event.set()

    approval.register_gateway_notify(session_key, resolve_directly)
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )
    token = approval.set_current_session_key(session_key)

    try:
        result = approval.check_all_command_guards(
            "rm -rf /tmp/hermes-approval-legacy-direct",
            "local",
        )

        assert result["approved"] is True
        assert len(terminal_descriptors) == 1
        descriptor = terminal_descriptors[0]
        assert descriptor["state"] == "resolved"
        assert descriptor["resolution"]["choice"] == "once"
        assert descriptor["resolution"]["metadata"] == {
            "source": "legacy_direct_callback"
        }
    finally:
        approval.reset_current_session_key(token)
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_late_expired_and_stale_resolutions_are_deterministic():
    from tools import approval

    expired_session = "mobile-approval-expired"
    expired = _queue_entry(
        approval,
        expired_session,
        approval_id="approval-expired",
        timeout_seconds=0,
    )

    expired_outcome = approval.resolve_gateway_approval_by_id(
        expired_session,
        expired.approval_id,
        "once",
    )
    assert expired_outcome["outcome"] == "expired"
    assert expired_outcome["approval"]["state"] == "expired"
    assert expired_outcome["approval"]["resolution"]["choice"] is None
    assert expired.event.is_set()

    stale_session = "mobile-approval-stale"
    approval.register_gateway_notify(stale_session, lambda _: None)
    stale = _queue_entry(
        approval,
        stale_session,
        approval_id="approval-stale",
        timeout_seconds=30,
    )
    approval.unregister_gateway_notify(stale_session)

    stale_outcome = approval.resolve_gateway_approval_by_id(
        stale_session,
        stale.approval_id,
        "once",
    )
    assert stale_outcome["outcome"] == "stale"
    assert stale_outcome["approval"]["state"] == "stale"
    assert stale_outcome["approval"]["resolution"]["choice"] is None
    assert stale.event.is_set()

    missing = approval.resolve_gateway_approval_by_id(
        stale_session,
        "a" * 32,
        "once",
    )
    assert missing == {
        "outcome": "not_found",
        "approval_id": "a" * 32,
    }


def test_public_descriptor_redacts_all_payload_and_resolution_metadata_paths():
    from tools import approval

    session_key = "mobile-approval-redaction"
    secret = "ghp_" + "A" * 36
    entry = approval._ApprovalEntry(
        {
            "command": f"curl -H 'Authorization: Bearer {secret}' example.test",
            "description": f"nested {secret}",
            "metadata": {
                "list": [secret, {"deep": secret}],
                "tuple": (secret,),
            },
        },
        approval_id="approval-redacted",
        timeout_seconds=30,
    )
    approval._gateway_queues.setdefault(session_key, []).append(entry)

    pending = approval.list_pending_gateway_approvals(session_key)[0]
    assert secret not in repr(pending)

    resolved = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "deny",
        reason=f"reason {secret}",
        resolution_metadata={"nested": [secret, {"deep": secret}]},
    )
    assert secret not in repr(resolved)

    # Returned descriptors are snapshots, not mutable views into core state.
    resolved["approval"]["command"] = secret
    duplicate = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "deny",
    )
    assert secret not in repr(duplicate)

    missing = approval.resolve_gateway_approval_by_id(
        session_key,
        f"missing-{secret}",
        "deny",
    )
    assert secret not in repr(missing)


def test_concurrent_targeted_responses_resolve_exactly_once():
    from tools import approval

    session_key = "mobile-approval-concurrent"
    terminal_descriptors: list[dict] = []
    approval.register_gateway_resolution_notify(
        session_key,
        terminal_descriptors.append,
    )
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-concurrent",
        timeout_seconds=30,
    )
    barrier = threading.Barrier(3)
    outcomes: list[dict] = []

    def resolve(choice: str) -> None:
        barrier.wait()
        outcomes.append(
            approval.resolve_gateway_approval_by_id(
                session_key,
                entry.approval_id,
                choice,
            )
        )

    workers = [
        threading.Thread(target=resolve, args=(choice,))
        for choice in ("once", "deny")
    ]
    try:
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(timeout=5)

        assert all(not worker.is_alive() for worker in workers)
        assert sorted(item["outcome"] for item in outcomes) == [
            "already_resolved",
            "resolved",
        ]
        assert outcomes[0]["approval"] == outcomes[1]["approval"]
        assert terminal_descriptors == [outcomes[0]["approval"]]
    finally:
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)


def test_tombstones_are_short_lived(monkeypatch):
    from tools import approval

    session_key = "mobile-approval-tombstone-ttl"
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-short-lived",
        timeout_seconds=30,
    )
    monkeypatch.setattr(
        approval,
        "_GATEWAY_APPROVAL_TOMBSTONE_TTL_SECONDS",
        0,
    )

    first = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "once",
    )
    after_ttl = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "once",
    )

    assert first["outcome"] == "resolved"
    assert after_ttl == {
        "outcome": "not_found",
        "approval_id": "invalid-approval-id",
    }


def test_invalid_choice_fails_without_consuming_pending_approval():
    from tools import approval

    session_key = "mobile-approval-invalid-choice"
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="b" * 32,
        timeout_seconds=30,
    )

    outcome = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "approve-ish",
    )

    assert outcome == {
        "outcome": "invalid_choice",
        "approval_id": "b" * 32,
    }
    assert entry.event.is_set() is False
    assert approval.resolve_gateway_approval(session_key, "approve-ish") == 0
    assert approval.list_pending_gateway_approvals(session_key)[0][
        "approval_id"
    ] == entry.approval_id
    approval.clear_session(session_key)


def test_non_finite_timeouts_fall_back_to_finite_public_expiry():
    from tools import approval

    for timeout in (math.nan, math.inf, -math.inf):
        entry = approval._ApprovalEntry(
            {"command": "dangerous"},
            timeout_seconds=timeout,
        )
        descriptor = entry.public_descriptor()
        assert descriptor["expires_at"] - descriptor["created_at"] == 300
        json.dumps(descriptor, allow_nan=False)
