"""Regression coverage for durable process-notification reconciliation."""

import time
from unittest.mock import patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


def _registry(tmp_path):
    state_path = tmp_path / "process_notification_state.sqlite3"
    with patch("tools.process_registry.NOTIFICATION_STATE_PATH", state_path):
        return ProcessRegistry()


def _session(
    sid: str,
    *,
    started_at: float,
    command: str = "python -m http.server 4312",
    exited: bool = False,
    termination_source: str = "",
    completion_reason: str = "exited",
):
    return ProcessSession(
        id=sid,
        command=command,
        task_id="evaluation",
        session_key="desktop-session",
        started_at=started_at,
        exited=exited,
        exit_code=-15 if exited else None,
        termination_source=termination_source,
        completion_reason=completion_reason,
        output_buffer="Local: http://127.0.0.1:4312/\n",
    )


def _watch_event(session, *, sequence=1, output="Local: http://127.0.0.1:4312/"):
    return {
        "type": "watch_match",
        "event_id": (
            f"watch:{session.id}:{int(session.started_at * 1_000_000)}:{sequence}"
        ),
        "session_id": session.id,
        "started_at": session.started_at,
        "session_key": session.session_key,
        "command": session.command,
        "pattern": "Local:",
        "output": output,
    }


def _register(registry, session):
    target = registry._finished if session.exited else registry._running
    target[session.id] = session


@pytest.mark.parametrize(
    "process_id",
    ["proc_6ea6744be486", "proc_fb4cddce1ea3", "proc_b9a4d9c996c4"],
)
def test_intentional_cleanup_with_closed_port_suppresses_late_readiness(
    tmp_path, monkeypatch, process_id
):
    registry = _registry(tmp_path)
    session = _session(
        process_id,
        started_at=1000.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    _register(registry, session)
    event = _watch_event(session)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])

    registry.observe_notification(event)
    registry.reconcile_process_notifications(session)

    assert registry.is_notification_actionable(event) is False


def test_later_polling_pass_suppresses_same_terminal_event(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    session = _session(
        "proc_poll_later",
        started_at=1001.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    _register(registry, session)
    event = _watch_event(session)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])

    registry.observe_notification(event)
    registry.reconcile_process_notifications(session)

    assert not registry.is_notification_actionable(dict(event))
    assert not registry.is_notification_actionable(dict(event))


def test_event_queued_before_cleanup_is_reconciled_before_delivery(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    session = _session("proc_delayed", started_at=1002.0)
    _register(registry, session)
    event = _watch_event(session)
    registry.observe_notification(event)
    assert registry.is_notification_actionable(event)

    registry._running.pop(session.id)
    session.exited = True
    session.exit_code = -15
    session.completion_reason = "killed"
    session.termination_source = "process.kill"
    registry._finished[session.id] = session
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])
    registry.reconcile_process_notifications(session)

    assert not registry.is_notification_actionable(event)


def test_delivered_event_dedupes_across_independent_registry_instances(tmp_path):
    registry = _registry(tmp_path)
    session = _session("proc_repeat", started_at=1003.0)
    _register(registry, session)
    event = _watch_event(session)
    registry.observe_notification(event)
    assert registry.is_notification_actionable(event)
    registry.mark_notification_delivered(event)

    restarted = _registry(tmp_path)
    assert not restarted.is_notification_actionable(dict(event))


def test_unexpected_process_death_during_active_run_still_alerts(tmp_path):
    registry = _registry(tmp_path)
    session = _session(
        "proc_unexpected",
        started_at=1004.0,
        exited=True,
        termination_source="",
        completion_reason="exited",
    )
    session.exit_code = 1
    _register(registry, session)
    event = _watch_event(session)
    registry.observe_notification(event)

    assert registry.is_notification_actionable(event)


def test_intentional_kill_with_open_readiness_port_alerts_cleanup_defect(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    session = _session(
        "proc_leaked_port",
        started_at=1005.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    _register(registry, session)
    event = _watch_event(session)
    registry.observe_notification(event)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [4312])

    assert registry.is_notification_actionable(event)
    assert event["lifecycle_alert"] == "post_cleanup_listener"
    assert event["cleanup_open_ports"] == [4312]

    from tools.process_registry import format_process_notification

    text = format_process_notification(event)
    assert text is not None
    assert "remain open or were rebound" in text
    assert "4312" in text


def test_new_run_of_same_application_is_independently_eligible(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    old = _session(
        "proc_old_run",
        started_at=1006.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    _register(registry, old)
    old_event = _watch_event(old)
    registry.observe_notification(old_event)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])
    registry.reconcile_process_notifications(old)
    assert not registry.is_notification_actionable(old_event)

    new = _session("proc_new_run", started_at=1007.0)
    _register(registry, new)
    new_event = _watch_event(new)
    registry.observe_notification(new_event)
    assert registry.is_notification_actionable(new_event)


def test_different_applications_do_not_dedupe_each_other(tmp_path):
    registry = _registry(tmp_path)
    first = _session("proc_app_a", started_at=1008.0, command="app-a --port 4312")
    second = _session("proc_app_b", started_at=1008.0, command="app-b --port 4312")
    _register(registry, first)
    _register(registry, second)
    first_event = _watch_event(first)
    second_event = _watch_event(second)
    registry.observe_notification(first_event)
    registry.observe_notification(second_event)
    registry.mark_notification_delivered(first_event)

    assert not registry.is_notification_actionable(first_event)
    assert registry.is_notification_actionable(second_event)


def test_terminal_run_state_survives_registry_reload(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    session = _session(
        "proc_persisted_terminal",
        started_at=1009.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    _register(registry, session)
    event = _watch_event(session)
    registry.observe_notification(event)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])
    registry.reconcile_process_notifications(session)

    restarted = _registry(tmp_path)
    assert not restarted.is_notification_actionable(dict(event))


def test_reused_process_id_with_new_spawn_epoch_is_not_suppressed(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    old = _session(
        "proc_reused",
        started_at=1010.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    _register(registry, old)
    old_event = _watch_event(old)
    registry.observe_notification(old_event)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])
    registry.reconcile_process_notifications(old)

    replacement = _session("proc_reused", started_at=1011.0)
    registry._finished.pop(old.id)
    _register(registry, replacement)
    replacement_event = _watch_event(replacement)
    registry.observe_notification(replacement_event)

    assert registry.is_notification_actionable(replacement_event)


def test_intentional_cleanup_does_not_suppress_requested_completion_summary(
    tmp_path, monkeypatch
):
    registry = _registry(tmp_path)
    session = _session(
        "proc_bulk_cleanup",
        started_at=1012.0,
        exited=True,
        termination_source="kill_all",
        completion_reason="killed",
    )
    _register(registry, session)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])
    registry.reconcile_process_notifications(session)
    completion = {
        "type": "completion",
        "session_id": session.id,
        "started_at": session.started_at,
        "command": session.command,
        "exit_code": -15,
        "completion_reason": "killed",
        "termination_source": "kill_all",
        "output": session.output_buffer,
    }
    registry.observe_notification(completion)

    assert registry.is_notification_actionable(completion)


def test_remote_namespace_cleanup_is_not_certified_by_host_port_probe(
    tmp_path, monkeypatch
):
    registry = _registry(tmp_path)
    session = _session(
        "proc_remote_cleanup",
        started_at=1013.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    session.pid_scope = "sandbox"
    _register(registry, session)
    event = _watch_event(session)
    registry.observe_notification(event)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [4312])

    assert registry.reconcile_process_notifications(session) is False
    assert registry.is_notification_actionable(event)
    assert "lifecycle_alert" not in event


def test_terminal_run_does_not_suppress_non_readiness_watch_event(
    tmp_path, monkeypatch
):
    registry = _registry(tmp_path)
    session = _session(
        "proc_mixed_watch_events",
        started_at=1014.0,
        exited=True,
        termination_source="process.kill",
        completion_reason="killed",
    )
    _register(registry, session)
    readiness = _watch_event(session, sequence=1)
    build_complete = _watch_event(session, sequence=2, output="BUILD COMPLETE")
    build_complete["pattern"] = "BUILD COMPLETE"
    registry.observe_notification(readiness)
    registry.observe_notification(build_complete)
    monkeypatch.setattr(registry, "_open_loopback_ports", lambda _event, _session=None: [])

    assert registry.reconcile_process_notifications(session) is True
    assert registry.is_notification_actionable(readiness) is False
    assert registry.is_notification_actionable(build_complete) is True


def test_loopback_endpoint_parser_supports_ipv4_localhost_and_ipv6():
    text = " ".join(
        [
            "http://127.0.0.1:4312/",
            "http://localhost:4313/",
            "http://[::1]:4314/",
            "https://example.com:4315/",
        ]
    )

    assert ProcessRegistry._loopback_endpoints(text) == [
        ("127.0.0.1", 4312),
        ("localhost", 4313),
        ("::1", 4314),
    ]
