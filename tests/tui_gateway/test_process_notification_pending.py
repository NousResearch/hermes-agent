"""The process RPC must not advertise Done while its follow-up is outstanding."""

import asyncio
import json
import shlex
from collections import OrderedDict
from queue import Queue
import sys
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.session_context import scoped_current_session_id
from tools import process_registry as processes
from tools.process_registry_notifications import format_process_notification
from tui_gateway import server


@pytest.fixture
def delivery(monkeypatch, tmp_path):
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    session: dict = dict(session_key="pending-owner", agent=SimpleNamespace(session_id="pending-owner"),
                   history=[], history_lock=threading.Lock(), running=False)
    monkeypatch.setitem(server._sessions, "pending-ui", session)
    events = []
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: events.append(args))

    def rows():
        response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "process.list",
                                          "params": {"session_id": "pending-ui"}})
        assert response is not None and "error" not in response, response
        return {row["session_id"]: row for row in response["result"]["processes"]}

    def spawn(code="input(); print('finished', flush=True)", **kwargs):
        command = f"{shlex.quote(sys.executable)} -u -c {shlex.quote(code)}"
        child = subprocess.Popen([sys.executable, "-u", "-c", code], text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 cwd=tmp_path)
        with scoped_current_session_id(session["session_key"]):
            return registry.adopt_local(child, command=command, cwd=str(tmp_path),
                                        session_key=session["session_key"], notify_on_complete=False, **kwargs)

    yield registry, session, events, rows, spawn
    for proc in list(registry._running.values()) + list(registry._finished.values()):
        if not proc.exited:
            registry.kill_process(proc.id)
        if proc._reader_thread:
            proc._reader_thread.join(timeout=5)


def _fast_terminal_launch(registry, rows, monkeypatch, *, watch=False):
    from tools.terminal_tool_background import spawn_background_process
    from hermes_constants import get_hermes_home

    track = registry._track_started

    def finish_before_spawn_returns(proc, *args):
        track(proc, *args)
        assert proc._completion_event.wait(5)
        assert proc.exit_code == 0
        assert rows()[proc.id]["notification_pending"] is True

    monkeypatch.setattr(registry, "_track_started", finish_before_spawn_returns)
    result = json.loads(spawn_background_process(
        command=f"{shlex.quote(sys.executable)} -c \"print('READY')\"",
        env=SimpleNamespace(env={}), env_type="local", effective_task_id="owner", task_id="owner",
        session_key="pending-owner", workdir=None, cwd=str(get_hermes_home()), effective_pty=False,
        notify_on_complete=not watch, watch_patterns=["READY"] if watch else None,
        approval_note=None, pty_disabled_reason=None))
    assert result.get("error") is None, result
    return registry.get(result["session_id"])


@pytest.mark.parametrize("settlement", ["poller", "post_turn", "wait", "log", "kill", "suppressed", "orphan", "dispatch_error", "gateway", "cli", "launch"])
def test_completion_pending_survives_public_exit_dequeue_and_busy_turn(delivery, monkeypatch, settlement):
    registry, session, events, rows, spawn = delivery
    if settlement == "launch":
        from tools.async_delegation import complete_event_delivery
        proc = _fast_terminal_launch(registry, rows, monkeypatch)
        complete_event_delivery(registry.completion_queue.get(timeout=5), "")
        assert rows()[proc.id]["notification_pending"] is False
        return
    proc = spawn(owner_task_id="sa-child" if settlement == "suppressed" else "owner")
    proc.notify_on_complete = True
    reached_exit, release_exit = threading.Event(), threading.Event()
    checkpoint = registry._write_checkpoint

    def pause_checkpoint():
        if proc.exited:
            reached_exit.set()
            assert release_exit.wait(5)
        checkpoint()

    monkeypatch.setattr(registry, "_write_checkpoint", pause_checkpoint)
    try:
        registry.submit_stdin(proc.id)
        assert reached_exit.wait(5)
        row = rows()[proc.id]
        assert row["status"] == "exited"
        assert row.get("notification_pending") is True
        assert registry.completion_queue.empty()  # exit is visible BEFORE enqueue
    finally:
        release_exit.set()
    assert proc._completion_event.wait(5)
    assert proc.exit_code == 0
    assert proc.output_buffer.strip() == "finished"
    # The monitor's poll is observation, not consumption.
    registry.poll(proc.id)
    assert rows()[proc.id]["notification_pending"] is True
    proc.started_at -= processes.FINISHED_TTL_SECONDS + 1
    with registry._lock:
        registry._prune_if_needed()
    assert proc.id in rows(), "pending finished processes must survive pruning"

    evt = registry.completion_queue.get(timeout=5)
    assert rows()[proc.id]["notification_pending"] is True
    foreign = dict(session_key="foreign-owner", agent=SimpleNamespace(session_id="foreign-owner"),
                   history_lock=threading.Lock(), running=False)
    deferred = []
    server._notif_handle_ready("foreign-ui", foreign, [evt], set(), registry,
                               format_process_notification, deferred)
    assert deferred == [evt]
    assert rows()[proc.id]["notification_pending"] is True  # foreign shutdown drain
    server._notif_handle_ready("foreign-ui", foreign, deferred, set(), registry,
                               format_process_notification, None)
    assert rows()[proc.id]["notification_pending"] is True  # foreign poller requeue
    assert registry.completion_queue.get(timeout=5) is evt
    registry.completion_queue.put(evt)
    session["running"] = True
    server._run_post_turn_followups("test", "pending-ui", session, {}, None)
    if settlement == "suppressed":
        assert registry.completion_queue.empty()
        assert rows()[proc.id]["notification_pending"] is False
        return
    assert rows()[proc.id]["notification_pending"] is True
    assert registry.completion_queue.qsize() == 1  # busy requeue is NOT acceptance
    session["running"] = False

    def accepted(_rid, _sid, target, text):
        assert "finished" in str(text)
        assert rows()[proc.id]["notification_pending"] is True
        target["running"] = False
        if settlement == "dispatch_error":
            raise RuntimeError("bounded test dispatch failure")

    monkeypatch.setattr(server, "_run_prompt_submit", accepted)
    if settlement == "cli":
        from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin

        class InputQueue(Queue):
            def put(self, item, *args, **kwargs):
                assert rows()[proc.id]["notification_pending"] is True
                super().put(item, *args, **kwargs)

        consumer = CLIProcessNotificationsMixin()
        consumer.session_id = session["session_key"]
        consumer._pending_input = InputQueue()
        # CLI's poll-observed suppression is tested elsewhere; exercise acceptance.
        registry._poll_observed.clear()
        consumer._drain_process_notifications("test")
        assert consumer._pending_input.qsize() == 1
    elif settlement == "gateway":
        from gateway.run import _drain_gateway_watch_events
        from gateway.run_notifications import GatewayNotificationsMixin

        # Messaging drains discard the queue copy; the watcher still owns delivery.
        assert _drain_gateway_watch_events(registry.completion_queue) == []
        assert rows()[proc.id]["notification_pending"] is True
        runner = GatewayNotificationsMixin()
        runner._completion_delivery_lock = threading.Lock()
        runner._completion_deliveries_inflight = set()
        runner._completion_deliveries_delivered = OrderedDict()
        runner._completion_delivery_retention = 64
        accepting = False

        async def inject(_text, _event):
            assert rows()[proc.id]["notification_pending"] is True
            return accepting

        monkeypatch.setattr(runner, "_inject_watch_notification", inject)
        assert asyncio.run(runner._deliver_completion_notification("finished", evt)) is False
        assert rows()[proc.id]["notification_pending"] is True
        accepting = True
        assert asyncio.run(runner._deliver_completion_notification("finished", evt)) is True
    elif settlement in {"wait", "log", "kill"}:
        {"wait": registry.wait, "log": registry.read_log, "kill": registry.kill_process}[settlement](proc.id)
    elif settlement == "post_turn":
        server._run_post_turn_followups("test", "pending-ui", session, {}, None)
    else:
        evt = registry.completion_queue.get(timeout=5)
        if settlement == "orphan":
            session["session_key"] = "another-owner"
            session["agent"].session_id = "another-owner"
        server._notif_handle_ready("pending-ui", session, [evt], set(), registry,
                                   format_process_notification, [])
        session["session_key"] = "pending-owner"
    assert rows()[proc.id]["notification_pending"] is False
    if settlement in {"poller", "post_turn"}:
        assert any(event[0] == "message.start" for event in events)
    # Settled entries become eligible again, rather than pinning the registry forever.
    with registry._lock:
        registry._prune_if_needed()
    assert proc.id not in rows()
    with scoped_current_session_id("pending-owner"):
        retained = {row["session_id"]: row for row in registry.list_sessions(include_retained=True)}
    assert retained[proc.id]["notification_pending"] is False  # receipts do not replay notices


@pytest.mark.parametrize("settlement", ["poller", "post_turn", "suppressed", "poller_suppressed", "breaker", "disabled", "gateway", "gateway_off", "launch"])
def test_watch_pending_bridges_first_hit_until_each_notice_is_settled(delivery, monkeypatch, settlement):
    from tools.async_delegation import complete_event_delivery

    registry, session, events, rows, spawn = delivery
    if settlement == "launch":
        proc = _fast_terminal_launch(registry, rows, monkeypatch, watch=True)
        complete_event_delivery(registry.completion_queue.get(timeout=5), "")
        assert rows()[proc.id]["notification_pending"] is False
        return
    monkeypatch.setattr(processes, "WATCH_MIN_INTERVAL_SECONDS", 0)
    if settlement == "disabled":
        monkeypatch.setattr(processes, "WATCH_LIFETIME_MAX_HITS", 1)
    proc = spawn("input(); print('READY', flush=True); input(); print('READY', flush=True); input()",
                 owner_task_id="sa-child" if "suppressed" in settlement else "owner")
    proc.watch_patterns = ["READY"]
    reached_hit, release_hit, ingested = threading.Event(), threading.Event(), threading.Event()
    admit = registry._global_watch_admit
    if settlement == "breaker":
        registry._global_watch_tripped_until = time.time() + 60

    def pause_admission(now):
        reached_hit.set()
        assert release_hit.wait(5)
        return admit(now)

    monkeypatch.setattr(registry, "_global_watch_admit", pause_admission)
    registry.on_output = lambda _proc, text: ingested.set() if "READY" in text else None
    try:
        registry.submit_stdin(proc.id)
        assert reached_hit.wait(5)
        row = rows()[proc.id]
        assert row.get("notification_pending") is True
        assert not registry.is_session_waiting(proc.id) or settlement == "disabled"
        assert registry.completion_queue.empty()  # first hit is public, not queued yet
    finally:
        release_hit.set()
    assert ingested.wait(5)
    if settlement == "breaker":
        assert registry.completion_queue.empty()
        assert rows()[proc.id]["notification_pending"] is False
        return

    first = registry.completion_queue.get(timeout=5)
    assert rows()[proc.id]["notification_pending"] is True
    if settlement == "disabled":
        # The disable summary and promised exit notice remain independent obligations.
        second = registry.completion_queue.get(timeout=5)
        complete_event_delivery(first, "")
        complete_event_delivery(second, "")
        assert rows()[proc.id]["notification_pending"] is True  # exit still promised
        registry.submit_stdin(proc.id)
        registry.submit_stdin(proc.id)
        assert proc._completion_event.wait(5)
        complete_event_delivery(registry.completion_queue.get(timeout=5), "")
        assert rows()[proc.id]["notification_pending"] is False
        return

    ingested.clear()
    registry.submit_stdin(proc.id)
    assert ingested.wait(5)
    second = registry.completion_queue.get(timeout=5)
    complete_event_delivery(first, "")
    complete_event_delivery(first, "")  # duplicate ack must not consume a newer hit
    assert rows()[proc.id]["notification_pending"] is True
    registry.completion_queue.put(second)
    registry.submit_stdin(proc.id)
    assert proc._completion_event.wait(5)
    assert rows()[proc.id]["status"] == "exited"
    assert rows()[proc.id]["notification_pending"] is True

    def accepted(_rid, _sid, target, text):
        assert "READY" in str(text)
        assert rows()[proc.id]["notification_pending"] is True
        target["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", accepted)
    session["running"] = True
    if settlement.startswith("gateway"):
        from gateway.run import GatewayRunner
        from hermes_constants import get_hermes_home

        runner = object.__new__(GatewayRunner)
        mode = "off" if settlement == "gateway_off" else "all"
        (get_hermes_home() / "config.yaml").write_text(
            f"display:\n  background_process_notifications: {mode!r}\n")

        async def inject(text, _event):
            assert "READY" in text
            assert rows()[proc.id]["notification_pending"] is True
            return True

        monkeypatch.setattr(runner, "_inject_watch_notification", inject)
        asyncio.run(runner._drain_watch_notifications(registry.completion_queue))
        assert rows()[proc.id]["notification_pending"] is False
        return
    if settlement == "poller_suppressed":
        event = registry.completion_queue.get(timeout=5)
        server._notif_handle_ready("pending-ui", session, [event], set(), registry,
                                   format_process_notification, [])
    else:
        server._run_post_turn_followups("test", "pending-ui", session, {}, None)
    if "suppressed" in settlement:
        assert registry.completion_queue.empty()
        assert rows()[proc.id]["notification_pending"] is False
        assert not any(event[0] == "message.start" for event in events)
        return
    assert rows()[proc.id]["notification_pending"] is True
    assert registry.completion_queue.qsize() == 1
    session["running"] = False
    if settlement == "post_turn":
        server._run_post_turn_followups("test", "pending-ui", session, {}, None)
    else:
        server._notif_handle_ready("pending-ui", session, [registry.completion_queue.get(timeout=5)],
                                   set(), registry, format_process_notification, [])
    assert rows()[proc.id]["notification_pending"] is False
    assert any(event[0] == "message.start" for event in events)
