"""A detached Desktop/TUI session waiting on a background process's completion notice must survive the reapers.

The agent starts ``terminal(background=True, notify_on_complete=True)`` and ends its turn on "I'll report back".
The completion reaches the model only through that session's notification poller, and closing the session kills
the process (``agent_close``). The LRU cap evicted such sessions seconds after the turn ended, so the process died
with exit -15, no completion was ever delivered, and the promised follow-up silently never happened.
"""

from __future__ import annotations

import contextlib
import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending_ws_reaps.clear()


def _drain(registry):
    while not registry.completion_queue.empty():
        registry.completion_queue.get_nowait()


@pytest.fixture()
def registry():
    from tools.process_registry import process_registry

    saved = dict(process_registry._running), dict(process_registry._finished)
    process_registry._running.clear()
    process_registry._finished.clear()
    _drain(process_registry)
    yield process_registry
    _drain(process_registry)
    process_registry._running.clear()
    process_registry._finished.clear()
    process_registry._running.update(saved[0])
    process_registry._finished.update(saved[1])


class _Timer:
    created: list = []

    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback
        _Timer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture()
def detached(server, monkeypatch):
    _Timer.created = []
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    popped = []
    monkeypatch.setattr(server, "_teardown_popped_session", lambda s, **kw: popped.append(s))
    sid = "sid-detached-proc"
    session = {
        "session_key": "desktop-proc-session",
        "transport": server._detached_ws_transport,
        "running": False,
        "history_lock": threading.Lock(),
        "last_active": 0.0,
        "created_at": 0.0,
    }
    server._sessions[sid] = session
    return sid, session, popped


def _spawn(registry, owner: str, **kw):
    """A background process as the TUI turn spawns it: owner_task_id is the turn's task id (the session key)."""
    from tools.process_registry import ProcessSession

    proc = ProcessSession(id=f"proc_{len(registry._running)}", command="hermes -p reviewer chat",
                          session_key=owner, owner_task_id=owner, started_at=time.time(), **kw)
    registry._running[proc.id] = proc
    return proc


@pytest.mark.parametrize("notify", [{"notify_on_complete": True}, {"watch_patterns": ["VERDICT"]}])
def test_cap_and_ttl_reapers_spare_session_with_pending_process_notice(server, registry, detached, notify):
    sid, session, _ = detached
    assert server._session_is_reapable(sid, session)  # idle and clientless: evictable today

    _spawn(registry, session["session_key"], **notify)

    assert not server._session_is_reapable(sid, session)
    assert not server._session_is_evictable(sid, session, time.time() + 10 * 86400)


def test_orphan_reap_rearms_while_process_will_notify(server, registry, detached):
    sid, session, popped = detached
    _spawn(registry, session["session_key"], notify_on_complete=True)

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert server._sessions.get(sid) is session and not popped
    assert server._pending_ws_reaps[sid] is _Timer.created[-1]


def test_exited_process_keeps_session_until_its_notice_is_routed(server, registry, detached):
    """Exit recorded → completion published → dequeued by a poller → routed: a reaper firing at any point of that
    hand-off must not close the session and lose the follow-up. Protection ends only when routing returns."""
    from tools.process_registry_awaiting import release, take_ready

    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.exited = True  # mark_exited ran; _move_to_finished (publish) has not yet
    assert not server._session_is_reapable(sid, session)

    registry._move_to_finished(proc)  # published: queued
    assert not server._session_is_reapable(sid, session)

    ready = take_ready(registry.completion_queue, 0)  # a poller took it and is routing it
    assert registry.completion_queue.empty() and len(ready) == 1
    assert not server._session_is_reapable(sid, session)

    release(ready)  # routing returned (turn admitted → ``running`` protects it, or requeued, or dropped)
    assert server._session_is_reapable(sid, session)


def test_unpublished_completion_survives_registry_pruning(server, registry, detached, monkeypatch):
    """Pruning can run (a concurrent spawn) after the exit is recorded but before the completion is queued; the
    entry must survive it, since it is the only record that a notice is still coming."""
    from tools.process_registry import FINISHED_TTL_SECONDS

    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.started_at = time.time() - FINISHED_TTL_SECONDS - 1
    proc.mark_exited(0)
    publish = registry._publish_finished
    seen = []

    def prune_mid_publish(s, was_running):
        with registry._lock:
            registry._prune_if_needed()
        seen.append((s.id in registry._finished, server._session_is_reapable(sid, session)))
        publish(s, was_running)

    monkeypatch.setattr(registry, "_publish_finished", prune_mid_publish)
    registry._move_to_finished(proc)

    assert seen == [(True, False)]
    with registry._lock:  # once published, the entry is ordinary history again
        registry._prune_if_needed()
    assert proc.id not in registry._finished


def test_poller_holds_session_between_dequeue_and_routing(server, registry, detached, monkeypatch):
    """The real poller loop: a reaper that fires after the poller dequeued the completion but before it routed it
    must still see the session as protected."""
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    for name in ("_maybe_fire_tui_loop_tick", "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(registry, "restore_completions", lambda: 0)
    stop, observed = threading.Event(), []

    def routing(*args, **kwargs):
        if kwargs.get("deferred", args[6] if len(args) > 6 else None) is None and not observed:
            observed.append((registry.completion_queue.empty(), server._session_is_reapable(sid, session)))
            stop.set()

    monkeypatch.setattr(server, "_notif_handle_ready", routing)
    worker = threading.Thread(target=server._notification_poller_scoped_loop, args=(stop, sid, session), daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert observed == [(True, False)], "dequeued but unrouted: the session must stay protected"
    assert server._session_is_reapable(sid, session), "routing returned: the hold is released"


@pytest.mark.parametrize("persist", [False, True])
def test_stop_releases_session_even_with_a_running_watch(server, registry, detached, persist):
    """After Stop no notification turn can start until the user's next prompt, so a never-exiting watched server
    must not keep the stopped, disconnected session alive."""
    sid, session, _ = detached
    _spawn(registry, session["session_key"], watch_patterns=["ERROR"], persist_on_release=persist)
    assert not server._session_is_reapable(sid, session)

    session["_turn_cancel_requested"] = True

    assert server._session_is_reapable(sid, session)


def test_queued_notice_survives_registry_pruning(server, registry, detached):
    """Retained history is pruned by start time and a process cap, on every spawn: a long job's queued completion
    must keep protecting its owner after its finished entry is gone."""
    from tools.process_registry import FINISHED_TTL_SECONDS

    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.started_at = time.time() - FINISHED_TTL_SECONDS - 1
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    with registry._lock:
        registry._prune_if_needed()

    assert proc.id not in registry._finished
    assert not server._session_is_reapable(sid, session)


def test_notice_held_after_stop_does_not_pin_session(server, registry, detached):
    """After the user's Stop the poller requeues notices until their next prompt; that wait must not be forever."""
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    session["_turn_cancel_requested"] = True

    assert server._session_is_reapable(sid, session)


def test_recovered_process_whose_pid_died_releases_session(server, registry, detached, monkeypatch):
    """A checkpoint-recovered process has no reader thread to observe its exit; a dead PID must not pin its owner."""
    sid, session, _ = detached
    _spawn(registry, session["session_key"], notify_on_complete=True, detached=True, pid=12345)
    monkeypatch.setattr(registry, "_detached_host_fate", lambda *_: "gone")

    assert server._session_is_reapable(sid, session)


@pytest.mark.parametrize("unit", ["", "hermes-proc.scope"])
def test_uncollected_recovered_exit_never_pins_session(server, registry, detached, unit):
    """A recovered entry closed without a collected exit (exited_at stays 0, a scope keeps it retained) queues no
    completion, so it must not hold its owner however long ago it ended."""
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True, detached=True, pid=12345,
                  systemd_unit=unit)
    registry._prune_uncollected_detached(proc)

    assert proc.exited and not proc.exited_at
    assert server._session_is_reapable(sid, session)


def test_silent_background_process_does_not_pin_session(server, registry, detached):
    """A process nobody asked to hear from (a dev server) must not make its session immortal."""
    sid, session, _ = detached
    _spawn(registry, session["session_key"])

    assert server._session_is_reapable(sid, session)


def test_another_sessions_process_does_not_pin_session(server, registry, detached):
    sid, session, _ = detached
    _spawn(registry, "some-other-session", notify_on_complete=True)

    assert server._session_is_reapable(sid, session)


def test_process_started_before_compression_still_pins_session(server, registry, detached):
    """Compression rotates session_key to the continuation id; the process keeps its original owner id. The agent
    still kills it on close (its _process_owner_task_ids keep the old id), so it must still pin the session."""
    from types import SimpleNamespace

    sid, session, _ = detached
    original = session["session_key"]
    _spawn(registry, original, notify_on_complete=True)
    session["session_key"] = "desktop-proc-session-continuation"
    session["agent"] = SimpleNamespace(session_id=session["session_key"],
                                       _process_owner_task_ids={original, session["session_key"]})

    assert not server._session_is_reapable(sid, session)


def test_disabled_watch_does_not_pin_session(server, registry, detached):
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], watch_patterns=["VERDICT"])
    proc._watch_disabled = True

    assert server._session_is_reapable(sid, session)


# ── the post-turn safety-net drain and agent rebuilds ──────────────────────
# Session B finishing a turn drains the shared completion queue; an event owned by idle, detached session A passes
# through B's hands before it is requeued. If A were reapable in that window, the reaper would close A and the
# requeued notice would be dropped as unowned.


def _post_turn_followups(server, namespace_overrides):
    from tui_gateway import prompt_turn
    from tui_gateway.method_ctx import rebind

    namespace = dict(vars(server))
    namespace.update(_drain_queued_prompt=lambda *_: False,
                     _session_profile_runtime_scope=lambda *_: contextlib.nullcontext(), **namespace_overrides)
    return rebind(prompt_turn._run_post_turn_followups, namespace)


def test_foreign_notice_in_another_sessions_post_turn_drain_keeps_its_owner(server, registry, detached, monkeypatch):
    sid, session, popped = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    other_sid, other = "other-sid", dict(session, session_key="other-owner", history_lock=threading.Lock())
    server._sessions[other_sid] = other
    monkeypatch.setattr(registry, "restore_completions", lambda: 0)
    real_owns, observed = server._session_owns_notification_event, []

    def owns(draining_sid, draining_session, event):
        # B is deciding ownership of A's notice; the queue is empty, so only the in-flight hold covers A.
        if draining_sid == other_sid and event["owner_task_id"] == session["session_key"]:
            observed.append((registry.completion_queue.empty(), server._session_is_reapable(sid, session)))
            server._schedule_ws_orphan_reap(sid)
            server._pending_ws_reaps[sid].callback()
        return real_owns(draining_sid, draining_session, event)

    routed = []
    _post_turn_followups(server, {"_session_owns_notification_event": owns,
                                  "_notif_handle_ready": lambda *a, **k: routed.append(a[2])})(
        None, other_sid, other, {}, None)

    assert observed == [(True, False)], "A's notice was in B's hands: A must stay protected"
    assert sid in server._sessions and not popped
    assert [e["owner_task_id"] for e in registry.completion_queue.queue] == [session["session_key"]]
    assert routed == [[]], "B routes none of A's events"


def test_post_turn_drain_routes_its_own_notice_and_releases_every_hold(server, registry, detached, monkeypatch):
    from tools import process_registry_awaiting as awaiting

    sid, session, _ = detached
    for owner in (session["session_key"], "someone-else"):
        proc = _spawn(registry, owner, notify_on_complete=True)
        proc.mark_exited(0)
        registry._move_to_finished(proc)
    monkeypatch.setattr(registry, "restore_completions", lambda: 0)
    routed = []
    _post_turn_followups(server, {"_notif_handle_ready": lambda *a, **k: routed.append(
        [(e["owner_task_id"], server._session_is_reapable(sid, session)) for e in a[2]])})(None, sid, session, {}, None)

    assert routed == [[(session["session_key"], False)]], "own notice routed while its session is still held"
    assert [e["owner_task_id"] for e in registry.completion_queue.queue] == ["someone-else"]
    assert not awaiting._in_flight, "every hold released once routing returned"
    assert server._session_is_reapable(sid, session)


def test_post_turn_drain_releases_holds_when_routing_raises(server, registry, detached, monkeypatch):
    from tools import process_registry_awaiting as awaiting

    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    monkeypatch.setattr(registry, "restore_completions", lambda: 0)

    def fail(*_a, **_k):
        raise RuntimeError("routing failed")

    _post_turn_followups(server, {"_notif_handle_ready": fail})(None, sid, session, {}, None)
    assert not awaiting._in_flight


def test_agent_rebuild_keeps_pre_compression_process_owners(server, registry, detached, monkeypatch):
    """A Bot Chat capability refresh replaces the agent; a process the old agent started under its pre-compression
    id must still pin the session (and still be killed by the new agent's close)."""
    sid, session, _ = detached
    session["agent"] = SimpleNamespace(_process_owner_task_ids={"pre-compression-id", session["session_key"]},
                                       _session_db=None, _owns_session_db=False)
    _spawn(registry, "pre-compression-id", notify_on_complete=True)
    assert not server._session_is_reapable(sid, session)
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: SimpleNamespace(
        _process_owner_task_ids={session["session_key"]}, _session_db=None))
    monkeypatch.setattr(server, "_release_build_profile_scopes", lambda *_: None)

    server._rebuild_session_agent(sid, session)

    assert "pre-compression-id" in session["agent"]._process_owner_task_ids
    assert not server._session_is_reapable(sid, session)
