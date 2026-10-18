"""Automatic reclamation must let owned completion receipts settle first."""

import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from run_agent import AIAgent
from hermes_state import SessionDB
from tools.process_registry import ProcessRegistry, ProcessSession
from tui_gateway import server


@pytest.fixture
def receipt_session(tmp_path, monkeypatch):
    import tools.process_registry as processes

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "profile-a-canonical"
    agent._process_owner_task_ids = {"profile-a-turn"}
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._memory_manager = None
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(agent.session_id, source="desktop")
    other_home = tmp_path / "other-home"
    other_home.mkdir()
    other_db = SessionDB(db_path=other_home / "state.db")
    other_db.create_session("profile-b-canonical", source="desktop")
    agent._session_db = db
    session = dict(agent=agent, session_key=agent.session_id, running=False,
                   profile_home=str(home), history_lock=threading.RLock(),
                   transport=server._detached_ws_transport)
    monkeypatch.setattr(server, "_sessions", {"profile-a-runtime": session})
    other_session = dict(agent=SimpleNamespace(session_id="profile-b-canonical"),
                         session_key="profile-b-canonical", running=False,
                         profile_home=str(other_home), history_lock=threading.RLock())
    server._sessions["profile-b-runtime"] = other_session
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(server.threading, "Timer", Timer)
    # Keep teardown's process owner real; other resources are irrelevant here.
    monkeypatch.setattr(server, "_teardown_popped_session",
                        lambda current, **kw: current and agent._close_task_resources(agent.session_id))

    def spawn(*, notify=True, owner="profile-a-turn"):
        release = tmp_path / (owner + "-release")
        code = ("import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); "
                "\nwhile not p.exists(): time.sleep(.01)"
                "\nprint('SYNTHETIC-RECEIPT', flush=True)")
        child = subprocess.Popen([sys.executable, "-u", "-c", code, str(release)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        proc = registry.adopt_local(child, command="synthetic receipt", cwd=str(tmp_path),
                                    task_id="shared-environment", owner_task_id=owner,
                                    session_key=agent.session_id, notify_on_complete=notify)
        return proc, release

    yield SimpleNamespace(registry=registry, agent=agent, session=session,
                          timers=timers, spawn=spawn, db=db,
                          other_db=other_db, other_session=other_session)
    registry.kill_all(source="test_cleanup", consume_output=True)
    db.close()
    other_db.close()


@pytest.mark.parametrize("path", ["orphan", "lru"])
def test_pending_receipt_blocks_automatic_reclamation(receipt_session, path, monkeypatch):
    fixture = receipt_session
    proc, release = fixture.spawn()
    if path == "orphan":
        server._schedule_ws_orphan_reap("profile-a-runtime")
        fixture.timers[0].callback()
        assert server._sessions.get("profile-a-runtime") is fixture.session
    else:
        assert not server._session_is_lru_evictable("profile-a-runtime", fixture.session)
    assert proc.process.poll() is None
    release.touch()
    assert proc._completion_event.wait(10)
    assert proc.exit_code == 0
    # Exited but unread receipts still need their original session, even between
    # reader completion and notification consumption.
    assert not server._session_is_lru_evictable("profile-a-runtime", fixture.session)
    fixture.registry.poll(proc.id)  # Read-only status must not release the guard.
    assert not server._session_is_lru_evictable("profile-a-runtime", fixture.session)
    event = fixture.registry.completion_queue.get(timeout=2)
    # Even dequeued notices keep the session until dispatch claims its turn.
    assert not server._session_is_lru_evictable("profile-a-runtime", fixture.session)
    monkeypatch.setattr(server, "_emit", lambda *args: None)
    monkeypatch.setattr(server, "_background_notifications_off", lambda current: False)

    def accept_receipt(rid, sid, current, text, **kwargs):
        assert sid == "profile-a-runtime"
        assert current is fixture.session
        fixture.db.append_message(current["session_key"], "user", text,
                                  display_kind=kwargs["display_kind"])
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", accept_receipt)
    from tools.process_registry_notifications import format_process_notification
    emitted = set()
    server._notif_handle_ready("profile-b-runtime", fixture.other_session, [event], set(),
                               fixture.registry, format_process_notification, None)
    assert not fixture.other_db.get_messages("profile-b-canonical")
    event = fixture.registry.completion_queue.get(timeout=2)
    for _ in range(2):
        server._notif_handle_ready("profile-a-runtime", fixture.session, [event], emitted,
                                   fixture.registry, format_process_notification, None)
    assert len(fixture.db.get_messages("profile-a-canonical")) == 1
    assert not fixture.other_db.get_messages("profile-b-canonical")
    assert not fixture.registry.is_completion_consumed(proc.id)
    assert fixture.registry.is_completion_delivered(proc.id)
    fixture.session["running"] = False  # Synthetic notification turn settled.
    assert server._session_is_lru_evictable("profile-a-runtime", fixture.session)
    assert "SYNTHETIC-RECEIPT" in fixture.registry.read_log(proc.id)["output"]
    server._close_session_by_id("profile-a-runtime", end_reason="lru_evict")
    assert "profile-a-runtime" not in server._sessions


@pytest.mark.parametrize("notify,owner", [(False, "profile-a-turn"), (True, "profile-b-turn")])
def test_unnotified_or_foreign_process_does_not_hold_session(receipt_session, notify, owner):
    fixture = receipt_session
    proc, _ = fixture.spawn(notify=notify, owner=owner)
    assert server._session_is_lru_evictable("profile-a-runtime", fixture.session)
    server._schedule_ws_orphan_reap("profile-a-runtime")
    fixture.timers[0].callback()
    assert "profile-a-runtime" not in server._sessions
    if owner == "profile-a-turn":
        assert proc.process.poll() is not None
        assert proc.termination_source == "agent_close"
    else:
        assert proc.process.poll() is None


def test_uncollectable_recovered_exit_does_not_hold_session(receipt_session):
    fixture = receipt_session
    recovered = ProcessSession(
        id="proc_recovered_without_status",
        command="recovered detached process",
        owner_task_id="profile-a-turn",
        notify_on_complete=True,
        detached=True,
        exited=True,
        exit_code=None,
    )
    fixture.registry._finished[recovered.id] = recovered

    assert not fixture.registry.has_pending_completions_owned_by({"profile-a-turn"})
    assert server._session_is_lru_evictable("profile-a-runtime", fixture.session)


def test_exit_publication_window_still_holds_session(receipt_session):
    fixture = receipt_session
    publishing = ProcessSession(
        id="proc_publishing_exit",
        command="process publishing its exit",
        owner_task_id="profile-a-turn",
        notify_on_complete=True,
        exited=True,
        exit_code=None,
    )
    fixture.registry._running[publishing.id] = publishing

    assert fixture.registry.has_pending_completions_owned_by({"profile-a-turn"})
    assert not server._session_is_lru_evictable("profile-a-runtime", fixture.session)


def test_explicit_close_still_cancels_receipt(receipt_session):
    fixture = receipt_session
    proc, _ = fixture.spawn()
    server._close_session_by_id("profile-a-runtime", end_reason="tui_close")
    assert "profile-a-runtime" not in server._sessions
    assert proc.process.poll() is not None
    assert proc.termination_source == "agent_close"


def test_explicit_process_stop_still_cancels_receipt(receipt_session):
    fixture = receipt_session
    proc, _ = fixture.spawn()
    fixture.registry.kill_process(proc.id, source="process.kill")
    assert proc.process.poll() is not None
    assert not fixture.registry.has_pending_completions_owned_by({"profile-a-turn"})


def test_status_only_receipt_releases_session(receipt_session, monkeypatch):
    fixture = receipt_session
    proc, release = fixture.spawn()
    release.touch()
    assert proc._completion_event.wait(10)
    event = fixture.registry.completion_queue.get(timeout=2)
    monkeypatch.setattr(server, "_emit", lambda *args: None)
    monkeypatch.setattr(server, "_background_notifications_off", lambda current: True)
    from tools.process_registry_notifications import format_process_notification
    server._notif_handle_ready("profile-a-runtime", fixture.session, [event], set(),
                               fixture.registry, format_process_notification, None)
    assert fixture.registry.is_completion_delivered(proc.id)
    assert not fixture.registry.is_completion_consumed(proc.id)
    assert server._session_is_lru_evictable("profile-a-runtime", fixture.session)


@pytest.mark.parametrize("failure", ["submit", "prepare", "claim", "refusal"])
def test_failed_completion_admission_requeues_receipt(receipt_session, monkeypatch, failure):
    fixture = receipt_session
    proc, release = fixture.spawn()
    release.touch()
    assert proc._completion_event.wait(10)
    event = fixture.registry.completion_queue.get(timeout=2)
    monkeypatch.setattr(server, "_emit", lambda *args: None)
    monkeypatch.setattr(server, "_background_notifications_off", lambda current: False)

    def cannot_start(*args, **kwargs):
        raise RuntimeError("synthetic worker unavailable")

    if failure == "submit":
        monkeypatch.setattr(server, "_run_prompt_submit", cannot_start)
    elif failure == "refusal":
        monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kwargs: False)
    elif failure == "prepare":
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", cannot_start)
    else:
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *args: None)
    from tools.process_registry_notifications import format_process_notification
    server._notif_handle_ready("profile-a-runtime", fixture.session, [event], set(),
                               fixture.registry, format_process_notification, None)
    assert not fixture.session["running"]
    assert not fixture.registry.is_completion_delivered(proc.id)
    assert fixture.registry.completion_queue.get(timeout=2) == event
    assert not server._session_is_lru_evictable("profile-a-runtime", fixture.session)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *args: "")

    def accepted(rid, sid, current, text, **kwargs):
        fixture.db.append_message(current["session_key"], "user", text,
                                  display_kind=kwargs["display_kind"])
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", accepted)
    for _ in range(2):
        server._notif_handle_ready("profile-a-runtime", fixture.session, [event], set(),
                                   fixture.registry, format_process_notification, None)
    assert len(fixture.db.get_messages("profile-a-canonical")) == 1
    assert fixture.registry.is_completion_delivered(proc.id)
    fixture.session["running"] = False
    assert server._session_is_lru_evictable("profile-a-runtime", fixture.session)
