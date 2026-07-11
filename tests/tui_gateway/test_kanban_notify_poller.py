"""Kanban terminal events delivered to the owning TUI session."""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tui_gateway import server


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_sub(chat_id: str):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Desktop completion", assignee="worker")
        kb.add_notify_sub(conn, task_id=task_id, platform="tui", chat_id=chat_id)
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, kind="completed")
    return task_id


def _cursor(task_id: str, chat_id: str) -> int:
    with kb.connect() as conn:
        return int(kb.list_notify_subs(conn, task_id=task_id)[0]["last_event_id"])


def _session(**extra):
    return {
        "session_key": "sess-key-1",
        "history_lock": threading.Lock(),
        "running": False,
        **extra,
    }


def test_tui_poller_claims_once_emits_and_chains_turn(kanban_home, monkeypatch):
    task_id = _seed_sub("sess-key-1")
    emitted, submitted = [], []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args: submitted.append(args))

    server._poll_kanban_tui_subs("sid1", _session())

    status_updates = [event for event in emitted if event[0] == "status.update"]
    assert len(status_updates) == 1
    assert status_updates[0][0:2] == ("status.update", "sid1")
    assert "Desktop completion" in status_updates[0][2]["text"]
    assert "completed" in status_updates[0][2]["text"]
    assert len(submitted) == 1
    assert _cursor(task_id, "sess-key-1") > 0

    server._poll_kanban_tui_subs("sid1", _session())
    assert len([event for event in emitted if event[0] == "status.update"]) == 1


def test_tui_poller_emits_but_does_not_chain_busy_session(kanban_home, monkeypatch):
    _seed_sub("sess-key-1")
    emitted, submitted = [], []
    session = _session(running=True)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args: submitted.append(args))

    server._poll_kanban_tui_subs("sid1", session)

    assert len([event for event in emitted if event[0] == "status.update"]) == 1
    assert submitted == []
    assert len(session["_pending_kanban_turns"]) == 1


def test_tui_poller_drains_busy_session_turn_when_idle(kanban_home, monkeypatch):
    _seed_sub("sess-key-1")
    submitted = []
    session = _session(running=True)
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args: submitted.append(args))

    server._poll_kanban_tui_subs("sid1", session)
    session["running"] = False
    server._poll_kanban_tui_subs("sid1", session)

    assert len(submitted) == 1
    assert "[kanban] Desktop completion: completed" in submitted[0][-1]
    assert session["_pending_kanban_turns"] == []


def test_tui_poller_isolates_per_event_delivery_failures(kanban_home, monkeypatch):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Desktop completion", assignee="worker")
        kb.add_notify_sub(conn, task_id=task_id, platform="tui", chat_id="sess-key-1")
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, kind="blocked")
            kb._append_event(conn, task_id, kind="completed")
    emitted = []

    def _emit(*args):
        if args[0] == "status.update" and not emitted:
            emitted.append(("failed", args))
            raise RuntimeError("first event failed")
        emitted.append(("ok", args))

    monkeypatch.setattr(server, "_emit", _emit)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args: None)

    server._poll_kanban_tui_subs("sid1", _session())

    successful = [args for result, args in emitted if result == "ok" and args[0] == "status.update"]
    assert len(successful) == 1
    assert "completed" in successful[0][2]["text"]
    assert _cursor(task_id, "sess-key-1") > 0


def test_tui_poller_accepts_stale_session_key(kanban_home, monkeypatch):
    _seed_sub("old-key")
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args: None)

    server._poll_kanban_tui_subs(
        "sid1", _session(session_key="new-key", _stale_session_keys=["old-key"])
    )

    assert len([event for event in emitted if event[0] == "status.update"]) == 1


def test_compression_preserves_old_session_key_for_kanban_poller(monkeypatch):
    session = {
        "agent": types.SimpleNamespace(session_id="new-key"),
        "session_key": "old-key",
    }
    approval = types.SimpleNamespace(
        disable_session_yolo=lambda *_args: None,
        enable_session_yolo=lambda *_args: None,
        is_session_yolo_enabled=lambda *_args: False,
        register_gateway_notify=lambda *_args: None,
        unregister_gateway_notify=lambda *_args: None,
    )
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_args, **_kwargs: True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "tools.approval", approval)
        server._sync_session_key_after_compress(
            "sid1", session, clear_pending_title=False, restart_slash_worker=False
        )

    assert session["session_key"] == "new-key"
    assert session["_stale_session_keys"] == ["old-key"]


def test_compression_caps_stale_session_keys(monkeypatch):
    session = {
        "agent": types.SimpleNamespace(session_id="new-key-0"),
        "session_key": "old-key",
    }
    approval = types.SimpleNamespace(
        disable_session_yolo=lambda *_args: None,
        enable_session_yolo=lambda *_args: None,
        is_session_yolo_enabled=lambda *_args: False,
        register_gateway_notify=lambda *_args: None,
        unregister_gateway_notify=lambda *_args: None,
    )
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_args, **_kwargs: True)
    old_keys = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "tools.approval", approval)
        for i in range(10):
            old_keys.append(session["session_key"])
            server._sync_session_key_after_compress(
                "sid1", session, clear_pending_title=False, restart_slash_worker=False
            )
            session["agent"].session_id = f"new-key-{i + 1}"

    assert len(session["_stale_session_keys"]) == 8
    assert session["_stale_session_keys"][-1] == old_keys[-1]


def test_tui_poller_consumes_event_when_status_delivery_fails(kanban_home, monkeypatch):
    task_id = _seed_sub("sess-key-1")

    def _boom(*_args):
        raise RuntimeError("emit failed")

    monkeypatch.setattr(server, "_emit", _boom)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args: None)

    server._poll_kanban_tui_subs("sid1", _session())

    assert _cursor(task_id, "sess-key-1") > 0


def test_tui_poller_leaves_foreign_subscription_unclaimed(kanban_home, monkeypatch):
    task_id = _seed_sub("other-sess")
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    server._poll_kanban_tui_subs("sid1", _session())

    assert emitted == []
    assert _cursor(task_id, "other-sess") == 0
