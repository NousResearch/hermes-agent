"""Regression: background process completions carry origin_ui_session_id.

WebUI/desktop multi-chat hosts race process-global session keys when concurrent
turns spawn terminal(notify_on_complete=True). The commissioning tab's UI id
must be stamped immutably on ProcessSession and completion events so consumers
can route wakeups to the true owner even when session_key is contaminated or
belongs to a temporary delegated child.
"""
from __future__ import annotations

import os
import time

import pytest


def test_spawn_local_stamps_origin_ui_session_id(tmp_path):
    from tools.process_registry import ProcessRegistry

    reg = ProcessRegistry()
    sess = reg.spawn_local(
        command="printf ok",
        cwd=str(tmp_path),
        session_key="child-task-key",
        origin_ui_session_id="webui-tab-A",
    )
    assert sess.origin_ui_session_id == "webui-tab-A"
    assert sess.spawn_session_id == "webui-tab-A"
    assert sess.session_key == "child-task-key"

    # Wait for exit + completion enqueue.
    deadline = time.time() + 5
    while not sess.exited and time.time() < deadline:
        time.sleep(0.05)
    # Force move if reader hasn't yet.
    if not sess.exited:
        sess.exited = True
        sess.exit_code = 0
        sess.notify_on_complete = True
        reg._move_to_finished(sess)
    else:
        sess.notify_on_complete = True
        # If already finished without notify, re-enqueue isn't automatic;
        # inspect the finished session fields instead.
        assert reg.get(sess.id) is not None

    # Drain any completion event and assert origin is present when notify set
    # before finish. For processes that finished before notify_on_complete was
    # set, verify the persisted ProcessSession still carries the origin.
    finished = reg.get(sess.id)
    assert finished is not None
    assert finished.origin_ui_session_id == "webui-tab-A"


def test_completion_event_includes_origin_ui_session_id(tmp_path):
    from tools.process_registry import ProcessRegistry

    reg = ProcessRegistry()
    sess = reg.spawn_local(
        command="true",
        cwd=str(tmp_path),
        session_key="sess-key",
        origin_ui_session_id="owner-tab",
    )
    sess.notify_on_complete = True
    deadline = time.time() + 5
    while sess.id in reg._running and time.time() < deadline:
        time.sleep(0.05)
    # Ensure finished path ran
    if sess.id in reg._running:
        sess.exited = True
        sess.exit_code = 0
        reg._move_to_finished(sess)

    events = []
    while True:
        try:
            events.append(reg.completion_queue.get_nowait())
        except Exception:
            break
    completion = [e for e in events if e.get("type") == "completion" and e.get("session_id") == sess.id]
    assert completion, f"expected completion event, got {events!r}"
    assert completion[0].get("origin_ui_session_id") == "owner-tab"


def test_bind_ui_session_id_is_nestable():
    from gateway.session_context import (
        _SESSION_UI_SESSION_ID,
        _UNSET,
        bind_ui_session_id,
        get_session_env,
    )

    assert _SESSION_UI_SESSION_ID.get() is _UNSET
    with bind_ui_session_id("outer"):
        assert get_session_env("HERMES_UI_SESSION_ID", "") == "outer"
        with bind_ui_session_id("inner"):
            assert get_session_env("HERMES_UI_SESSION_ID", "") == "inner"
        assert get_session_env("HERMES_UI_SESSION_ID", "") == "outer"
    assert _SESSION_UI_SESSION_ID.get() is _UNSET


def test_terminal_capture_uses_ui_session_id(monkeypatch, tmp_path):
    """terminal_tool must prefer HERMES_UI_SESSION_ID over child session_key."""
    pytest.importorskip("tools.terminal_tool")
    from gateway.session_context import bind_ui_session_id, set_session_vars, clear_session_vars
    from tools.approval import set_current_session_key, reset_current_session_key
    from tools import terminal_tool as tt
    from tools.process_registry import process_registry

    # Isolate registry queue noise
    tokens = set_session_vars(
        platform="webui",
        session_key="child-or-wrong-key",
        ui_session_id="parent-tab-XYZ",
        async_delivery=True,
    )
    approval_tok = set_current_session_key("child-or-wrong-key")
    try:
        with bind_ui_session_id("parent-tab-XYZ"):
            # Force local env path with a no-op command
            monkeypatch.setenv("TERMINAL_ENV", "local")
            raw = tt.terminal_tool(
                command="true",
                background=True,
                notify_on_complete=True,
                workdir=str(tmp_path),
            )
            import json
            data = json.loads(raw)
            assert data.get("session_id"), data
            sess = process_registry.get(data["session_id"])
            assert sess is not None
            assert sess.origin_ui_session_id == "parent-tab-XYZ"
            assert sess.session_key == "child-or-wrong-key"
    finally:
        reset_current_session_key(approval_tok)
        clear_session_vars(tokens)
