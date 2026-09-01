"""Desktop/TUI turn-dispatch observability (#86647).

During the #79278/#86647 persistent-mute investigation the decisive evidence
was an *absence*: a Desktop request left no INFO record in ``agent.log`` or
``gateway.log`` at all (``0 platform=desktop`` across the whole file), so a
muted window was structurally indistinguishable from a request that never
arrived. This suite pins the two-record contract that fixes that:

* ``_run_prompt_submit`` logs one ``tui prompt accepted`` INFO record before
  the turn thread starts, carrying the UI session id, the gateway
  ``session_key``, and the agent's live ``session_id`` (rotated independently
  by compression — the triple is what a rotation-mute trace needs).
* The turn's ``finally`` logs exactly one ``tui turn finished`` bookend on
  every path (success, returned error, exception), re-reading
  ``agent.session_id`` so a mid-turn compression rotation shows up as an
  accepted/finished pair with different agent ids.
* No prompt content is ever logged.
"""

from __future__ import annotations

import logging
import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "gw-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _records(caplog, needle):
    return [r for r in caplog.records if needle in r.getMessage()]


SECRETISH_PROMPT = "please rotate QDRANT_API_KEY=hunter2-super-secret now"


def test_accepted_and_finished_records_on_success(turn_env, caplog):
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        server._run_prompt_submit("rid", "ui-sid", session, SECRETISH_PROMPT)

    accepted = _records(caplog, "tui prompt accepted")
    finished = _records(caplog, "tui turn finished")
    assert len(accepted) == 1
    assert len(finished) == 1

    msg = accepted[0].getMessage()
    # The full id triple a rotation-mute trace needs.
    assert "ui_session=ui-sid" in msg
    assert "session_key=gw-session-key" in msg
    assert "agent_session_id=agent-sid-1" in msg
    # Prompt content is never logged — only its length.
    assert "hunter2" not in msg
    assert "QDRANT_API_KEY" not in msg
    assert f"chars={len(SECRETISH_PROMPT)}" in msg

    fin = finished[0].getMessage()
    assert "ui_session=ui-sid" in fin
    assert "status=complete" in fin
    assert "hunter2" not in fin


def test_finished_record_reflects_mid_turn_rotation(turn_env, caplog):
    """Compression rotating agent.session_id mid-turn must be visible as an
    accepted/finished pair with different agent ids — that pair IS the
    rotation trace #86647 asks for."""

    agent = types.SimpleNamespace(session_id="parent-sid", clear_interrupt=lambda: None)

    def _rotate_and_finish(*a, **k):
        agent.session_id = "continuation-sid"  # what _compress_context does
        return {"final_response": "done"}

    agent.run_conversation = _rotate_and_finish
    session = _session(agent=agent, running=True)

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        server._run_prompt_submit("rid", "ui-sid", session, "go")

    accepted = _records(caplog, "tui prompt accepted")[0].getMessage()
    finished = _records(caplog, "tui turn finished")[0].getMessage()
    assert "agent_session_id=parent-sid" in accepted
    assert "agent_session_id=continuation-sid" in finished


def test_finished_record_fires_on_exception_path(turn_env, caplog):
    def _boom(*a, **k):
        raise RuntimeError("connection reset mid-stream")

    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=_boom,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        server._run_prompt_submit("rid", "ui-sid", session, "go")

    finished = _records(caplog, "tui turn finished")
    assert len(finished) == 1
    msg = finished[0].getMessage()
    assert "status=error" in msg
    assert "error_retained=True" in msg


def test_finished_record_fires_on_returned_error(turn_env, caplog):
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {
            "final_response": "",
            "error": "provider 402: billing wall",
            "failed": True,
        },
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        server._run_prompt_submit("rid", "ui-sid", session, "go")

    finished = _records(caplog, "tui turn finished")
    assert len(finished) == 1
    assert "status=error" in finished[0].getMessage()


def test_blocked_context_reference_completes_hosted_terminal_callback(
    turn_env,
    monkeypatch,
):
    from agent import context_references, model_metadata

    run_calls = []
    callbacks = []
    persisted = []
    retired = []
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        model="test-model",
        base_url="",
        api_key="",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=lambda *a, **k: run_calls.append((a, k)),
    )
    session = _session(
        agent=agent,
        running=True,
        _hosted_room_task={
            "room_id": "room-1",
            "task_id": "task-1",
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "execution_generation": 1,
        },
    )
    monkeypatch.setattr(
        model_metadata,
        "get_model_context_length",
        lambda *a, **k: 100,
    )
    monkeypatch.setattr(
        context_references,
        "preprocess_context_references",
        lambda *a, **k: types.SimpleNamespace(
            blocked=True,
            warnings=["Context reference exceeds the safe limit."],
            message="",
        ),
    )

    def persist(_session, receipt):
        persisted.append(receipt)
        return ({**receipt, "settlement_id": "reply:task-1:1"}, True)

    monkeypatch.setattr(server, "_persist_hosted_terminal_receipt", persist)
    monkeypatch.setattr(
        server,
        "_retire_turn_marker",
        lambda _session, marker_key: retired.append(marker_key),
    )

    server._run_prompt_submit(
        "rid",
        "ui-sid",
        session,
        "inspect @file",
        terminal_callback=callbacks.append,
    )

    assert run_calls == []
    assert persisted == [
        {
            "status": "failed",
            "text": "",
            "error": "Context reference exceeds the safe limit.",
        }
    ]
    assert callbacks == [
        {
            **persisted[0],
            "settlement_id": "reply:task-1:1",
        }
    ]
    assert retired == ["gw-session-key"]


def test_agent_init_failure_completes_hosted_terminal_callback(
    turn_env,
    monkeypatch,
):
    callbacks = []
    persisted = []
    agent_builds = []
    hosted_task = {
        "room_id": "room-1",
        "task_id": "task-1",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "execution_generation": 1,
    }
    session = _session(source="bot_room")

    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(
        server,
        "_sess_nowait",
        lambda _params, _rid: (session, None),
    )
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(
        server,
        "_load_dashboard_process_isolation_config",
        lambda: {},
    )
    monkeypatch.setattr(
        server,
        "_session_uses_compute_host",
        lambda _session, _config: False,
    )
    monkeypatch.setattr(server, "current_transport", lambda: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *_args: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda sid, _session: agent_builds.append(sid),
    )
    monkeypatch.setattr(
        server,
        "_wait_agent_for_prompt",
        lambda _session, _rid, _sid: {
            "error": {"message": "agent initialization failed: invalid config"}
        },
    )
    monkeypatch.setattr(
        server, "_emit_terminal_turn_error", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: pytest.fail(
            "failed agent initialization must not run the hosted prompt"
        ),
    )

    def persist(_session, receipt):
        persisted.append(receipt)
        return ({**receipt, "settlement_id": "reply:task-1:1"}, True)

    monkeypatch.setattr(server, "_persist_hosted_terminal_receipt", persist)

    result = server._methods["prompt.submit"](
        "rid",
        {
            "session_id": "ui-sid",
            "text": "continue",
            "_hosted_task": hosted_task,
            "_hosted_terminal_callback": callbacks.append,
        },
    )

    assert result["result"]["status"] == "streaming"
    assert agent_builds == ["ui-sid"]
    assert persisted == [
        {
            "status": "failed",
            "text": "",
            "error": "agent initialization failed: invalid config",
        }
    ]
    assert callbacks == [
        {
            **persisted[0],
            "settlement_id": "reply:task-1:1",
        }
    ]
    assert session["running"] is False
    assert "_hosted_room_task" not in session
