"""Per-message system-context support for ``tui_gateway.prompt.submit``."""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import methods_prompt, server


class _InlineThread:
    """Run a prompt.submit dispatch closure synchronously in tests."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False


def _session(agent, **extra):
    return {
        "agent": agent,
        "session_key": "gateway-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "transport": None,
        **extra,
    }


def _stub_turn_runtime(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})


@pytest.mark.parametrize(
    ("params", "expected", "error"),
    [
        ({}, None, None),
        ({"system_message": None}, None, "system_message must be a string"),
        (
            {"system_message": None, "instructions": "fallback instruction"},
            None,
            "system_message must be a string",
        ),
        ({"instructions": None}, None, "instructions must be a string"),
        ({"system_message": "  answer in JSON  "}, "  answer in JSON  ", None),
        ({"instructions": "use metric units"}, "use metric units", None),
        (
            {"system_message": "system wins", "instructions": "ignored"},
            "system wins",
            None,
        ),
        (
            {"system_message": "", "instructions": "fallback instruction"},
            None,
            "system_message must not be blank",
        ),
        (
            {"system_message": [], "instructions": "fallback instruction"},
            None,
            "system_message must be a string",
        ),
        (
            {"system_message": "   ", "instructions": "fallback instruction"},
            None,
            "system_message must not be blank",
        ),
        ({"instructions": ["not", "text"]}, None, "instructions must be a string"),
        ({"instructions": "   "}, None, "instructions must not be blank"),
    ],
)
def test_prompt_extra_system_parses_prompt_submit_aliases(params, expected, error):
    """Both public aliases normalize to one optional, one-turn string."""
    assert methods_prompt._prompt_extra_system(params) == (expected, error)


def test_prompt_submit_forwards_extra_system_to_turn_runner(monkeypatch):
    """The JSON-RPC method forwards the normalized alias to its runner."""
    sid = "live-session"
    session = _session(types.SimpleNamespace())
    captured = {}
    server._sessions[sid] = session
    try:
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_: None)
        monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_: False)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_: True)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda *_: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *_: None)
        monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_: None)
        monkeypatch.setattr(server, "_start_inflight_turn", lambda *_: None)
        monkeypatch.setattr(server.threading, "Thread", _InlineThread)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda _rid, _sid, _session, _text, **kwargs: captured.update(kwargs),
        )

        response = server._methods["prompt.submit"](
            "request-id",
            {
                "session_id": sid,
                "text": "hello",
                "system_message": "Reply using terse YAML.",
                "instructions": "This alias must lose to system_message.",
            },
        )

        assert response["result"]["status"] == "streaming"
        assert captured["extra_system"] == "Reply using terse YAML."
    finally:
        server._sessions.pop(sid, None)


def test_extra_system_is_visible_only_during_its_turn(monkeypatch, tmp_path):
    """A one-off override augments the current call but never leaks afterward."""
    seen_system_prompts = []
    agent = types.SimpleNamespace(
        session_id="agent-session",
        ephemeral_system_prompt="Keep responses professional.",
        clear_interrupt=lambda: None,
    )

    def run_conversation(*_args, **_kwargs):
        seen_system_prompts.append(agent.ephemeral_system_prompt)
        return {"final_response": "done"}

    agent.run_conversation = run_conversation
    session = _session(agent, running=True)

    _stub_turn_runtime(monkeypatch, tmp_path)

    server._run_prompt_submit(
        "request-id",
        "ui-session",
        session,
        "hello",
        extra_system="This response must be valid YAML.",
    )

    assert seen_system_prompts == [
        "Keep responses professional.\n\nThis response must be valid YAML."
    ]
    assert agent.ephemeral_system_prompt == "Keep responses professional."


def test_extra_system_restores_after_internal_prompt_normalization(monkeypatch, tmp_path):
    """Provider retry normalization must not make a one-turn overlay persist."""
    agent = types.SimpleNamespace(
        session_id="agent-session",
        ephemeral_system_prompt="Respond in café style.",
        clear_interrupt=lambda: None,
    )

    def run_conversation(*_args, **_kwargs):
        agent.ephemeral_system_prompt = "Respond in caf style.\n\nUse rsum format."
        return {"final_response": "done"}

    agent.run_conversation = run_conversation
    session = _session(agent, running=True)
    _stub_turn_runtime(monkeypatch, tmp_path)

    server._run_prompt_submit(
        "request-id",
        "ui-session",
        session,
        "hello",
        extra_system="Use résumé format.",
    )

    assert agent.ephemeral_system_prompt == "Respond in café style."


def test_extra_system_preserves_mid_turn_personality_pivot(monkeypatch, tmp_path):
    """Immediate personality changes must outlive a one-turn system overlay."""
    seen_system_prompts = []
    agent = types.SimpleNamespace(
        session_id="agent-session",
        ephemeral_system_prompt="Keep responses professional.",
        clear_interrupt=lambda: None,
    )
    session = _session(agent, running=True)

    def run_conversation(*_args, **_kwargs):
        seen_system_prompts.append(agent.ephemeral_system_prompt)
        server._apply_personality_to_session(
            "ui-session",
            session,
            "Answer tersely.",
            "terse",
        )
        return {"final_response": "done"}

    agent.run_conversation = run_conversation
    _stub_turn_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})

    server._run_prompt_submit(
        "request-id",
        "ui-session",
        session,
        "hello",
        extra_system="This response must be valid YAML.",
    )

    assert seen_system_prompts == [
        "Keep responses professional.\n\nThis response must be valid YAML."
    ]
    assert agent.ephemeral_system_prompt == "Answer tersely."

def test_busy_submit_queues_its_extra_system_without_merging(monkeypatch):
    """Distinct one-turn instructions retain FIFO boundaries while a turn runs."""
    sid = "busy-session"
    session = _session(types.SimpleNamespace(), running=True)
    server._sessions[sid] = session
    try:
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_: None)

        first = server._methods["prompt.submit"](
            "first-request",
            {
                "session_id": sid,
                "text": "first follow-up",
                "instructions": "Answer the first request as CSV.",
                "queued": True,
            },
        )
        second = server._methods["prompt.submit"](
            "second-request",
            {
                "session_id": sid,
                "text": "second follow-up",
                "instructions": "Answer the second request as JSON.",
                "queued": True,
            },
        )

        assert first["result"]["status"] == "queued"
        assert second["result"]["status"] == "queued"
        assert session["queued_prompt"]["extra_system"] == (
            "Answer the first request as CSV."
        )
        assert session["queued_prompts"][0]["extra_system"] == (
            "Answer the second request as JSON."
        )
    finally:
        server._sessions.pop(sid, None)


def test_same_text_with_extra_system_is_not_deduplicated():
    """An override makes an otherwise duplicate prompt a distinct request."""
    session = _session(
        types.SimpleNamespace(),
        inflight_turn={"user": "repeat this prompt"},
    )

    server._enqueue_prompt(
        session,
        "repeat this prompt",
        None,
        extra_system="Return the result as JSON.",
    )
    server._drop_queued_duplicates_of_inflight_user(session)

    assert session["queued_prompt"] == {
        "text": "repeat this prompt",
        "transport": None,
        "extra_system": "Return the result as JSON.",
    }


def test_queued_extra_system_reaches_the_next_turn_runner(monkeypatch):
    """Queue drain preserves the override until the queued turn actually starts."""
    captured = {}
    session = _session(
        types.SimpleNamespace(),
        queued_prompt={
            "text": "queued follow-up",
            "transport": None,
            "extra_system": "Return only XML.",
        },
    )
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_: False)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, _text, **kwargs: captured.update(kwargs),
    )

    assert server._drain_queued_prompt("request-id", "ui-session", session) is True
    assert captured["extra_system"] == "Return only XML."


def test_compute_host_frame_carries_extra_system(monkeypatch):
    """Isolated turns receive the same one-turn override as in-process turns."""
    session = _session(types.SimpleNamespace(), history=[{"role": "user", "content": "hi"}])
    monkeypatch.setattr(server, "_session_cwd", lambda _session: "/tmp/workspace")
    monkeypatch.setattr(server, "_context_cwd_is_launch_artifact", lambda _session: False)
    monkeypatch.setattr(server, "_session_source", lambda _session: "desktop")

    frame = server._compute_host_turn_frame(
        "request-id",
        "ui-session",
        session,
        "hello",
        extra_system="Respond in one sentence.",
    )

    assert frame["extra_system"] == "Respond in one sentence."


def test_extra_system_never_clobbers_mid_turn_prompt_reload(monkeypatch, tmp_path):
    """A mid-turn /prompt reload must outlive a one-turn system overlay."""
    agent = types.SimpleNamespace(
        session_id="agent-session",
        ephemeral_system_prompt="Keep responses professional.",
        clear_interrupt=lambda: None,
    )
    session = _session(agent, running=True)

    def run_conversation(*_args, **_kwargs):
        server._mirror_prompt("ui-session", session, agent, "")
        return {"final_response": "done"}

    agent.run_conversation = run_conversation
    _stub_turn_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"agent": {"system_prompt": "Reloaded prompt."}})

    server._run_prompt_submit(
        "request-id",
        "ui-session",
        session,
        "hello",
        extra_system="This response must be valid YAML.",
    )

    assert agent.ephemeral_system_prompt == "Reloaded prompt."


def test_blank_system_message_is_rejected_at_the_rpc_boundary(monkeypatch):
    """A present-but-blank overlay key errors instead of silently degrading."""
    sid = "validation-session"
    session = _session(types.SimpleNamespace())
    server._sessions[sid] = session
    try:
        response = server._methods["prompt.submit"](
            "request-id",
            {"session_id": sid, "text": "x", "system_message": ""},
        )

        assert response["error"]["code"] == 4004
        assert response["error"]["message"] == "system_message must not be blank"
    finally:
        server._sessions.pop(sid, None)


def test_busy_steer_mode_queues_overlay_prompts_instead_of_steering(monkeypatch):
    """Steering would drop the one-turn system context, so overlay prompts queue."""
    sid = "steer-session"
    steer_calls = []
    agent = types.SimpleNamespace(steer=lambda *a, **k: steer_calls.append((a, k)))
    session = _session(agent, running=True)
    server._sessions[sid] = session
    try:
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_: None)
        monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")

        response = server._methods["prompt.submit"](
            "request-id",
            {
                "session_id": sid,
                "text": "mid-turn follow-up",
                "instructions": "Answer as CSV.",
            },
        )

        assert response["result"]["status"] == "queued"
        assert session["queued_prompt"]["extra_system"] == "Answer as CSV."
        assert steer_calls == []
    finally:
        server._sessions.pop(sid, None)


def test_drain_forwards_extra_system_to_the_compute_host(monkeypatch):
    """Isolated-worker drains carry the override exactly like in-process drains."""
    captured = {}
    session = _session(
        types.SimpleNamespace(),
        queued_prompt={
            "text": "queued follow-up",
            "transport": None,
            "extra_system": "Return only XML.",
        },
    )
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_: True)

    def submit_to_host(_rid, _sid, _session, _text, **kwargs):
        captured.update(kwargs)
        return {"result": {}}

    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", submit_to_host)

    assert server._drain_queued_prompt("request-id", "ui-session", session) is True
    assert captured["extra_system"] == "Return only XML."
