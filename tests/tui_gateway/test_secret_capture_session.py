"""Skill credential prompts go to the session that owns the turn, never the last wired one.

``set_secret_capture_callback`` holds one process-global callback, so the closure sid is just
whichever session called ``_wire_callbacks`` last. The prompt recipient has to be the UI owner
bound by ``_set_session_context``: the same session whose profile scope the answer is saved into.
Without a bound owner there is nobody to ask, so the prompt is skipped rather than guessed.
"""

import json
import sys
import threading

import pytest


def _gateway(monkeypatch):
    """Import the real gateway after neutralizing process-wide import side effects."""
    from hermes_cli import banner

    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    from agent.vault_backends import unlock
    from tools import project_tools, skills_tool, terminal_tool, terminal_tool_sudo
    from tui_gateway import server, server_requests

    monkeypatch.setattr(terminal_tool, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_current_session_tls", threading.local())
    monkeypatch.setattr(project_tools, "_workspace_callback", None)
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", None)
    monkeypatch.setattr(terminal_tool_sudo, "_sudo_password_cache", {})
    monkeypatch.delenv("HERMES_UI_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    return server, server_requests, skills_tool


def _client(server, server_requests, monkeypatch, values):
    """Stand in for the renderer: record every frame and answer each ``secret`` request with the
    value its recipient would type. Returns ``(frames, completed)``; ``completed`` is set when a
    ``background.complete`` event arrives."""
    frames, completed = [], threading.Event()

    def write(frame):
        frames.append(frame)
        if frame.get("method") == "secret":
            sid = frame["params"]["session_id"]
            assert server_requests.resolve_response(
                {"jsonrpc": "2.0", "id": frame["id"], "result": {"value": values[sid]}}
            )
        elif (frame.get("params") or {}).get("type") == "background.complete":
            completed.set()
        return True

    monkeypatch.setattr(server, "write_json", write)
    return frames, completed


def _capture(server, server_requests, skills_tool, monkeypatch, *, values):
    """Drive the real secret ask and record which session received it and what was stored."""
    stored = []

    def save(key, value):
        stored.append((key, value))
        return {"success": True, "stored_as": key, "validated": False}

    frames, _ = _client(server, server_requests, monkeypatch, values)
    monkeypatch.setattr("hermes_cli.config.save_env_value_secure", save)
    result = skills_tool._capture_required_environment_variables(
        "demo-skill", [{"name": "DEMO_TOKEN", "prompt": "Token"}]
    )
    return frames, stored, result


def test_secret_prompt_goes_to_active_turn_not_last_wired_session(monkeypatch):
    """While session A's turn is active, a prompt must not land on the closure sid."""
    from gateway.session_context import get_session_env

    server, server_requests, skills_tool = _gateway(monkeypatch)
    server._wire_callbacks("session-A")
    server._wire_callbacks("session-B")  # replaces the process-global callback

    tokens = server._set_session_context("turn-A", ui_session_id="session-A")
    try:
        assert get_session_env("HERMES_UI_SESSION_ID") == "session-A"
        frames, stored, result = _capture(
            server, server_requests, skills_tool, monkeypatch,
            values={"session-A": "owner-secret", "session-B": "closure-secret"},
        )
    finally:
        server._clear_session_context(tokens)
        server_requests.reset_for_tests()

    assert [frame["method"] for frame in frames] == ["secret"]
    assert frames[0]["params"]["session_id"] == "session-A"
    assert stored == [("DEMO_TOKEN", "owner-secret")]
    assert result == {"missing_names": [], "setup_skipped": False, "gateway_setup_hint": None}
    assert server_requests.open_requests("session-A") == []
    assert server_requests.open_requests("session-B") == []


def test_secret_prompt_without_bound_owner_is_skipped_not_guessed(monkeypatch):
    """No UI owner in context: the wired sid is not ownership evidence, so nobody is asked."""
    from gateway.session_context import get_session_env

    server, server_requests, skills_tool = _gateway(monkeypatch)
    server._wire_callbacks("only-session")

    tokens = server._set_session_context("ownerless-task", cwd="")
    try:
        assert get_session_env("HERMES_UI_SESSION_ID") == ""
        frames, stored, result = _capture(
            server, server_requests, skills_tool, monkeypatch,
            values={"only-session": "guessed-secret"},
        )
    finally:
        server._clear_session_context(tokens)
        server_requests.reset_for_tests()

    assert frames == []
    assert stored == []
    assert result == {"missing_names": ["DEMO_TOKEN"], "setup_skipped": True, "gateway_setup_hint": None}


_SKILL = """---
name: secret-skill
description: needs a token
required_environment_variables:
  - name: DEMO_TOKEN
    prompt: Demo token
---

# secret-skill
"""


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    homes = [launch / "profiles" / name for name in ("a", "b")]
    for home in homes:
        skill_dir = home / "skills" / "demo" / "secret-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    return launch, *homes


def _saved_tokens(home):
    env = home / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    return [line for line in lines if line.startswith("DEMO_TOKEN=")]


def test_background_prompt_secret_reaches_its_own_session_and_profile(two_profiles, monkeypatch):
    """prompt.background for A after B wired last: A is asked, and A's answer lands in A/.env.

    The real RPC, worker thread, profile scope, skill_view readiness path, request settlement and
    credential saver run; only the model boundary is replaced (it loads the skill as a tool call would).
    """
    launch, home_a, home_b = two_profiles
    server, server_requests, skills_tool = _gateway(monkeypatch)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    class ModelBoundary:
        def __init__(self, **_kw):
            pass

        def run_conversation(self, **_kw):
            view = json.loads(skills_tool.skill_view("secret-skill", preprocess=False))
            return {"final_response": view.get("readiness_status") or view.get("error")}

    monkeypatch.setattr("run_agent.AIAgent", ModelBoundary)
    monkeypatch.setattr(server, "_background_agent_kwargs", lambda _agent, _task_id: {})
    sessions = {
        sid: {"agent": object(), "session_key": f"key-{sid}", "profile_home": str(home), "cwd": str(home)}
        for sid, home in (("session-A", home_a), ("session-B", home_b))
    }
    frames, completed = _client(
        server, server_requests, monkeypatch, {"session-A": "a-secret", "session-B": "b-secret"})
    with server._sessions_lock:
        server._sessions.update(sessions)
    try:
        server._wire_callbacks("session-A")
        server._wire_callbacks("session-B")  # B owns the process-global callback now
        reply = server._methods["prompt.background"](
            "rid", {"session_id": "session-A", "text": "load secret-skill"})
        assert "error" not in reply, reply
        assert completed.wait(30), "background worker never completed"
    finally:
        with server._sessions_lock:
            for sid in sessions:
                server._sessions.pop(sid, None)
        server_requests.reset_for_tests()

    asks = [frame["params"] for frame in frames if frame.get("method") == "secret"]
    assert [(ask["session_id"], ask["env_var"]) for ask in asks] == [("session-A", "DEMO_TOKEN")]
    assert _saved_tokens(home_a) == ["DEMO_TOKEN=a-secret"]
    assert _saved_tokens(home_b) == []
    assert _saved_tokens(launch) == []
    done = [frame["params"] for frame in frames if (frame.get("params") or {}).get("type") == "background.complete"]
    assert [(event["session_id"], event["payload"]["text"]) for event in done] == [("session-A", "available")]
