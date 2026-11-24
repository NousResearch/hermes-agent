"""The secure-input prompt belongs to the session whose agent asked for it.

Regression: the callback lived in one process-global, so in a multi-session TUI/desktop
gateway the session that wired callbacks last owned every other session's
prompt and received the value typed there.
"""

import json
import threading
from unittest.mock import patch

from agent import secure_input_broker as broker
from tools import secure_input_tool
from tools.secure_input_tool import request_secure_input, set_secure_input_callback


def setup_function():
    broker.clear_all()
    set_secure_input_callback(None)


def teardown_function():
    broker.clear_all()
    set_secure_input_callback(None)


def _run(fn):
    out = {}

    def target():
        try:
            out["value"] = fn()
        except BaseException as exc:  # surfaced by the assertion below
            out["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive()
    assert "error" not in out, out.get("error")
    return out.get("value")


def _wire_then_wait(callback, wired, go):
    """A turn thread: bind *callback*, signal, wait, then return a request."""
    def fn():
        set_secure_input_callback(callback)
        wired.set()
        go.wait(timeout=10)
        return json.loads(request_secure_input(purpose="github_token", title="token"))
    return fn


def test_request_uses_the_calling_turns_callback_not_the_last_wired_one():
    calls = []
    a_wired, b_wired, go = threading.Event(), threading.Event(), threading.Event()
    a_result = {}

    def turn_a():
        a_result.update(_wire_then_wait(lambda p, t, m=None: calls.append("A") or "value-typed-in-A",
                                        a_wired, go)())

    ta = threading.Thread(target=turn_a)
    ta.start()
    assert a_wired.wait(5)
    # Session B builds or starts a turn while A's turn is running.
    tb = threading.Thread(target=lambda: (set_secure_input_callback(
        lambda p, t, m=None: calls.append("B") or "value-typed-in-B"), b_wired.set()))
    tb.start()
    assert b_wired.wait(5)
    tb.join(5)
    go.set()
    ta.join(10)

    assert calls == ["A"]
    assert a_result["success"] is True
    assert broker.consume_secret(a_result["secret_ref"], consumer="terminal") == "value-typed-in-A"


def test_context_without_a_callback_fails_closed():
    calls = []
    set_secure_input_callback(lambda p, t, m=None: calls.append("other") or "borrowed")

    result = _run(lambda: json.loads(request_secure_input(purpose="github_token")))

    assert result["success"] is False
    assert calls == []
    # The tool stays advertised on an interactive surface.
    assert secure_input_tool.check_secure_input_requirements() is True


def test_tool_worker_threads_inherit_the_turns_callback():
    from tools.thread_context import propagate_context_to_thread

    def turn():
        set_secure_input_callback(lambda p, t, m=None: "value-from-turn")
        worker_out = {}
        worker = threading.Thread(target=propagate_context_to_thread(
            lambda: worker_out.update(json.loads(request_secure_input(purpose="npm_token")))))
        worker.start()
        worker.join(10)
        return worker_out

    result = _run(turn)
    assert result["success"] is True
    assert broker.consume_secret(result["secret_ref"], consumer="terminal") == "value-from-turn"


def test_tui_gateway_routes_secret_request_to_the_requesting_session(monkeypatch):
    from tests.tui_gateway.test_secret_capture_session import _gateway, _client
    server, requests, _ = _gateway(monkeypatch)
    frames, _ = _client(server, requests, monkeypatch, {"sess-A": "value-typed-in-sess-A"})
    monkeypatch.setitem(server._sessions, "sess-A", {"agent": object(), "session_key": "key-A"})
    monkeypatch.setitem(server._sessions, "sess-B", {"agent": object(), "session_key": "key-B"})
    a_wired, b_wired, go = threading.Event(), threading.Event(), threading.Event()
    a_result = {}

    def turn_a():
        tokens = server._set_session_context("key-A", ui_session_id="sess-A")
        server._wire_callbacks("sess-A")
        a_wired.set()
        go.wait(timeout=10)
        try:
            a_result.update(json.loads(request_secure_input(purpose="github_token", title="A token")))
        finally:
            server._clear_session_context(tokens)

    try:
        ta = threading.Thread(target=turn_a)
        ta.start()
        assert a_wired.wait(5)
        tb = threading.Thread(target=lambda: (server._wire_callbacks("sess-B"), b_wired.set()))
        tb.start()
        assert b_wired.wait(5)
        tb.join(5)
        go.set()
        ta.join(15)
    finally:
        from tools.terminal_tool import set_sudo_password_callback
        from tools.skills_tool import set_secret_capture_callback
        from tools.project_tools import set_project_workspace_callback
        set_sudo_password_callback(None)
        set_secret_capture_callback(None)
        set_project_workspace_callback(None)

    prompts = [frame["params"] for frame in frames if frame.get("method") == "secret"]
    assert [prompt["session_id"] for prompt in prompts] == ["sess-A"]
    assert a_result["success"] is True
    assert broker.consume_secret(a_result["secret_ref"], consumer="terminal") == "value-typed-in-sess-A"


def test_cli_agent_thread_reaches_the_cli_secure_prompt():
    """The CLI runs the agent on its own thread: binding the callback only on
    the caller thread (chat()) must not leave the agent failing closed."""
    import cli as cli_module
    from cli import HermesCLI

    clean_config = {
        "model": {"default": "anthropic/claude-opus-4.6", "base_url": "https://openrouter.ai/api/v1",
                  "provider": "auto"},
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    seen = {}

    class _FakeAgent:
        def run_conversation(self, **kwargs):
            seen["agent_thread"] = threading.current_thread().name
            seen["result"] = json.loads(request_secure_input(purpose="github_token"))
            return {"final_response": "ok", "messages": [], "api_calls": 0, "completed": True}

        def __getattr__(self, name):
            raise AttributeError(name)

    with patch("cli.get_tool_definitions", return_value=[]), patch.dict(
        "os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False
    ), patch.dict(cli_module.__dict__, {"CLI_CONFIG": clean_config}):
        cli_obj = HermesCLI()
        cli_obj._secure_input_callback = lambda purpose, prompt, metadata=None: "cli-typed-value"
        cli_obj.agent = _FakeAgent()
        cli_obj._active_agent_route_signature = "sig"
        route = {"signature": "sig", "model": None, "runtime": None}
        with patch.object(cli_obj, "_ensure_runtime_credentials", return_value=True), \
                patch.object(cli_obj, "_resolve_turn_agent_config", return_value=route), \
                patch.object(cli_obj, "_init_agent", return_value=True):
            try:
                cli_obj.chat("hello")
            except Exception:
                pass  # display plumbing after the turn is not under test

    assert seen.get("agent_thread") not in (None, threading.current_thread().name)
    assert seen["result"]["success"] is True, seen["result"]
    assert broker.consume_secret(seen["result"]["secret_ref"], consumer="terminal") == "cli-typed-value"
