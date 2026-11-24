"""Secure-input contracts with the toolset, terminal and UI transports."""
import json
import shlex
import sys
from pathlib import Path

import pytest

from agent import secure_input_broker as broker
from tools.secure_input_tool import request_secure_input, set_secure_input_callback


@pytest.fixture(autouse=True)
def clean_broker():
    broker.clear_all()
    set_secure_input_callback(None)
    yield
    broker.clear_all()
    set_secure_input_callback(None)


def test_surface_gate_is_session_scoped_not_a_cached_process_probe():
    from toolsets import session_disabled_toolsets, session_platform_tool_drops, resolve_toolset
    set_secure_input_callback(lambda *a: "synthetic-token")
    assert "request_secure_input" not in resolve_toolset("hermes-api-server")
    assert "request_secure_input" in session_platform_tool_drops("api_server")
    assert "request_secure_input" not in session_platform_tool_drops("desktop")
    assert "secure_input" in session_disabled_toolsets(None, "cli", oneshot=True)


def test_capture_error_never_echoes_transport_exception():
    def fail(*args):
        raise RuntimeError("synthetic-transport-secret")
    set_secure_input_callback(fail)
    result = request_secure_input(purpose="test")
    assert "synthetic-transport-secret" not in result
    assert not json.loads(result)["success"]


def test_dispatch_forwards_refs_without_reading_values(monkeypatch):
    from tools import terminal_tool as terminal
    seen = {}
    def run(**kwargs):
        seen.update(kwargs)
        return '{}'
    monkeypatch.setattr(terminal, "terminal_tool", run)
    terminal._handle_terminal({"command": "echo harmless", "stdin_secret_ref": "secret://session/opaque",
                               "env_secret_refs": {"DEMO": "secret://session/another"}})
    assert seen["stdin_secret_ref"] == "secret://session/opaque"
    assert seen["env_secret_refs"] == {"DEMO": "secret://session/another"}


def test_local_transient_env_is_not_in_snapshot_or_later_command(tmp_path):
    from tools.environments.local import LocalEnvironment
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        env.init_session()
        result = env.execute('printf "%s" "$D5_EPHEMERAL"', transient_env={"D5_EPHEMERAL": "synthetic-transient-value"})
        assert result["returncode"] == 0
        assert "synthetic-transient-value" in result["output"]
        assert "synthetic-transient-value" not in Path(env._snapshot_path).read_text()
        later = env.execute('printf "%s" "${D5_EPHEMERAL-missing}"')
        assert later["output"] == "missing"
    finally:
        env.cleanup()


def test_approval_denial_keeps_single_use_ref(tmp_path, monkeypatch):
    from tools import terminal_tool as terminal
    monkeypatch.setenv("TERMINAL_ENV", "local")
    ref = broker.register_secret("synthetic-denied-value", purpose="test")["secret_ref"]
    def refuse(*args, **kwargs):
        raise terminal._Rejected(terminal._error_json("denied", status="blocked"))
    monkeypatch.setattr(terminal, "_run_approval_guards", refuse)
    result = json.loads(terminal.terminal_tool("echo harmless", workdir=str(tmp_path), stdin_secret_ref=ref))
    assert result["status"] == "blocked"
    assert broker.consume_secret(ref, consumer="terminal") == "synthetic-denied-value"


@pytest.mark.parametrize("backend", ["ssh", "modal", "daytona", "vercel_sandbox", "unknown-plugin"])
def test_env_transport_gate_keeps_ref_without_backend_allocation(backend, monkeypatch):
    from tools import terminal_tool as terminal
    monkeypatch.setenv("TERMINAL_ENV", backend)
    def forbidden(*a):
        pytest.fail("unsupported secret transport allocated a backend")
    monkeypatch.setattr(terminal, "_acquire_env", forbidden)
    ref = broker.register_secret("synthetic-blocked-value", purpose="test")["secret_ref"]
    result = json.loads(terminal.terminal_tool("echo harmless", env_secret_refs={"DEMO": ref}))
    assert result["status"] == "blocked"
    assert broker.consume_secret(ref, consumer="terminal") == "synthetic-blocked-value"


def test_closed_tui_owner_cannot_create_a_prompt(monkeypatch):
    from tests.tui_gateway.test_secret_capture_session import _gateway
    server, _, _ = _gateway(monkeypatch)
    def forbidden(*args, **kw):
        pytest.fail("closed UI session was prompted")
    monkeypatch.setattr(server, "_ask", forbidden)
    tokens = server._set_session_context("closed-key", ui_session_id="closed-d5")
    try:
        server._wire_callbacks("closed-d5")
        result = json.loads(request_secure_input(purpose="test"))
        assert not result["success"]
    finally:
        server._clear_session_context(tokens)
        set_secure_input_callback(None)


def test_registry_capture_and_real_terminal_dispatch(tmp_path, monkeypatch):
    from tools.registry import registry
    monkeypatch.setenv("TERMINAL_ENV", "local")
    set_secure_input_callback(lambda *args: "synthetic-dispatched-value")
    captured = json.loads(registry.dispatch("request_secure_input", {"purpose": "test"}))
    assert captured["success"]
    assert "synthetic-dispatched-value" not in json.dumps(captured)
    command = shlex.quote(sys.executable) + " -c 'import sys; print(sys.stdin.read())'"
    result = json.loads(registry.dispatch("terminal", {"command": command, "workdir": str(tmp_path),
                            "stdin_secret_ref": captured["secret_ref"]}))
    assert result["exit_code"] == 0
    assert "[REDACTED SECURE INPUT]" in result["output"]
    assert "synthetic-dispatched-value" not in result["output"]


def test_tui_owner_closed_during_answer_cannot_register_secret(monkeypatch):
    from tests.tui_gateway.test_secret_capture_session import _gateway
    server, _, _ = _gateway(monkeypatch)
    sid = "d5-closing-owner"
    monkeypatch.setitem(server._sessions, sid, {"agent": object(), "session_key": "d5-closing-key"})
    def answer_then_close(*args, **kwargs):
        server._sessions.pop(sid)
        return "synthetic-late-answer"
    monkeypatch.setattr(server, "_ask", answer_then_close)
    tokens = server._set_session_context("d5-closing-key", ui_session_id=sid)
    try:
        server._wire_callbacks(sid)
        result = json.loads(request_secure_input(purpose="test"))
        assert not result["success"]
    finally:
        server._clear_session_context(tokens)
        set_secure_input_callback(None)
