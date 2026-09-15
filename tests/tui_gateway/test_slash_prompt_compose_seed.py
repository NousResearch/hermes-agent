"""Desktop slash.exec must compose via /prompt and /compose, not view the system prompt.

/prompt was live-answered as a system-prompt viewer; /compose ran in the slash
worker, set ``_pending_agent_seed``, and returned empty stdout so desktop
showed ``(no output)``. The worker must surface a non-empty seed and
slash.exec must map it to ``{type: send, message}``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from tui_gateway import server
from tui_gateway.transport import StdioTransport


class _SeedCLI:
    console = None

    def __init__(self, seed):
        self._pending_agent_seed = None
        self._seed_to_set = seed

    def process_command(self, cmd: str) -> None:
        self._pending_agent_seed = self._seed_to_set


class _WorkerReply:
    def __init__(self, raw):
        self.raw = raw

    def run(self, command: str):
        return self.raw


def _session_with_secret_prompt():
    agent = SimpleNamespace(
        ephemeral_system_prompt="SECRET SYS",
        _cached_system_prompt="SECRET SYS",
    )
    return {
        "agent": agent,
        "session_key": "compose-key",
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "transport": StdioTransport(lambda: None, threading.Lock()),
        "cwd": "",
        "source": "desktop",
    }


def _unpack_run(raw):
    """Normalize the worker ``_run`` protocol to ``(output, seed)``."""
    if isinstance(raw, tuple) and len(raw) >= 2:
        return raw[0], raw[1]
    if isinstance(raw, dict):
        return raw.get("output", ""), raw.get("pending_agent_seed")
    raise AssertionError(
        f"slash_worker._run must surface (output, pending_agent_seed), got {type(raw).__name__}: {raw!r}"
    )


def test_prompt_is_not_live_answered_as_system_prompt_viewer():
    sid = "compose-sid"
    session = _session_with_secret_prompt()
    out = server._live_slash_command_output(sid, session, "prompt", "")
    assert out is None
    assert server._live_slash_command_output(sid, session, "compose", "") is None


def test_live_slash_output_drops_prompt_keeps_other_names():
    names = set(server._LIVE_SLASH_OUTPUT)
    assert "prompt" not in names
    assert "compose" not in names
    assert names >= {
        "usage", "help", "history", "context", "tools", "status",
        "review", "compress", "clear", "models", "rename", "effort",
    }
    assert "prompt" not in server._SLASH_MIRRORS
    assert "prompt" not in server._MUTATES_WHILE_RUNNING


def test_run_surfaces_and_consumes_pending_agent_seed():
    from tui_gateway import slash_worker

    cli = _SeedCLI("composed turn")
    output, seed = _unpack_run(slash_worker._run(cli, "/compose"))
    assert seed == "composed turn"
    assert cli._pending_agent_seed in (None, "")
    assert output == ""


def test_run_surfaces_prompt_alias_seed():
    from tui_gateway import slash_worker

    cli = _SeedCLI("composed via prompt")
    _, seed = _unpack_run(slash_worker._run(cli, "/prompt"))
    assert seed == "composed via prompt"
    assert cli._pending_agent_seed in (None, "")


def test_run_empty_seed_is_fail_open():
    from tui_gateway import slash_worker

    for empty in (None, ""):
        cli = _SeedCLI(empty)
        _, seed = _unpack_run(slash_worker._run(cli, "/compose"))
        assert not seed


def test_slash_exec_maps_pending_seed_to_type_send():
    sid = "compose-exec-sid"
    session = _session_with_secret_prompt()
    session["slash_worker"] = _WorkerReply({"output": "", "pending_agent_seed": "composed turn"})
    server._sessions[sid] = session
    try:
        resp = server._methods["slash.exec"](
            1, {"session_id": sid, "command": "/compose"},
        )
    finally:
        server._sessions.pop(sid, None)
    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "composed turn"


def test_slash_exec_maps_prompt_alias_seed_to_type_send():
    sid = "prompt-exec-sid"
    session = _session_with_secret_prompt()
    session["slash_worker"] = _WorkerReply({"output": "", "pending_agent_seed": "composed turn"})
    server._sessions[sid] = session
    try:
        resp = server._methods["slash.exec"](
            1, {"session_id": sid, "command": "/prompt"},
        )
    finally:
        server._sessions.pop(sid, None)
    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "composed turn"


def test_slash_exec_empty_seed_is_not_a_send():
    sid = "empty-exec-sid"
    session = _session_with_secret_prompt()
    session["slash_worker"] = _WorkerReply({"output": "", "pending_agent_seed": None})
    server._sessions[sid] = session
    try:
        resp = server._methods["slash.exec"](
            1, {"session_id": sid, "command": "/compose"},
        )
    finally:
        server._sessions.pop(sid, None)
    assert "error" not in resp
    result = resp["result"]
    assert result.get("type") != "send"
    assert "message" not in result or not result.get("message")
