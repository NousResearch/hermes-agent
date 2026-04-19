import json
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

from tui_gateway import server


class _ChunkyStdout:
    def __init__(self):
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        for ch in text:
            self.parts.append(ch)
            time.sleep(0.0001)
        return len(text)

    def flush(self) -> None:
        return None


class _BrokenStdout:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def test_write_json_serializes_concurrent_writes(monkeypatch):
    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)

    threads = [
        threading.Thread(target=server.write_json, args=({"seq": i, "text": "x" * 24},))
        for i in range(8)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    lines = "".join(out.parts).splitlines()

    assert len(lines) == 8
    assert {json.loads(line)["seq"] for line in lines} == set(range(8))


def test_write_json_returns_false_on_broken_pipe(monkeypatch):
    monkeypatch.setattr(server, "_real_stdout", _BrokenStdout())

    assert server.write_json({"ok": True}) is False


def test_status_callback_emits_kind_and_text():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("context_pressure", "85% to compaction")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "context_pressure", "text": "85% to compaction"},
    )


def test_status_callback_accepts_single_message_argument():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("thinking...")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "status", "text": "thinking..."},
    )


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
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
        **extra,
    }


def test_config_set_yolo_toggles_session_scope():
    from tools.approval import clear_session, is_session_yolo_enabled

    server._sessions["sid"] = _session()
    try:
        resp_on = server.handle_request({"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "yolo"}})
        assert resp_on["result"]["value"] == "1"
        assert is_session_yolo_enabled("session-key") is True

        resp_off = server.handle_request({"id": "2", "method": "config.set", "params": {"session_id": "sid", "key": "yolo"}})
        assert resp_off["result"]["value"] == "0"
        assert is_session_yolo_enabled("session-key") is False
    finally:
        clear_session("session-key")
        server._sessions.clear()


def test_enable_gateway_prompts_sets_gateway_env(monkeypatch):
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    server._enable_gateway_prompts()

    assert server.os.environ["HERMES_GATEWAY_SESSION"] == "1"
    assert server.os.environ["HERMES_EXEC_ASK"] == "1"
    assert server.os.environ["HERMES_INTERACTIVE"] == "1"


def test_setup_status_reports_provider_config(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: False)

    resp = server.handle_request({"id": "1", "method": "setup.status", "params": {}})

    assert resp["result"]["provider_configured"] is False


def test_config_set_reasoning_updates_live_session_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)

    resp_effort = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "reasoning", "value": "low"}}
    )
    assert resp_effort["result"]["value"] == "low"
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}

    resp_show = server.handle_request(
        {"id": "2", "method": "config.set", "params": {"session_id": "sid", "key": "reasoning", "value": "show"}}
    )
    assert resp_show["result"]["value"] == "show"
    assert server._sessions["sid"]["show_reasoning"] is True


def test_config_set_verbose_updates_session_mode_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(verbose_logging=False)
    server._sessions["sid"] = _session(agent=agent)

    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "verbose", "value": "cycle"}}
    )

    assert resp["result"]["value"] == "verbose"
    assert server._sessions["sid"]["tool_progress_mode"] == "verbose"
    assert agent.verbose_logging is True


def test_config_set_model_uses_live_switch_path(monkeypatch):
    server._sessions["sid"] = _session()
    seen = {}

    def _fake_apply(sid, session, raw):
        seen["args"] = (sid, session["session_key"], raw)
        return {"value": "new/model", "warning": "catalog unreachable"}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)
    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "model", "value": "new/model"}}
    )

    assert resp["result"]["value"] == "new/model"
    assert resp["result"]["warning"] == "catalog unreachable"
    assert seen["args"] == ("sid", "session-key", "new/model")


def test_config_set_model_global_persists(monkeypatch):
    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-old"

        def switch_model(self, **kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="anthropic/claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-new",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )
    seen = {}
    saved = {}

    def _switch_model(**kwargs):
        seen.update(kwargs)
        return result

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch_model)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.update(cfg))

    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "model", "value": "anthropic/claude-sonnet-4.6 --global"}}
    )

    assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
    assert seen["is_global"] is True
    assert saved["model"]["default"] == "anthropic/claude-sonnet-4.6"
    assert saved["model"]["provider"] == "anthropic"
    assert saved["model"]["base_url"] == "https://api.anthropic.com"


def test_config_set_personality_rejects_unknown_name(monkeypatch):
    monkeypatch.setattr(server, "_available_personalities", lambda cfg=None: {"helpful": "You are helpful."})
    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"key": "personality", "value": "bogus"}}
    )

    assert "error" in resp
    assert "Unknown personality" in resp["error"]["message"]


def test_config_set_personality_resets_history_and_returns_info(monkeypatch):
    session = _session(agent=types.SimpleNamespace(), history=[{"role": "user", "text": "hi"}], history_version=4)
    new_agent = types.SimpleNamespace(model="x")
    emits = []

    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_available_personalities", lambda cfg=None: {"helpful": "You are helpful."})
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None: new_agent)
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": getattr(agent, "model", "?")})
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    monkeypatch.setattr(server, "_write_config_key", lambda path, value: None)

    resp = server.handle_request(
        {"id": "1", "method": "config.set", "params": {"session_id": "sid", "key": "personality", "value": "helpful"}}
    )

    assert resp["result"]["history_reset"] is True
    assert resp["result"]["info"] == {"model": "x"}
    assert session["history"] == []
    assert session["history_version"] == 5
    assert ("session.info", "sid", {"model": "x"}) in emits


def test_session_compress_uses_compress_helper(monkeypatch):
    agent = types.SimpleNamespace()
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(server, "_compress_session_history", lambda session, focus_topic=None: (2, {"total": 42}))
    monkeypatch.setattr(server, "_session_info", lambda _agent: {"model": "x"})

    with patch("tui_gateway.server._emit") as emit:
        resp = server.handle_request({"id": "1", "method": "session.compress", "params": {"session_id": "sid"}})

    assert resp["result"]["removed"] == 2
    assert resp["result"]["usage"]["total"] == 42
    emit.assert_called_once_with("session.info", "sid", {"model": "x"})


def test_prompt_submit_sets_approval_session_key(monkeypatch):
    from tools.approval import get_current_session_key

    captured = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            captured["session_key"] = get_current_session_key(default="")
            return {"final_response": "ok", "messages": [{"role": "assistant", "content": "ok"}]}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    resp = server.handle_request({"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "ping"}})

    assert resp["result"]["status"] == "streaming"
    assert captured["session_key"] == "session-key"


def test_prompt_submit_expands_context_refs(monkeypatch):
    captured = {}

    class _Agent:
        model = "test/model"
        base_url = ""
        api_key = ""

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            captured["prompt"] = prompt
            return {"final_response": "ok", "messages": [{"role": "assistant", "content": "ok"}]}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    fake_ctx = types.ModuleType("agent.context_references")
    fake_ctx.preprocess_context_references = lambda message, **kwargs: types.SimpleNamespace(
        blocked=False, message="expanded prompt", warnings=[], references=[], injected_tokens=0
    )
    fake_meta = types.ModuleType("agent.model_metadata")
    fake_meta.get_model_context_length = lambda *args, **kwargs: 100000

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setitem(sys.modules, "agent.context_references", fake_ctx)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", fake_meta)

    server.handle_request({"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "@diff"}})

    assert captured["prompt"] == "expanded prompt"


def test_image_attach_appends_local_image(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png"}
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: Path("/tmp/cat.png")

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request({"id": "1", "method": "image.attach", "params": {"session_id": "sid", "path": "/tmp/cat.png"}})

    assert resp["result"]["attached"] is True
    assert resp["result"]["name"] == "cat.png"
    assert len(server._sessions["sid"]["attached_images"]) == 1


def test_commands_catalog_surfaces_quick_commands(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": {
        "build": {"type": "exec", "command": "npm run build"},
        "git": {"type": "alias", "target": "/shell git"},
        "notes": {"type": "exec", "command": "cat NOTES.md", "description": "Open design notes"},
    }})

    resp = server.handle_request({"id": "1", "method": "commands.catalog", "params": {}})

    pairs = dict(resp["result"]["pairs"])
    assert "npm run build" in pairs["/build"]
    assert pairs["/git"].startswith("alias →")
    assert pairs["/notes"] == "Open design notes"

    user_cat = next(c for c in resp["result"]["categories"] if c["name"] == "User commands")
    user_pairs = dict(user_cat["pairs"])
    assert set(user_pairs) == {"/build", "/git", "/notes"}

    assert resp["result"]["canon"]["/build"] == "/build"
    assert resp["result"]["canon"]["/notes"] == "/notes"


def test_command_dispatch_exec_nonzero_surfaces_error(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": {"boom": {"type": "exec", "command": "boom"}}})
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="failed"),
    )

    resp = server.handle_request({"id": "1", "method": "command.dispatch", "params": {"name": "boom"}})

    assert "error" in resp
    assert "failed" in resp["error"]["message"]


def test_plugins_list_surfaces_loader_error(monkeypatch):
    with patch("hermes_cli.plugins.get_plugin_manager", side_effect=Exception("boom")):
        resp = server.handle_request({"id": "1", "method": "plugins.list", "params": {}})

    assert "error" in resp
    assert "boom" in resp["error"]["message"]


def test_complete_slash_surfaces_completer_error(monkeypatch):
    with patch("hermes_cli.commands.SlashCommandCompleter", side_effect=Exception("no completer")):
        resp = server.handle_request({"id": "1", "method": "complete.slash", "params": {"text": "/mo"}})

    assert "error" in resp
    assert "no completer" in resp["error"]["message"]


def test_input_detect_drop_attaches_image(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: {
        "path": Path("/tmp/cat.png"),
        "is_image": True,
        "remainder": "",
    }

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request(
        {"id": "1", "method": "input.detect_drop", "params": {"session_id": "sid", "text": "/tmp/cat.png"}}
    )

    assert resp["result"]["matched"] is True
    assert resp["result"]["is_image"] is True
    assert resp["result"]["text"] == "[User attached image: cat.png]"


def test_rollback_restore_resolves_number_and_file_path():
    calls = {}

    class _Mgr:
        enabled = True

        def list_checkpoints(self, cwd):
            return [{"hash": "aaa111"}, {"hash": "bbb222"}]

        def restore(self, cwd, target, file_path=None):
            calls["args"] = (cwd, target, file_path)
            return {"success": True, "message": "done"}

    server._sessions["sid"] = _session(agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()), history=[])
    resp = server.handle_request(
        {
            "id": "1",
            "method": "rollback.restore",
            "params": {"session_id": "sid", "hash": "2", "file_path": "src/app.tsx"},
        }
    )

    assert resp["result"]["success"] is True
    assert calls["args"][1] == "bbb222"
    assert calls["args"][2] == "src/app.tsx"


# ── session.steer ────────────────────────────────────────────────────


def test_session_steer_calls_agent_steer_when_agent_supports_it():
    """The TUI RPC method must call agent.steer(text) and return a
    queued status without touching interrupt state.
    """
    calls = {}

    class _Agent:
        def steer(self, text):
            calls["steer_text"] = text
            return True

        def interrupt(self, *args, **kwargs):
            calls["interrupt_called"] = True

    server._sessions["sid"] = _session(agent=_Agent())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "also check auth.log"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "result" in resp, resp
    assert resp["result"]["status"] == "queued"
    assert resp["result"]["text"] == "also check auth.log"
    assert calls["steer_text"] == "also check auth.log"
    assert "interrupt_called" not in calls  # must NOT interrupt


def test_session_steer_rejects_empty_text():
    server._sessions["sid"] = _session(agent=types.SimpleNamespace(steer=lambda t: True))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "   "},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "error" in resp, resp
    assert resp["error"]["code"] == 4002


def test_session_steer_errors_when_agent_has_no_steer_method():
    server._sessions["sid"] = _session(agent=types.SimpleNamespace())  # no steer()
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "error" in resp, resp
    assert resp["error"]["code"] == 4010


def test_session_info_includes_mcp_servers(monkeypatch):
    fake_status = [
        {"name": "github", "transport": "http", "tools": 12, "connected": True},
        {"name": "filesystem", "transport": "stdio", "tools": 4, "connected": True},
        {"name": "broken", "transport": "stdio", "tools": 0, "connected": False},
    ]
    fake_mod = types.ModuleType("tools.mcp_tool")
    fake_mod.get_mcp_status = lambda: fake_status
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", fake_mod)

    info = server._session_info(types.SimpleNamespace(tools=[], model=""))

    assert info["mcp_servers"] == fake_status


# ---------------------------------------------------------------------------
# History-mutating commands must reject while session.running is True.
# Without these guards, prompt.submit's post-run history write either
# clobbers the mutation (version matches) or silently drops the agent's
# output (version mismatch) — both produce UI<->backend state desync.
# ---------------------------------------------------------------------------


def test_session_undo_rejects_while_running():
    """Fix for TUI silent-drop #1: /undo must not mutate history
    while the agent is mid-turn — would either clobber the undo or
    cause prompt.submit to silently drop the agent's response."""
    server._sessions["sid"] = _session(running=True, history=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.undo", "params": {"session_id": "sid"}}
        )
        assert resp.get("error"), "session.undo should reject while running"
        assert resp["error"]["code"] == 4009
        assert "session busy" in resp["error"]["message"]
        # History must be unchanged
        assert len(server._sessions["sid"]["history"]) == 2
    finally:
        server._sessions.pop("sid", None)


def test_session_undo_allowed_when_idle():
    """Regression guard: when not running, /undo still works."""
    server._sessions["sid"] = _session(running=False, history=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.undo", "params": {"session_id": "sid"}}
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert resp["result"]["removed"] == 2
        assert server._sessions["sid"]["history"] == []
    finally:
        server._sessions.pop("sid", None)


def test_session_compress_rejects_while_running(monkeypatch):
    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )
        assert resp.get("error")
        assert resp["error"]["code"] == 4009
    finally:
        server._sessions.pop("sid", None)


def test_rollback_restore_rejects_full_history_while_running(monkeypatch):
    """Full-history rollback must reject; file-scoped rollback still allowed."""
    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "rollback.restore", "params": {"session_id": "sid", "hash": "abc"}}
        )
        assert resp.get("error"), "full-history rollback should reject while running"
        assert resp["error"]["code"] == 4009
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_history_version_mismatch_surfaces_warning(monkeypatch):
    """Fix for TUI silent-drop #2: the defensive backstop at prompt.submit
    must attach a 'warning' to message.complete when history was
    mutated externally during the turn (instead of silently dropping
    the agent's output)."""
    # Agent bumps history_version itself mid-run to simulate an external
    # mutation slipping past the guards.
    session_ref = {"s": None}

    class _RacyAgent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            # Simulate: something external bumped history_version
            # while we were running.
            with session_ref["s"]["history_lock"]:
                session_ref["s"]["history_version"] += 1
            return {"final_response": "agent reply", "messages": [{"role": "assistant", "content": "agent reply"}]}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_RacyAgent())
    session_ref["s"] = server._sessions["sid"]
    emits: list[tuple] = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        resp = server.handle_request(
            {"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hi"}}
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        # History should NOT contain the agent's output (version mismatch)
        assert server._sessions["sid"]["history"] == []

        # message.complete must carry a 'warning' so the UI / operator
        # knows the output was not persisted.
        complete_calls = [a for a in emits if a[0] == "message.complete"]
        assert len(complete_calls) == 1
        _, _, payload = complete_calls[0]
        assert "warning" in payload, (
            "message.complete must include a 'warning' field on "
            "history_version mismatch — otherwise the UI silently "
            "shows output that was never persisted"
        )
        assert "not saved" in payload["warning"].lower() or "changed" in payload["warning"].lower()
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_history_version_match_persists_normally(monkeypatch):
    """Regression guard: the backstop does not affect the happy path."""
    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            return {"final_response": "reply", "messages": [{"role": "assistant", "content": "reply"}]}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_Agent())
    emits: list[tuple] = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        resp = server.handle_request(
            {"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hi"}}
        )
        assert resp.get("result")

        # History was written
        assert server._sessions["sid"]["history"] == [{"role": "assistant", "content": "reply"}]
        assert server._sessions["sid"]["history_version"] == 1

        # No warning should be attached
        complete_calls = [a for a in emits if a[0] == "message.complete"]
        assert len(complete_calls) == 1
        _, _, payload = complete_calls[0]
        assert "warning" not in payload
    finally:
        server._sessions.pop("sid", None)


# ---------------------------------------------------------------------------
# session.interrupt must only cancel pending prompts owned by the calling
# session — it must not blast-resolve clarify/sudo/secret prompts on
# unrelated sessions sharing the same tui_gateway process.  Without
# session scoping the other sessions' prompts silently resolve to empty
# strings, unblocking their agent threads as if the user cancelled.
# ---------------------------------------------------------------------------


def test_interrupt_only_clears_own_session_pending():
    """session.interrupt on session A must NOT release pending prompts
    that belong to session B."""
    import types

    session_a = _session()
    session_a["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    session_b = _session()
    session_b["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    server._sessions["sid_a"] = session_a
    server._sessions["sid_b"] = session_b

    try:
        # Simulate pending prompts on both sessions (what _block creates
        # while a clarify/sudo/secret request is outstanding).
        ev_a = threading.Event()
        ev_b = threading.Event()
        server._pending["rid-a"] = ("sid_a", ev_a)
        server._pending["rid-b"] = ("sid_b", ev_b)
        server._answers.clear()

        # Interrupt session A.
        resp = server.handle_request(
            {"id": "1", "method": "session.interrupt", "params": {"session_id": "sid_a"}}
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        # Session A's pending must be released to empty.
        assert ev_a.is_set(), "sid_a pending Event should be set after interrupt"
        assert server._answers.get("rid-a") == ""

        # Session B's pending MUST remain untouched — no cross-session blast.
        assert not ev_b.is_set(), (
            "CRITICAL: session.interrupt on sid_a released a pending prompt "
            "belonging to sid_b — other sessions' clarify/sudo/secret "
            "prompts are being silently cancelled"
        )
        assert "rid-b" not in server._answers
    finally:
        server._sessions.pop("sid_a", None)
        server._sessions.pop("sid_b", None)
        server._pending.pop("rid-a", None)
        server._pending.pop("rid-b", None)
        server._answers.pop("rid-a", None)
        server._answers.pop("rid-b", None)


def test_interrupt_clears_multiple_own_pending():
    """When a single session has multiple pending prompts (uncommon but
    possible via nested tool calls), interrupt must release all of them."""
    import types

    sess = _session()
    sess["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    server._sessions["sid"] = sess

    try:
        ev1, ev2 = threading.Event(), threading.Event()
        server._pending["r1"] = ("sid", ev1)
        server._pending["r2"] = ("sid", ev2)

        resp = server.handle_request(
            {"id": "1", "method": "session.interrupt", "params": {"session_id": "sid"}}
        )
        assert resp.get("result")
        assert ev1.is_set() and ev2.is_set()
        assert server._answers.get("r1") == "" and server._answers.get("r2") == ""
    finally:
        server._sessions.pop("sid", None)
        for key in ("r1", "r2"):
            server._pending.pop(key, None)
            server._answers.pop(key, None)


def test_clear_pending_without_sid_clears_all():
    """_clear_pending(None) is the shutdown path — must still release
    every pending prompt regardless of owning session."""
    ev1, ev2, ev3 = threading.Event(), threading.Event(), threading.Event()
    server._pending["a"] = ("sid_x", ev1)
    server._pending["b"] = ("sid_y", ev2)
    server._pending["c"] = ("sid_z", ev3)
    try:
        server._clear_pending(None)
        assert ev1.is_set() and ev2.is_set() and ev3.is_set()
    finally:
        for key in ("a", "b", "c"):
            server._pending.pop(key, None)
            server._answers.pop(key, None)


def test_respond_unpacks_sid_tuple_correctly():
    """After the (sid, Event) tuple change, _respond must still work."""
    ev = threading.Event()
    server._pending["rid-x"] = ("sid_x", ev)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "clarify.respond",
             "params": {"request_id": "rid-x", "answer": "the answer"}}
        )
        assert resp.get("result")
        assert ev.is_set()
        assert server._answers.get("rid-x") == "the answer"
    finally:
        server._pending.pop("rid-x", None)
        server._answers.pop("rid-x", None)

