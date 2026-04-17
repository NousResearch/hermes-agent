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
