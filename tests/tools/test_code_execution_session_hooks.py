"""Nested execute_code hooks retain the parent session identity (#51931)."""

import json

from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from tools.code_kernel import shutdown_all_kernels
from tools.registry import registry


def test_nested_tool_hooks_receive_each_cell_session(tmp_path, monkeypatch):
    seen = []
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="session-probe"), manager)
    context.register_hook("pre_tool_call", lambda **kw: seen.append(("pre", kw["tool_name"], kw["session_id"])))
    context.register_hook("post_tool_call", lambda **kw: seen.append(("post", kw["tool_name"], kw["session_id"])))
    manager._discovered = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr("tools.code_execution_tool._load_config", lambda: {"mode": "strict", "timeout": 30})
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    shutdown_all_kernels()
    try:
        for session in ("session-one", "session-two"):
            note = tmp_path / f"{session}.txt"
            note.write_text("nested content", encoding="utf-8")
            code = f"from hermes_tools import read_file\nprint(read_file({str(note)!r}))"
            raw = registry.dispatch("execute_code", {"code": code}, task_id="hook-test",
                                    enabled_tools=["read_file"], session_id=session)
            assert isinstance(raw, str)
            result = json.loads(raw)
            assert result["status"] == "success", result
            assert "nested content" in result["output"]
        assert seen == [(phase, "read_file", session) for session in ("session-one", "session-two")
                        for phase in ("pre", "post")]
    finally:
        shutdown_all_kernels()
        manager.unload()
