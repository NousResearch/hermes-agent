"""Regression coverage for declarative shell hooks in the stdio TUI."""

from __future__ import annotations

import io
from pathlib import Path

from agent import outbound_webhooks, shell_hooks
from hermes_cli import plugins
from hermes_cli import config as hermes_config
from tui_gateway import entry


def _run_main_to_eof(monkeypatch) -> None:
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "write_json", lambda payload: True)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *args: False)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))
    entry.main()


def test_main_registers_terminal_policy_before_accepting_requests(
    monkeypatch, tmp_path: Path,
):
    """The TUI must enforce the same configured pre-tool hook as CLI runs."""
    policy = tmp_path / "terminal-policy.py"
    policy.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "command = payload.get('tool_input', {}).get('command', '')\n"
        "allowed = command.startswith('agent-browser ') and not any(\n"
        "    token in command for token in ('&&', ';', '|')\n"
        ")\n"
        "if not allowed:\n"
        "    print(json.dumps({'decision': 'block', 'reason': 'terminal policy'}))\n",
        encoding="utf-8",
    )
    policy.chmod(0o755)

    cfg = {
        "hooks_auto_accept": True,
        "hooks": {
            "pre_tool_call": [
                {"matcher": "terminal", "command": str(policy)},
            ],
        },
    }
    monkeypatch.setattr(hermes_config, "load_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.model_switch.prewarm_picker_cache_async", lambda: None,
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    outbound_webhooks.reset_for_tests()

    try:
        _run_main_to_eof(monkeypatch)

        for command in ("python3 check.py", "sleep 15", "agent-browser open x && sleep 1"):
            assert plugins.get_pre_tool_call_block_message(
                tool_name="terminal", args={"command": command},
            ) == "terminal policy"

        assert plugins.get_pre_tool_call_block_message(
            tool_name="terminal",
            args={"command": "agent-browser open https://www.ainclave.com/"},
        ) is None
    finally:
        shell_hooks.reset_for_tests()
        outbound_webhooks.reset_for_tests()
        plugins._plugin_manager = None
