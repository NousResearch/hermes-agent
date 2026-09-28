"""trigger.list RPC — the TUI /trigger picker's data source.

Uses the shared ``configured_triggers`` helper (single source of truth with the CLI
``/trigger`` chooser in ``hermes_cli.cli_commands_mixin``), so config-defined triggers
surface identically across the classic CLI and the TUI.
"""

from __future__ import annotations

from tui_gateway import server


def test_trigger_list_returns_builtins(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    result = server._methods["trigger.list"](1, {})["result"]
    phrases = [t["phrase"] for t in result["triggers"]]
    assert {
        "todo tracking", "shepherd the flock", "Gate approved", "gate notebook",
    }.issubset(set(phrases))
    assert all(t["description"] for t in result["triggers"])


def test_trigger_list_includes_config_extends(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {
        "triggers": [
            {"phrase": "forensics", "description": "Forensic investigation of a failure"},
        ],
    })
    result = server._methods["trigger.list"](1, {})["result"]
    phrases = {t["phrase"]: t["description"] for t in result["triggers"]}
    # Built-ins retained, config trigger added.
    assert phrases["todo tracking"]
    assert phrases["forensics"] == "Forensic investigation of a failure"


def test_configured_triggers_helper_matches_cli():
    """The module helper (RPC source) and the CLI mixin resolve the same list."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from hermes_cli.cli_commands_mixin import configured_triggers

    cfg = {"triggers": [{"phrase": "custom", "description": "Custom trigger"}]}
    mixin = CLICommandsMixin()
    setattr(mixin, "config", cfg)
    assert configured_triggers(cfg) == mixin._configured_triggers()