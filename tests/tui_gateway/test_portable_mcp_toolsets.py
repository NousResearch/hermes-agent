"""Portable (plugin-shipped) MCP names in explicit ``HERMES_TUI_TOOLSETS`` pins (#119457 sibling)."""

import pytest


@pytest.mark.parametrize("native_override", [None, False, True])
def test_explicit_pin_accepts_portable_mcp_names(monkeypatch, tmp_path, capsys, native_override):
    from hermes_cli import plugins
    from hermes_cli.config import atomic_config_write
    from tui_gateway import server

    # Discovery is orthogonal to the name check: the shared resolver must carry the portable names,
    # while a native config entry wins on a collision (including an explicit disable).
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(plugins, "get_portable_mcp_server_names_nowait", lambda: {"portable-worker"})
    cfg = {"mcp_servers": {"native-worker": {"command": "python"}, "off-worker": {"enabled": False}}}
    if native_override is not None:
        cfg["mcp_servers"]["portable-worker"] = {"enabled": native_override}
    atomic_config_write(tmp_path / "config.yaml", cfg)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    monkeypatch.setenv(
        "HERMES_TUI_TOOLSETS", "terminal,portable-worker,native-worker,off-worker,portable-wroker"
    )
    result = server._load_enabled_toolsets()

    expected = (
        ["terminal", "native-worker"]
        if native_override is False
        else ["terminal", "portable-worker", "native-worker"]
    )
    assert result == expected
    notices = capsys.readouterr().err
    assert "ignoring unknown HERMES_TUI_TOOLSETS entries: portable-wroker" in notices
    assert "off-worker" in notices
    assert ("portable-worker" in notices) == (native_override is False)
