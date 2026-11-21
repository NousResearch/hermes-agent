"""Windows-specific MCP stdio environment resolution contracts."""

import pytest

from tools.mcp_tool_config import _resolve_stdio_command


@pytest.mark.platforms("windows")
def test_stdio_resolution_accepts_mixed_case_windows_environment_keys(tmp_path):
    launcher = tmp_path / "audit-mcp.cmd"
    launcher.write_text("@echo off\r\n")

    command, _ = _resolve_stdio_command(
        "audit-mcp",
        {"Path": str(tmp_path), "Pathext": ".COM;.EXE;.BAT;.CMD"},
    )

    assert command == str(launcher)


def test_stdio_resolution_keeps_posix_path_case_sensitive(tmp_path, monkeypatch):
    launcher = tmp_path / "audit-mcp"
    launcher.write_text("#!/bin/sh\n")
    monkeypatch.setattr("tools.mcp_tool_config.sys.platform", "linux")

    command, env = _resolve_stdio_command("audit-mcp", {"Path": str(tmp_path)})

    assert command == "audit-mcp"
    assert env == {"Path": str(tmp_path)}
