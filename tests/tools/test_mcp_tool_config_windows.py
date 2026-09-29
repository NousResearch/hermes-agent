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
