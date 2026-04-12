from tools.terminal_tool import TERMINAL_TOOL_DESCRIPTION


def test_terminal_tool_description_mentions_configured_environment():
    assert "configured environment" in TERMINAL_TOOL_DESCRIPTION
    assert "local machine" in TERMINAL_TOOL_DESCRIPTION
    assert "Docker container" in TERMINAL_TOOL_DESCRIPTION
    assert "Linux environment" not in TERMINAL_TOOL_DESCRIPTION


def test_terminal_tool_description_scopes_sandbox_warning():
    assert "sandboxes (when configured)" in TERMINAL_TOOL_DESCRIPTION
    assert "cloud sandboxes may be cleaned up" not in TERMINAL_TOOL_DESCRIPTION
