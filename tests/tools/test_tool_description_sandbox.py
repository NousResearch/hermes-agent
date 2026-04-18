from tools.terminal_tool import TERMINAL_TOOL_DESCRIPTION
from tools.code_execution_tool import build_execute_code_schema


def test_terminal_tool_description_mentions_configured_environment():
    assert "configured environment" in TERMINAL_TOOL_DESCRIPTION
    assert "local machine" in TERMINAL_TOOL_DESCRIPTION
    assert "Docker container" in TERMINAL_TOOL_DESCRIPTION
    assert "Linux environment" not in TERMINAL_TOOL_DESCRIPTION


def test_terminal_tool_description_scopes_sandbox_warning():
    assert "sandboxes (when configured)" in TERMINAL_TOOL_DESCRIPTION
    assert "cloud sandboxes may be cleaned up" not in TERMINAL_TOOL_DESCRIPTION


def test_execute_code_schema_scopes_sandbox_warning():
    """execute_code schema should not unconditionally claim cloud sandbox usage."""
    schema = build_execute_code_schema()
    description = schema["description"]
    assert "Sandboxes (when configured)" in description
    assert "cloud sandbox" not in description
