"""MCP shutdown releases only the selected profile's cached stderr handle."""

from hermes_constants import (
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.mcp_tool_config import _get_mcp_stderr_log
from tools.mcp_tool_lifecycle import shutdown_mcp_servers


def test_scoped_shutdown_releases_log_and_preserves_other_profile(tmp_path):
    handles = []
    for name in ("first", "second"):
        token = set_hermes_home_override(tmp_path / name)
        try:
            handles.append(_get_mcp_stderr_log())
        finally:
            reset_hermes_home_override(token)
    first, second = handles
    try:
        shutdown_mcp_servers(scope=hermes_home_key(tmp_path / "first"))
        assert first.closed
        second.write("other profile remains usable\n")
        second.flush()
        token = set_hermes_home_override(tmp_path / "first")
        try:
            reopened = _get_mcp_stderr_log()
            handles.append(reopened)
            reopened.write("reload can reopen the log\n")
            reopened.flush()
        finally:
            reset_hermes_home_override(token)
        shutdown_mcp_servers()
        assert second.closed and reopened.closed
    finally:
        for handle in handles:
            handle.close()