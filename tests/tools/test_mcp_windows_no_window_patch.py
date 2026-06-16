"""Regression test for the Windows MCP console-window suppression patch.

On Windows, npx/.cmd MCP servers pop a visible, never-closing console window
because the MCP SDK's ``create_windows_process()`` drops ``CREATE_NO_WINDOW``
in its ``except Exception`` retry branch. ``tools/mcp_tool.py`` monkey-patches
that function to keep the flag on every spawn path.

The critical detail this test guards: ``mcp/client/stdio/__init__.py`` does
``from mcp.os.win32.utilities import create_windows_process`` and calls the
name bound in *its own* module namespace. Patching only
``mcp.os.win32.utilities`` (as PR #41078 does) therefore has no effect on the
real call site. We assert the name actually used by ``mcp.client.stdio`` is
the patched one, so the fix can't silently regress back into ineffectiveness.
"""

import sys

import pytest


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="The MCP console-window patch only installs on Windows.",
)
def test_mcp_client_stdio_create_windows_process_is_patched():
    import tools.mcp_tool  # noqa: F401 -- importing installs the patch

    pytest.importorskip("mcp.client.stdio")
    import mcp.client.stdio as stdio

    assert (
        stdio.create_windows_process.__name__
        == "_hermes_create_windows_process_no_window"
    ), (
        "mcp.client.stdio.create_windows_process must be replaced by the "
        "Hermes no-console-window patch. Patching only mcp.os.win32.utilities "
        "(PR #41078) leaves this call site unpatched and the window still pops."
    )
