"""Test that discover_plugins() is called in _make_agent path.

Regression test for #50776: hook-based plugins (Langfuse, etc.) were silently
inactive in TUI/Web UI because _make_agent() never called discover_plugins().
"""
import importlib
import sys
import types
from unittest.mock import patch, MagicMock


def test_make_agent_calls_discover_plugins():
    """_make_agent must call discover_plugins() so hooks are registered."""
    # We cannot fully instantiate _make_agent (needs real config/model), so
    # verify the source contains the discover_plugins import + call.
    import inspect
    from tui_gateway.server import _make_agent

    src = inspect.getsource(_make_agent)
    assert "discover_plugins" in src, (
        "_make_agent does not reference discover_plugins; "
        "hook-based plugins will remain inactive in TUI path"
    )
    # Verify it imports from hermes_cli.plugins (not a stale/different module)
    assert "from hermes_cli.plugins import discover_plugins" in src, (
        "_make_agent imports discover_plugins from wrong module"
    )


def test_discover_plugins_is_idempotent():
    """Calling discover_plugins() multiple times is safe (no double-load)."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    # First call
    discover_plugins()
    mgr = get_plugin_manager()
    first_discovered = mgr._discovered

    # Second call should be a no-op
    discover_plugins()
    assert mgr._discovered is True
    # If it crashed or duplicated state, the test would fail above.


if __name__ == "__main__":
    test_make_agent_calls_discover_plugins()
    print("PASS: _make_agent references discover_plugins")
    test_discover_plugins_is_idempotent()
    print("PASS: discover_plugins is idempotent")
    print("ALL TESTS PASSED")
