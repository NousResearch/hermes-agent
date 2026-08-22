"""Compatibility and shared-state seams for the MCP connect cooldown extraction."""

from __future__ import annotations

from tools import mcp_connect_cooldown as cooldown
from tools import mcp_tool


_MOVED_NAMES = (
    "_server_connect_retry_after",
    "_server_connect_failures",
    "_CONNECT_RETRY_BASE_BACKOFF_SEC",
    "_CONNECT_RETRY_MAX_BACKOFF_SEC",
    "_record_connect_failure",
    "_clear_connect_failure",
    "_connect_cooldown_active",
)


def _clear_state() -> None:
    cooldown._server_connect_retry_after.clear()
    cooldown._server_connect_failures.clear()


def test_original_namespace_reexports_preserve_identity() -> None:
    for name in _MOVED_NAMES:
        assert getattr(mcp_tool, name) is getattr(cooldown, name)


def test_cooldown_behavior_and_cross_module_clear_mutation(monkeypatch) -> None:
    _clear_state()
    now = [100.0]
    monkeypatch.setattr(cooldown.time, "monotonic", lambda: now[0])

    mcp_tool._record_connect_failure("broken-server")
    assert cooldown._server_connect_failures == {"broken-server": 1}
    assert mcp_tool._server_connect_failures is cooldown._server_connect_failures
    assert mcp_tool._server_connect_retry_after is cooldown._server_connect_retry_after
    assert cooldown._connect_cooldown_active("broken-server") is True

    now[0] = 130.0
    assert mcp_tool._connect_cooldown_active("broken-server") is False

    # The legacy namespace is the mutation authority used by shutdown paths.
    mcp_tool._server_connect_retry_after["broken-server"] = 999.0
    mcp_tool._server_connect_failures["broken-server"] = 7
    cooldown._server_connect_retry_after.clear()
    assert mcp_tool._server_connect_retry_after == {}
    assert mcp_tool._server_connect_failures == {"broken-server": 7}

    # The reverse direction is also the same live object, not a copied dict.
    mcp_tool._server_connect_failures.clear()
    assert cooldown._server_connect_failures == {}

    mcp_tool._record_connect_failure("broken-server")
    assert cooldown._server_connect_failures["broken-server"] == 1
    cooldown._clear_connect_failure("broken-server")
    assert mcp_tool._server_connect_failures == {}
    assert mcp_tool._server_connect_retry_after == {}
