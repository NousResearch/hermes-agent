"""Tests for summarize_mcp_reload() -- the /reload-mcp diff (#80771)."""

from unittest.mock import patch


def _run(statuses, old, lazy=()):
    """Diff `statuses` (a get_mcp_status() snapshot) against `old`, the set of
    servers that were in ``_servers`` before the reload."""
    import tools.mcp_tool as mcp

    with patch.object(
        mcp, "get_mcp_status",
        return_value=[{"name": n, "status": s} for n, s in statuses.items()],
    ), patch.object(mcp, "_lazy_server_configs", {n: {} for n in lazy}):
        return mcp.summarize_mcp_reload(set(old))


def test_still_configured_servers_are_pending_not_removed():
    """Slow/failed/disabled servers stay in config -- they are not removals."""
    diff = _run(
        {"slow": "connecting", "broken": "failed", "off": "disabled", "ok": "connected"},
        old={"slow", "broken", "off", "ok"},
    )
    assert diff["removed"] == set()
    assert diff["pending"] == {
        "slow": "connecting", "broken": "failed", "off": "disabled",
    }
    assert diff["connected"] == {"ok"}
    assert diff["reconnected"] == {"ok"}


def test_server_dropped_from_config_is_removed():
    diff = _run({"ok": "connected"}, old={"ok", "gone"})
    assert diff["removed"] == {"gone"}
    assert diff["pending"] == {}
    assert diff["added"] == set()


def test_newly_connected_server_is_added():
    diff = _run({"fresh": "connected"}, old=set())
    assert diff["added"] == {"fresh"}
    assert diff["removed"] == set()


def test_lazy_server_counts_as_connected():
    """Lazy servers register tools from cache with no session -- available."""
    diff = _run({"notion": "configured"}, old=set(), lazy={"notion"})
    assert diff["connected"] == {"notion"}
    assert diff["pending"] == {}
