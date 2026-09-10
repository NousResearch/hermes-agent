"""_ensure_turn_mcp_discovery: only the isolated compute_host child arms per-turn MCP discovery.

The dashboard PARENT already discovered at startup (skip). The child arms discovery under the
already-installed home override, but skips homes with no MCP servers, and is fail-soft.
"""
from __future__ import annotations

import pytest

from tui_gateway import prompt_turn as pt


@pytest.fixture(autouse=True)
def _clear_child_env(monkeypatch):
    monkeypatch.delenv("HERMES_COMPUTE_HOST_CHILD", raising=False)
    yield


def test_parent_process_does_not_arm_discovery(monkeypatch):
    # No HERMES_COMPUTE_HOST_CHILD -> parent -> must not touch mcp_startup at all.
    called = {"ensure": 0, "probe": 0}
    import hermes_cli.mcp_startup as ms
    monkeypatch.setattr(ms, "_has_configured_mcp_servers", lambda: called.__setitem__("probe", called["probe"] + 1) or True)
    monkeypatch.setattr(ms, "ensure_mcp_discovery_before_agent_build", lambda **k: called.__setitem__("ensure", called["ensure"] + 1))
    pt._ensure_turn_mcp_discovery()
    assert called == {"ensure": 0, "probe": 0}


def test_child_with_no_mcp_servers_skips_discovery(monkeypatch):
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    called = {"ensure": 0}
    import hermes_cli.mcp_startup as ms
    monkeypatch.setattr(ms, "_has_configured_mcp_servers", lambda: False)
    monkeypatch.setattr(ms, "ensure_mcp_discovery_before_agent_build", lambda **k: called.__setitem__("ensure", called["ensure"] + 1))
    pt._ensure_turn_mcp_discovery()
    assert called == {"ensure": 0}  # probe False -> nothing armed


def test_child_with_mcp_servers_ensures_discovery(monkeypatch):
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    seen = {}
    import hermes_cli.mcp_startup as ms
    monkeypatch.setattr(ms, "_has_configured_mcp_servers", lambda: True)
    monkeypatch.setattr(ms, "ensure_mcp_discovery_before_agent_build",
                        lambda **k: seen.update(k) or seen.__setitem__("called", True))
    pt._ensure_turn_mcp_discovery()
    assert seen.get("called") is True
    assert seen.get("thread_name") == "compute-host-mcp-discovery"


def test_discovery_failure_is_fail_soft(monkeypatch):
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    import hermes_cli.mcp_startup as ms
    monkeypatch.setattr(ms, "_has_configured_mcp_servers", lambda: True)
    def _boom(**k):
        raise RuntimeError("discovery exploded")
    monkeypatch.setattr(ms, "ensure_mcp_discovery_before_agent_build", _boom)
    # Must not raise -- a discovery failure never breaks the turn.
    pt._ensure_turn_mcp_discovery()
