"""MCP targets of manage_connections (the fold that retired setup_mcp).

Contracts:
- mixed managed + MCP call off-desktop: managed proceeds, MCP settles ``unavailable`` with the
  terminal hint; neither leaks into the other's result
- callback-less registry dispatch is deterministic and never blocks
- the GUI callback round-trip: renderer answer folds into the operation, settles once
- catalog validation: install is catalog-only, enable/authorize need a configured server
- the replay shim keeps an old ``setup_mcp`` call dispatching
- deadline ownership: config key + sequential-deadline exemption
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.connections_tool  # registers the tool
from tools import connections_tool_operation as op
from tools.connections_tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections
from tools.registry import registry

CATALOG = ["figma", "linear", "notion"]
CONFIGURED = {"paper": {"command": "paper-mcp"}, "linear": {"url": "https://mcp.linear.app/mcp"}}


@pytest.fixture(autouse=True)
def _catalog():
    with patch("tools.connections_tool_mcp._catalog_names", return_value=CATALOG), \
         patch("tools.connections_tool_mcp._configured_names", return_value=sorted(CONFIGURED)):
        yield


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_connectors(self):
        self.calls.append("list")
        return [{"connector": "gmail", "enabled": True, "connected": False}]

    def connections(self, connectors, *, reinitiate=False):
        self.calls.append(("connections", tuple(connectors), reinitiate))
        return {"results": [{"connector": c, "status": "initiated", "connect_url": f"https://x/{c}"} for c in connectors]}


def _linear(**kw):
    return {"name": "linear", "mcp": True, **kw}


# ---------------------------------------------------------------------------
# off-desktop: no approval surface
# ---------------------------------------------------------------------------


def test_mcp_targets_without_a_callback_settle_unavailable_with_the_terminal_hint():
    out = json.loads(manage_connections({"action": "install", "connectors": [_linear()]}))
    assert out["status"] == "unavailable"
    assert out["settled_by"] == op.SETTLED_UNAVAILABLE
    (target,) = out["targets"]
    assert target["state"] == op.UNAVAILABLE
    assert target["hint"] == "hermes mcp install linear / hermes mcp login linear"
    assert "error" not in out


def test_registry_dispatch_never_blocks_and_never_reaches_a_card():
    # The exact failure mode the earlier fold died of: registry.dispatch forwards no callback.
    out = json.loads(registry.dispatch("manage_connections", {"action": "install", "connectors": [_linear()]}))
    assert out["status"] == "unavailable"


def test_a_managed_action_never_accepts_mcp_targets_and_vice_versa():
    client = FakeClient()
    out = json.loads(manage_connections(
        {"action": "connect", "connectors": ["gmail", _linear()]}, client_factory=lambda: client))
    assert "managed-connector action" in out["error"]
    assert client.calls == []  # rejected before any gateway call

    out = json.loads(manage_connections({"action": "install", "connectors": ["gmail", _linear()]}))
    assert "must carry" in out["error"]


def test_managed_leg_is_byte_for_byte_unchanged_by_the_fold():
    client = FakeClient()
    out = json.loads(manage_connections(
        {"action": "connect", "connectors": ["gmail", {"name": "gmail"}]}, client_factory=lambda: client))
    assert client.calls == [("connections", ("gmail",), False)]
    assert out["results"][0]["connect_url"] == "https://x/gmail"


def test_unknown_target_fields_are_rejected():
    out = json.loads(manage_connections({"action": "install", "connectors": [_linear(url="https://evil")]}))
    assert "unknown target field" in out["error"] and "url" in out["error"]


# ---------------------------------------------------------------------------
# catalog validation
# ---------------------------------------------------------------------------


def test_install_is_catalog_only_and_lists_the_catalog_on_a_miss():
    out = json.loads(manage_connections({"action": "install", "connectors": [{"name": "github", "mcp": True}]}))
    assert "github" in out["error"]
    assert "figma, linear, notion" in out["error"]


def test_enable_and_authorize_need_a_configured_server():
    out = json.loads(manage_connections({"action": "enable", "connectors": [{"name": "figma", "mcp": True}]}))
    assert "figma" in out["error"] and "paper" in out["error"]
    out = json.loads(manage_connections({"action": "authorize", "connectors": [{"name": "paper", "mcp": True}]}))
    assert out["status"] == "unavailable"  # known server, no card here


# ---------------------------------------------------------------------------
# the GUI round-trip
# ---------------------------------------------------------------------------


def test_callback_answer_folds_into_the_operation_and_settles_once():
    seen = []

    def callback(payload):
        seen.append(payload)
        return json.dumps({"settled_by": "all_resolved", "targets": [
            {"name": "linear", "status": "installed", "tools": ["a", "b"]},
            {"name": "figma", "status": "declined"},
        ]})

    out = json.loads(manage_connections(
        {"action": "install", "connectors": [_linear(), {"name": "figma", "mcp": True}], "reason": "tickets"},
        connection_callback=callback, wait_seconds=30))
    (payload,) = seen
    assert payload["reason"] == "tickets"
    assert [t["name"] for t in payload["targets"]] == ["linear", "figma"]
    assert payload["deadline_at"] == pytest.approx(payload["deadline_at"])  # server-owned, present
    assert payload["timeout_seconds"] == 30
    assert out["status"] == "settled" and out["settled_by"] == "all_resolved"
    by_name = {t["name"]: t for t in out["targets"]}
    assert by_name["linear"]["state"] == op.CONNECTED and by_name["linear"]["tools"] == ["a", "b"]
    assert by_name["figma"]["state"] == op.SKIPPED


def test_no_answer_settles_by_deadline_and_marks_targets_not_connected():
    out = json.loads(manage_connections(
        {"action": "install", "connectors": [_linear()]}, connection_callback=lambda payload: "", wait_seconds=5))
    assert out["settled_by"] == op.SETTLED_DEADLINE
    assert out["targets"][0]["state"] == op.NOT_CONNECTED
    assert "error" not in out


def test_mcp_secrets_never_reach_the_model():
    # A renderer that (wrongly) echoed a credential field: only the whitelisted keys survive.
    answer = json.dumps({"targets": [{"name": "linear", "status": "installed", "api_key": "sk-secret", "env": {"K": "v"}}]})
    out = json.loads(manage_connections(
        {"action": "install", "connectors": [_linear()]}, connection_callback=lambda payload: answer, wait_seconds=5))
    assert "sk-secret" not in json.dumps(out)


# ---------------------------------------------------------------------------
# the inline executor + replay shim
# ---------------------------------------------------------------------------


def _agent(callback):
    return SimpleNamespace(session_id="s1", connection_callback=callback)


def test_inline_executor_hands_the_agent_callback_to_the_tool():
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    calls = []

    def callback(payload):
        calls.append(payload)
        return json.dumps({"targets": [{"name": "linear", "status": "installed"}]})

    out = json.loads(INLINE_TOOL_EXECUTORS["manage_connections"](
        _agent(callback), {"action": "install", "connectors": [_linear()]}, InlineToolContext("task")))
    assert len(calls) == 1
    assert out["targets"][0]["state"] == op.CONNECTED


def test_setup_mcp_replay_shim_translates_to_an_mcp_target():
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    calls = []

    def callback(payload):
        calls.append(payload)
        return json.dumps({"targets": [{"name": "linear", "status": "declined"}]})

    out = json.loads(INLINE_TOOL_EXECUTORS["setup_mcp"](
        _agent(callback), {"server": "linear", "action": "install", "reason": "old convo"}, InlineToolContext("task")))
    assert calls[0]["targets"] == [{"name": "linear", "kind": "mcp", "action": "install"}]
    assert calls[0]["reason"] == "old convo"
    assert out["targets"][0]["state"] == op.SKIPPED


def test_setup_mcp_is_gone_from_every_advertised_toolset():
    from toolsets import TOOLSETS, resolve_toolset

    assert all("setup_mcp" not in resolve_toolset(name) for name in TOOLSETS)
    assert "manage_connections" in resolve_toolset("connections")
    assert "hand-edit" in MANAGE_CONNECTIONS_SCHEMA["description"]
    assert "mcp_servers" in MANAGE_CONNECTIONS_SCHEMA["description"]


# ---------------------------------------------------------------------------
# deadline ownership
# ---------------------------------------------------------------------------


def test_the_bounded_wait_owns_the_deadline_not_the_sequential_guard():
    from agent import tool_executor as te

    assert "manage_connections" in te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS


def test_default_wait_comes_from_the_config_key(monkeypatch):
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "3")
    seen = {}
    with patch("tools.connections_tool_mcp.resolve_wait_timeout", return_value=77.0):
        manage_connections({"action": "install", "connectors": [_linear()]},
                           connection_callback=lambda p: seen.update(p) or "")
    assert seen["timeout_seconds"] == 77.0
