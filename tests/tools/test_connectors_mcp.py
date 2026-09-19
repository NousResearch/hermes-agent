"""MCP targets of manage_connections (the fold that retired setup_mcp).

Contracts:
- mixed managed + MCP call off-desktop: managed proceeds, MCP settles ``unavailable`` with the
  terminal hint; neither leaks into the other's result
- callback-less registry dispatch is deterministic and never blocks
- the GUI callback round-trip: renderer answer folds into the operation, settles once
- catalog validation: install is catalog-only, enable/authorize need a configured server
- the replay shim keeps an old ``setup_mcp`` call dispatching
- deadline ownership: fixed operation deadline + sequential-deadline exemption
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.connectors.tool  # registers the tool
from tools.connectors.contract import SettleReason, TargetState
from tools.connectors import live
from tools.connectors import operation as op
from tools.connectors.mcp import apply_answer
from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections
from tools.registry import registry

CATALOG = ["figma", "linear", "notion"]
CONFIGURED = {"paper": {"command": "paper-mcp"}, "linear": {"url": "https://mcp.linear.app/mcp"}}


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


@pytest.fixture(autouse=True)
def _catalog():
    with patch("tools.connectors.mcp._catalog_names", return_value=CATALOG), \
         patch("tools.connectors.mcp._configured_names", return_value=sorted(CONFIGURED)), \
         patch("tools.connectors.mcp.session_platform", return_value="desktop"):
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


def test_mcp_targets_off_the_desktop_settle_unavailable_even_with_a_callback():
    """The Ink TUI has the gateway callback attached but no card; the surface decides, never the callback."""
    callback = _answering(None)
    with patch("tools.connectors.mcp.session_platform", return_value="tui"), \
         patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 5):
        out = _mcp({"action": "install", "connectors": [_linear()]}, callback)
    assert callback.seen == []
    assert out["status"] == "unavailable"
    assert out["settled_by"] == SettleReason.unavailable.value
    (target,) = out["targets"]
    assert target["state"] == TargetState.unavailable.value
    assert target["hint"] == "hermes mcp install linear / hermes mcp login linear"
    assert "error" not in out


def test_registry_dispatch_never_blocks_and_never_reaches_a_card():
    # registry.dispatch forwards no callback; the call must return, not block.
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


def test_a_managed_call_off_desktop_returns_a_link_per_target():
    client = FakeClient()
    out = json.loads(manage_connections(
        {"action": "connect", "connectors": ["gmail"]}, client_factory=lambda: client))
    assert client.calls == [("connections", ("gmail",), False)]
    assert out["targets"][0]["connect_url"] == "https://x/gmail"
    assert out["status"] == "initiated"


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
# the GUI round-trip: the card answers through connection.respond, the op settles
# ---------------------------------------------------------------------------


def _answering(answer, *, session_id="s1", delay=0.02):
    """A card that emits (callback returns None) and answers the live operation a moment later,
    the way ``connection.respond`` does from the renderer."""
    seen = []

    def callback(payload):
        seen.append(payload)

        def respond():
            operation = live.get(session_id, payload["op_id"])
            if operation is not None:
                apply_answer(operation, answer)
                if not operation.settled and answer:
                    operation.settle(SettleReason.all_resolved if operation.all_resolved else SettleReason.continue_)

        if answer is not None:
            threading.Timer(delay, respond).start()
        return None

    callback.seen = seen
    return callback


def _mcp(args, callback, **kw):
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        return json.loads(manage_connections(args, connection_callback=callback, session_id="s1", **kw))


def test_callback_answer_folds_into_the_operation_and_settles_once():
    callback = _answering(json.dumps({"settled_by": "all_resolved", "targets": [
        {"name": "linear", "status": "installed", "tools": ["a", "b"]},
        {"name": "figma", "status": "declined"},
    ]}))
    out = _mcp({"action": "install", "connectors": [_linear(), {"name": "figma", "mcp": True}]}, callback)
    (payload,) = callback.seen
    assert "reason" not in payload
    assert [t["name"] for t in payload["targets"]] == ["linear", "figma"]
    assert payload["timeout_seconds"] == op.OPERATION_DEADLINE_SECONDS
    assert out["status"] == "settled" and out["settled_by"] == SettleReason.all_resolved.value
    by_name = {t["name"]: t for t in out["targets"]}
    assert by_name["linear"]["state"] == TargetState.connected.value and by_name["linear"]["tools"] == ["a", "b"]
    assert by_name["figma"]["state"] == TargetState.skipped.value


def test_no_answer_settles_by_deadline_and_marks_targets_not_connected():
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(None))
    assert out["settled_by"] == SettleReason.deadline.value
    assert out["targets"][0]["state"] == TargetState.not_connected.value
    assert "error" not in out


def test_mcp_secrets_never_reach_the_model():
    # A renderer that echoes a credential field: only the allowed keys survive.
    answer = json.dumps({"targets": [{"name": "linear", "status": "installed", "api_key": "sk-secret", "env": {"K": "v"}}]})
    out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(answer))
    assert "sk-secret" not in json.dumps(out)


# ---------------------------------------------------------------------------
# the inline executor + replay shim
# ---------------------------------------------------------------------------


def _agent(callback):
    return SimpleNamespace(session_id="s1", connection_callback=callback)


def test_inline_executor_hands_the_agent_callback_to_the_tool():
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    callback = _answering(json.dumps({"targets": [{"name": "linear", "status": "installed"}]}))
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(INLINE_TOOL_EXECUTORS["manage_connections"](
            _agent(callback), {"action": "install", "connectors": [_linear()]}, InlineToolContext("task")))
    assert len(callback.seen) == 1
    assert out["targets"][0]["state"] == TargetState.connected.value


def test_setup_mcp_replay_shim_translates_to_an_mcp_target():
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    callback = _answering(json.dumps({"targets": [{"name": "linear", "status": "declined"}]}))
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(INLINE_TOOL_EXECUTORS["setup_mcp"](
            _agent(callback), {"server": "linear", "action": "install", "reason": "old convo"}, InlineToolContext("task", tool_call_id="call-9")))
    (target,) = callback.seen[0]["targets"]
    assert (target["name"], target["kind"], target["action"], target["state"]) == ("linear", "mcp", "install", "pending")
    assert callback.seen[0]["tool_call_id"] == "call-9"
    assert out["targets"][0]["state"] == TargetState.skipped.value


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


def test_settle_reason_comes_from_target_state_not_the_renderer():
    # The renderer answered one of two targets and claimed all_resolved; the operation is not resolved.
    answer = json.dumps({"settled_by": "all_resolved", "targets": [{"name": "linear", "status": "declined"}]})
    out = _mcp({"action": "install", "connectors": [_linear(), {"name": "figma", "mcp": True}]}, _answering(answer))
    assert out["settled_by"] == SettleReason.continue_.value
    by_name = {t["name"]: t for t in out["targets"]}
    assert by_name["linear"]["state"] == TargetState.skipped.value
    assert by_name["figma"]["state"] == TargetState.not_connected.value
    assert "detail" not in by_name["figma"]
