"""Public callback routing authenticates connector flows by server and OAuth state."""

import asyncio

import pytest

from hermes_constants import get_hermes_home
from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions as sessions


@pytest.fixture
def flow():
    sessions._sessions.clear()
    flow = DashboardOAuthFlow("chat-flow", "reports", None, str(get_hermes_home()), "")
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=fixture-state"))
    sessions.register_flow(flow)
    yield flow
    sessions._sessions.clear()


def test_callback_by_state_delivers_once_and_forwards_issuer(flow):
    out = sessions.deliver_callback_by_state(
        "reports", code="fixture-code", state="fixture-state", iss="https://idp.example")
    assert out["ok"] is True
    assert asyncio.run(flow.wait_for_callback(timeout=0)) == ("fixture-code", "fixture-state", "https://idp.example")
    replay = sessions.deliver_callback_by_state("reports", code="fixture-code", state="fixture-state")
    assert replay["ok"] is False


@pytest.mark.parametrize("server,state,status", [
    ("reports", "wrong", "authorization_required"),
    ("reports", None, "authorization_required"),
    ("unknown", "fixture-state", "authorization_required"),
    ("reports", "fixture-state", "approved"),
    ("reports", "fixture-state", "error"),
    ("reports", "fixture-state", "starting"),
])
def test_callback_by_state_rejects_unmatched_or_nonpending_flows(flow, server, state, status):
    flow.status = status
    assert sessions.deliver_callback_by_state(server, code="fixture-code", state=state)["ok"] is False
    assert not flow._callback_ready.is_set()


def test_callback_by_state_requires_expected_state(flow):
    flow.expected_state = None
    assert sessions.deliver_callback_by_state("reports", code="fixture-code", state="fixture-state")["ok"] is False


def test_callback_by_state_propagates_provider_error(flow):
    assert sessions.deliver_callback_by_state(
        "reports", code=None, state="fixture-state", error="access_denied")["ok"] is True
    with pytest.raises(RuntimeError, match="access_denied"):
        asyncio.run(flow.wait_for_callback(timeout=0))
