"""Account connector reads distinguish a missing sign-in from disabled rollout/settings."""

import threading

import pytest

from tools.connectors.gateway.config import ConnectorConfig
from tui_gateway import server


class _Transport:
    def __init__(self):
        self.frames = []
        self.arrived = threading.Event()

    def write(self, obj):
        self.frames.append(obj)
        if obj.get("id") == 1:
            self.arrived.set()
        return True

    def close(self):
        pass


def _rpc(method, **params):
    transport = _Transport()
    reply = server.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, transport
    )
    if reply is not None:
        return reply
    assert transport.arrived.wait(5), "no reply"
    return next(frame for frame in transport.frames if frame.get("id") == 1)


@pytest.mark.parametrize(
    ("method", "token", "enabled", "expected_reason"),
    [
        ("connectors.list", None, True, "NEEDS_NOUS_AUTH"),
        ("connectors.accounts", None, True, "NEEDS_NOUS_AUTH"),
        # A bearer without the portal's rollout claim remains unavailable.
        ("connectors.list", "cached-token", True, "CONNECTORS_UNAVAILABLE"),
        ("connectors.accounts", "cached-token", True, "CONNECTORS_UNAVAILABLE"),
        # Explicitly disabled settings take precedence over a missing sign-in.
        ("connectors.list", None, False, "CONNECTORS_UNAVAILABLE"),
        ("connectors.accounts", None, False, "CONNECTORS_UNAVAILABLE"),
    ],
)
def test_account_reads_report_sign_in_only_for_enabled_unauthed_profiles(
    monkeypatch, method, token, enabled, expected_reason
):
    availability_checks = []

    def unavailable():
        availability_checks.append(True)
        return False

    monkeypatch.setattr("tools.connectors.connectors_available", unavailable)
    monkeypatch.setattr(
        "tools.connectors.gateway.config.load_config",
        lambda: ConnectorConfig(enabled=enabled),
    )
    monkeypatch.setattr("tools.managed_tool_gateway.peek_nous_access_token", lambda: token)

    params = {"owner": {"type": "account"}} if method == "connectors.list" else {}
    reply = _rpc(method, **params)
    assert bool(availability_checks) is enabled

    if expected_reason == "NEEDS_NOUS_AUTH":
        assert reply.get("error", {}).get("data", {}).get("reason") == expected_reason, reply
    elif method == "connectors.list":
        assert reply.get("result") == {"available": False, "connectors": []}
    else:
        assert reply.get("error", {}).get("data", {}).get("reason") == expected_reason, reply
