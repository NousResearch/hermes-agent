"""Gateway wire model: the status vocabulary the gateway sends is typed, unknown values fail loud,
and the execute-path CONNECTION_REQUIRED error carries a link only where no card exists."""

import pytest
from pydantic import ValidationError

from tools.connectors.gateway import wire
from tools.connectors.gateway.merge import partition_calls, splice_remote_results


def test_connections_result_carries_status_reason_under_either_spelling():
    for key in ("statusReason", "status_reason"):
        row = wire.ConnectorConnectionResult.model_validate(
            {"connector": "gmail", "status": "failed", key: "vendor: bad scope"})
        assert row.status_reason == "vendor: bad scope"


def test_list_item_accepts_the_seven_states_and_absence():
    for value in ("active", "initiated", "failed", "expired", "revoked", "inactive", "initializing"):
        item = wire.ConnectorListItem.model_validate({"connector": "gmail", "connected": False, "connectionStatus": value})
        assert item.connection_status == value
    assert wire.ConnectorListItem.model_validate({"connector": "gmail", "connected": True}).connection_status is None


def test_list_item_rejects_an_unknown_status_loudly():
    with pytest.raises(ValidationError):
        wire.ConnectorListItem.model_validate({"connector": "gmail", "connected": False, "connectionStatus": "weird"})


def _connection_required_entry():
    planned = partition_calls([{"name": "connectors__gmail__SEND_EMAIL"}]).remote
    remote = [{"data": None, "error": {
        "code": "CONNECTION_REQUIRED", "message": "connect gmail first",
        "connect_url": "https://example.test/connect/abc", "hint": "then retry"}}]
    (entry,) = splice_remote_results(planned, remote)
    return entry["error"]


def test_connection_required_on_desktop_names_the_card_and_drops_the_url(monkeypatch):
    monkeypatch.setattr("tools.connectors.gateway.merge.session_platform", lambda: "desktop")
    error = _connection_required_entry()
    assert error["connector"] == "gmail"
    assert error["connect_card_available"] is True
    assert "connect_url" not in error


def test_connection_required_off_desktop_keeps_the_url(monkeypatch):
    monkeypatch.setattr("tools.connectors.gateway.merge.session_platform", lambda: "tui")
    error = _connection_required_entry()
    assert error["connect_url"] == "https://example.test/connect/abc"
    assert "connect_card_available" not in error
