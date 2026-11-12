"""Gateway wire model: the status vocabulary the gateway sends is typed, unknown values fail loud,
and the execute-path CONNECTION_REQUIRED error carries a link only where no card exists."""

import pytest
from pydantic import ValidationError

from tools.connectors.gateway import wire
from tools.connectors.gateway.merge import partition_calls, splice_remote_results
from tools.connectors.turn import CARD, LINK, SIDE, scoped_connection_surface


def test_connections_result_carries_status_reason_under_either_spelling():
    for key in ("statusReason", "status_reason"):
        row = wire.ConnectorConnectionResult.model_validate(
            {"connector": "gmail", "status": "failed", key: "vendor: bad scope"})
        assert row.status_reason == "vendor: bad scope"


ITEM = {"connector": "gmail", "connected": False}


def test_list_item_accepts_the_six_contract_states_and_absence():
    for value in ("pending", "active", "failed", "expired", "revoked", "inactive"):
        item = wire.ConnectorListItem.model_validate({**ITEM, "connectionStatus": value})
        assert item.connection_status == value
    assert wire.ConnectorListItem.model_validate({**ITEM, "connected": True}).connection_status is None


def test_list_item_rejects_an_unknown_status_loudly():
    with pytest.raises(ValidationError):
        wire.ConnectorListItem.model_validate({**ITEM, "connectionStatus": "weird"})


def _connection_required_entry():
    planned = partition_calls([{"name": "connectors__gmail__SEND_EMAIL"}]).remote
    remote = [{"data": None, "error": {
        "code": "CONNECTION_REQUIRED", "message": "connect gmail first",
        "connect_url": "https://example.test/connect/abc", "hint": "then retry"}}]
    (entry,) = splice_remote_results(planned, remote)
    return entry["error"]


def test_connection_required_with_a_card_or_in_a_side_agent_carries_no_link():
    with scoped_connection_surface(CARD):
        error = _connection_required_entry()
    assert error["connector"] == "gmail"
    assert error["connect_card_available"] is True
    assert "connect_url" not in error
    assert "manage_connections" in error["hint"]

    with scoped_connection_surface(SIDE):
        side = _connection_required_entry()
    assert "connect_url" not in side
    assert "connect_card_available" not in side
    assert "Only the main agent" in side["hint"]


def test_connection_required_without_a_card_keeps_the_url():
    with scoped_connection_surface(LINK):
        error = _connection_required_entry()
    assert error["connect_url"] == "https://example.test/connect/abc"
    assert error["hint"] == "then retry"
    assert "connect_card_available" not in error
