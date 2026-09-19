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


ITEM = {"connector": "gmail", "connected": False}


def test_list_item_accepts_the_six_contract_states_and_absence():
    for value in ("pending", "active", "failed", "expired", "revoked", "inactive"):
        item = wire.ConnectorListItem.model_validate({**ITEM, "connectionStatus": value})
        assert item.connection_status == value
    assert wire.ConnectorListItem.model_validate({**ITEM, "connected": True}).connection_status is None


def test_list_item_rejects_the_retired_seven_state_words():
    for value in ("initiated", "initializing"):
        with pytest.raises(ValidationError):
            wire.ConnectorListItem.model_validate({**ITEM, "connectionStatus": value})


def test_list_page_is_typed_whole():
    page = wire.ConnectorListResponse.model_validate({"items": [ITEM], "nextCursor": None})
    assert page.next_cursor is None and page.items[0].connector == "gmail"
    with pytest.raises(ValidationError):
        wire.ConnectorListResponse.model_validate({"items": [{"connected": False}], "nextCursor": None})


def test_the_account_id_is_optional_by_vendor_semantics_and_absent_means_nothing_to_watch():
    """A no-auth toolkit answers active with no account; a failed mint answers CONNECTION_REQUIRED with
    neither link nor id. The id is typed Optional and never inferred (portal handoff, rule 2)."""
    initiated = wire.ConnectorConnectionResult.model_validate(
        {"connector": "gmail", "status": "initiated", "connectUrl": "https://c/1", "connectionId": "ca_1"})
    assert initiated.connection_id == "ca_1"
    assert wire.ConnectorConnectionResult.model_validate(
        {"connector": "gmail", "status": "initiated", "connectUrl": "https://c/1"}).connection_id is None
    assert wire.ConnectorConnectionResult.model_validate({"connector": "wiki", "status": "active"}).connection_id is None
    base = {"code": "CONNECTION_REQUIRED", "message": "connect gmail", "connector": "gmail"}
    assert wire.ConnectorToolError.model_validate(base).connection_id is None
    assert wire.ConnectorToolError.model_validate({**base, "connectUrl": "https://c/1"}).connection_id is None
    assert wire.ConnectorToolError.model_validate({**base, "connectUrl": "https://c/1", "connectionId": "ca_1"}).connection_id == "ca_1"


def test_connect_and_execute_requests_carry_the_return_target_and_the_operation_id():
    body = wire.ConnectorConnectionsRequest(connectors=["gmail"], return_to="hermes-desktop", op="op_1").model_dump(
        by_alias=True, exclude_none=True)
    assert body == {"connectors": ["gmail"], "reinitiate": False, "returnTo": "hermes-desktop", "op": "op_1"}
    with pytest.raises(ValidationError):
        wire.ConnectorConnectionsRequest(connectors=["gmail"], return_to="portal-web")
    execute = wire.ConnectorExecuteRequest(tools=[], return_to="hermes-desktop-dev").model_dump(by_alias=True, exclude_none=True)
    assert execute == {"tools": [], "returnTo": "hermes-desktop-dev"}


def test_account_row_is_typed_per_the_contract():
    row = wire.ConnectorAccount.model_validate({
        "connectionId": "ca_1", "connector": "gmail", "status": "active", "label": "gmail_knop-bual",
        "active": True, "createdAt": "2026-09-14T10:00:00.000Z", "updatedAt": "2026-09-14T10:00:00.000Z"})
    assert row.alias is None and row.active is True
    with pytest.raises(ValidationError):
        wire.ConnectorAccount.model_validate({"connectionId": "ca_1", "connector": "gmail", "status": "initiated",
                                              "label": "x", "active": True, "createdAt": "t", "updatedAt": "t"})


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
