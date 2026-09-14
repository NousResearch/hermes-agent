"""Behavior tests for the pure tool_gateway merge/partition/name logic.

Pure functions, zero fakes, no I/O — matching the DI-callable test idiom
(``test_managed_tool_gateway.py``). Wire/client behavior is covered in the
client PR; this file owns partition → splice → assemble and the name codec.
"""

import pytest

from tools.tool_gateway.config import ConnectorConfig, connectors_available
from tools.tool_gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    IdempotencyConflict,
    ToolGatewayError,
    parse_gateway_error,
)
from tools.tool_gateway.merge import (
    assemble_results,
    fill_remote_failure,
    partition_calls,
    splice_remote_results,
)
from tools.tool_gateway.names import (
    CONNECTOR_BATCH_SENTINEL,
    format_connector_name,
    parse_connector_name,
)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def test_parse_round_trips_and_keeps_tool_slug_underscores():
    name = format_connector_name("gmail", "GMAIL_SEND_EMAIL")
    parsed = parse_connector_name(name)
    assert parsed is not None
    assert (parsed.connector, parsed.tool) == ("gmail", "SEND_EMAIL")
    assert parsed.raw == name


@pytest.mark.parametrize(
    "bad",
    [
        None,
        42,
        "",
        "tool_search",
        "connectors__",
        "connectors____",
        "connectors__gmail",
        "connectors__gmail__",
        "connectors____GMAIL_SEND_EMAIL",
        CONNECTOR_BATCH_SENTINEL,  # planner sentinel is not a callable name
    ],
)
def test_parse_rejects_malformed_names_without_raising(bad):
    assert parse_connector_name(bad) is None


def test_parse_preserves_case_both_directions():
    parsed = parse_connector_name("connectors__GitHub__Create_Issue")
    assert parsed is not None
    assert (parsed.connector, parsed.tool) == ("GitHub", "Create_Issue")


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------


def test_partition_splits_mixed_batch_preserving_positions():
    calls = [
        {"name": "local_tool", "arguments": {"a": 1}},
        {"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "x"}},
        {"name": "another_local", "arguments": {}},
        {"name": "connectors__slack__POST_MESSAGE"},
    ]
    part = partition_calls(calls)
    assert [pos for pos, _ in part.local] == [0, 2]
    assert [p.position for p in part.remote] == [1, 3]
    assert part.remote[0].connector == "gmail"
    assert part.remote[0].arguments == {"to": "x"}
    assert part.remote[1].arguments == {}  # missing arguments -> {}
    assert part.errors == ()


def test_partition_malformed_connector_name_is_per_entry_error_siblings_run():
    calls = [
        {"name": "connectors__broken"},  # claims prefix, doesn't parse
        {"name": "connectors__gmail__SEND_EMAIL"},
    ]
    part = partition_calls(calls)
    assert len(part.errors) == 1
    assert part.errors[0]["index"] == 0
    assert part.errors[0]["error"]["code"] == "TOOL_NOT_FOUND"
    assert [p.position for p in part.remote] == [1]


def test_partition_is_total_on_garbage_entries():
    part = partition_calls([None, "just-a-string", {"no_name": True}])
    assert len(part.local) == 3
    assert part.remote == ()
    assert part.errors == ()


# ---------------------------------------------------------------------------
# splice
# ---------------------------------------------------------------------------


def _plan(calls):
    return partition_calls(calls).remote


def test_splice_maps_by_slot_and_renders_success_and_error():
    planned = _plan(
        [
            {"name": "connectors__gmail__SEND_EMAIL"},
            {"name": "connectors__slack__POST_MESSAGE"},
        ]
    )
    remote = [
        {"data": {"id": "msg_1"}, "error": None},
        {
            "data": None,
            "error": {"code": "TOOL_NOT_ALLOWED", "message": "policy refused"},
        },
    ]
    entries = splice_remote_results(planned, remote)
    assert entries[0] == {
        "index": 0,
        "name": "connectors__gmail__SEND_EMAIL",
        "response": {"id": "msg_1"},
    }
    assert entries[1]["index"] == 1
    assert entries[1]["error"]["code"] == "TOOL_NOT_ALLOWED"


def test_splice_short_remote_response_fills_provider_error():
    planned = _plan(
        [
            {"name": "connectors__gmail__SEND_EMAIL"},
            {"name": "connectors__slack__POST_MESSAGE"},
        ]
    )
    entries = splice_remote_results(planned, [{"data": "ok", "error": None}])
    assert entries[0]["response"] == "ok"
    assert entries[1]["error"]["code"] == "PROVIDER_ERROR"
    assert entries[1]["name"] == "connectors__slack__POST_MESSAGE"


def test_splice_over_long_remote_response_drops_surplus():
    planned = _plan([{"name": "connectors__gmail__SEND_EMAIL"}])
    entries = splice_remote_results(
        planned, [{"data": "ok", "error": None}, {"data": "surplus", "error": None}]
    )
    assert len(entries) == 1
    assert entries[0]["response"] == "ok"


def test_splice_none_response_fills_every_slot():
    planned = _plan([{"name": "connectors__gmail__SEND_EMAIL"}])
    entries = splice_remote_results(planned, None)
    assert entries[0]["error"]["code"] == "PROVIDER_ERROR"


def test_connection_required_renders_shared_shape_with_connect_url():
    planned = _plan([{"name": "connectors__gmail__SEND_EMAIL"}])
    remote = [
        {
            "data": None,
            "error": {
                "code": "CONNECTION_REQUIRED",
                "message": "connect gmail first",
                "connect_url": "https://example.test/connect/abc",
                "hint": "then retry the call",
            },
        }
    ]
    (entry,) = splice_remote_results(planned, remote)
    error = entry["error"]
    assert error["code"] == "CONNECTION_REQUIRED"
    assert error["connector"] == "gmail"
    assert error["connect_url"] == "https://example.test/connect/abc"  # not redacted
    assert error["hint"] == "then retry the call"


def test_fill_remote_failure_marks_all_planned_slots():
    planned = _plan(
        [
            {"name": "connectors__gmail__SEND_EMAIL"},
            {"name": "connectors__slack__POST_MESSAGE"},
        ]
    )
    entries = fill_remote_failure(planned, "gateway unreachable")
    assert [e["index"] for e in entries] == [0, 1]
    assert all(e["error"]["code"] == "PROVIDER_ERROR" for e in entries)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def test_assemble_recomputes_counts_over_merged_array():
    local = [{"index": 0, "name": "local_tool", "response": "local ok"}]
    remote = [
        {"index": 1, "name": "connectors__gmail__G", "response": "sent"},
        {"index": 2, "name": "connectors__x__Y", "error": {"code": "PROVIDER_ERROR", "message": "boom"}},
    ]
    out = assemble_results(3, local, remote)
    assert [e["index"] for e in out["results"]] == [0, 1, 2]
    assert out["success_count"] == 2
    assert out["error_count"] == 1
    assert out["total_count"] == 3


def test_assemble_interleaves_back_into_original_order():
    # original: [remote, local, remote] — splice order must not matter.
    remote = [
        {"index": 0, "name": "connectors__a__T", "response": "r0"},
        {"index": 2, "name": "connectors__b__U", "response": "r2"},
    ]
    local = [{"index": 1, "name": "local_tool", "response": "l1"}]
    out = assemble_results(3, local, remote)
    assert [e.get("response") for e in out["results"]] == ["r0", "l1", "r2"]


def test_assemble_is_total_on_unclaimed_and_duplicate_slots():
    out = assemble_results(
        2,
        [{"index": 0, "name": "a", "response": "first"}],
        [{"index": 0, "name": "a", "response": "dupe"}],  # dropped
    )
    assert out["results"][0]["response"] == "first"
    assert out["results"][1]["error"]["code"] == "PROVIDER_ERROR"  # unclaimed
    assert out["total_count"] == 2


# ---------------------------------------------------------------------------
# errors: the one envelope parser
# ---------------------------------------------------------------------------


def test_envelope_parser_maps_statuses_to_exception_family():
    assert isinstance(parse_gateway_error(401, {}), GatewayAuthError)
    assert isinstance(parse_gateway_error(403, {}), GatewayAuthError)
    assert isinstance(parse_gateway_error(404, {}), GatewayUnavailable)
    assert isinstance(parse_gateway_error(409, {}), IdempotencyConflict)
    err = parse_gateway_error(500, {})
    assert type(err) is ToolGatewayError
    assert err.retryable is True
    assert parse_gateway_error(400, {}).retryable is False


def test_envelope_parser_reads_nested_envelope_and_is_total_on_garbage():
    err = parse_gateway_error(
        409,
        {"error": {"code": "IDEMPOTENCY_CONFLICT", "message": "key reused"}, "requestId": "req_1"},
    )
    assert err.code == "IDEMPOTENCY_CONFLICT"
    assert err.request_id == "req_1"
    assert str(err) == "key reused"
    # garbage bodies never raise
    for body in (None, "plain text", 42, ["list"]):
        assert isinstance(parse_gateway_error(502, body), ToolGatewayError)


# ---------------------------------------------------------------------------
# config gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, True),  # absent -> default enabled
        (True, True),
        (False, False),
        ({"enabled": True}, True),
        ({"enabled": False}, False),
        ({"enabled": "false"}, False),
        ({"enabled": "yes"}, True),
        ({}, True),
        ("garbage", True),  # unknown shape -> default, never raises
    ],
)
def test_connector_config_from_raw(raw, expected):
    assert ConnectorConfig.from_raw(raw).enabled is expected


def test_connectors_available_requires_both_legs_and_fails_closed():
    on = lambda: ConnectorConfig(enabled=True)
    off = lambda: ConnectorConfig(enabled=False)
    assert connectors_available(config_loader=on, entitlement_check=lambda: True) is True
    assert connectors_available(config_loader=on, entitlement_check=lambda: False) is False
    assert connectors_available(config_loader=off, entitlement_check=lambda: True) is False

    def boom():
        raise RuntimeError("portal exploded")

    assert connectors_available(config_loader=on, entitlement_check=boom) is False
    assert connectors_available(config_loader=boom, entitlement_check=lambda: True) is False
