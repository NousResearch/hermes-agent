"""Connection operation: exactly-once settlement, server-owned deadline, config-bounded wait."""

import pytest

from tools import connections_tool_operation as op


def _two_targets():
    return [op.Target("linear", "mcp", "install"), op.Target("figma", "mcp", "install")]


def test_settle_is_exactly_once_and_freezes_the_result():
    operation = op.ConnectionOperation(_two_targets(), wait_seconds=30)
    operation.record_target("linear", op.CONNECTED)
    assert operation.settle(op.SETTLED_CONTINUE) is True
    frozen = operation.result()
    # Later events update current state only, never the settled result.
    assert operation.settle(op.SETTLED_DEADLINE) is False
    operation.record_target("figma", op.CONNECTED)
    assert operation.settled_by == op.SETTLED_CONTINUE
    assert operation.result() == frozen
    assert operation.target("figma").state == op.CONNECTED  # live state did move


def test_unresolved_targets_are_marked_not_connected_at_settlement():
    operation = op.ConnectionOperation(_two_targets(), wait_seconds=30)
    operation.record_target("linear", op.CONNECTED)
    operation.settle(op.SETTLED_DEADLINE)
    states = {t["name"]: t for t in operation.result()["targets"]}
    assert states["linear"]["state"] == op.CONNECTED
    assert states["figma"]["state"] == op.NOT_CONNECTED
    assert states["figma"]["detail"] == op.SETTLED_DEADLINE


def test_all_resolved_means_connected_or_explicitly_skipped():
    operation = op.ConnectionOperation(_two_targets(), wait_seconds=30)
    operation.record_target("linear", op.CONNECTED)
    operation.record_target("figma", op.FAILED, "oauth denied")
    # A recoverable failure keeps the operation open.
    assert operation.settle_if_all_resolved() is False
    operation.record_target("figma", op.SKIPPED)
    assert operation.settle_if_all_resolved() is True
    assert operation.settled_by == op.SETTLED_ALL_RESOLVED


def test_deadline_is_set_at_creation_and_never_recomputed():
    operation = op.ConnectionOperation(_two_targets(), wait_seconds=42)
    first = operation.deadline_at
    assert first == pytest.approx(operation.created_at + 42)
    operation.record_target("linear", op.CONNECTED)
    payload = operation.request_payload()
    assert payload["deadline_at"] == first
    assert payload["op_id"] == operation.op_id
    assert [t["name"] for t in payload["targets"]] == ["linear", "figma"]


def test_record_target_rejects_unknown_names():
    operation = op.ConnectionOperation(_two_targets())
    assert operation.record_target("github", op.CONNECTED) is False


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, op.WAIT_TIMEOUT_DEFAULT_SECONDS),
        ({"connections": {"wait_timeout_seconds": 600}}, 600.0),
        ({"connections": {"wait_timeout_seconds": 1}}, op.WAIT_TIMEOUT_FLOOR_SECONDS),
        ({"connections": {"wait_timeout_seconds": "nope"}}, op.WAIT_TIMEOUT_DEFAULT_SECONDS),
        ({"connections": {"wait_timeout_seconds": True}}, op.WAIT_TIMEOUT_DEFAULT_SECONDS),
    ],
)
def test_wait_timeout_reads_only_its_own_key_with_a_floor_and_no_ceiling(config, expected):
    assert op.resolve_wait_timeout(config) == expected


def test_legacy_timeouts_never_proxy_for_the_wait(monkeypatch):
    # The batch guard env var and the clarify timeout are separate budgets.
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "7")
    assert op.resolve_wait_timeout({"agent": {"clarify_timeout": 9}}) == op.WAIT_TIMEOUT_DEFAULT_SECONDS
