"""Live operation registry: one open op per session, found by op_id, closed on settle."""

import pytest

from tools.connectors import live
from tools.connectors import operation as op


@pytest.fixture(autouse=True)
def _clean():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


def _op(session="s1"):
    return op.ConnectionOperation([op.Target("gmail", "connector", "connect")], session_key=session)


def test_open_then_get_by_session_and_op_id():
    operation = _op()
    live.open(operation)
    assert live.get("s1", operation.op_id) is operation
    assert live.current("s1") is operation
    assert live.get("s2", operation.op_id) is None  # another session cannot see it
    assert live.get("s1", "nope") is None


def test_one_open_operation_per_session():
    first, second = _op(), _op()
    live.open(first)
    with pytest.raises(live.OperationAlreadyOpen):
        live.open(second)
    live.close(first)
    live.open(second)
    assert live.current("s1") is second


def test_close_removes_and_is_idempotent():
    operation = _op()
    live.open(operation)
    live.close(operation)
    live.close(operation)
    assert live.current("s1") is None


def test_two_profiles_may_share_a_session_key_without_seeing_each_other(tmp_path):
    """The registry is keyed by profile home and session key: stored sessions of two multiplexed
    profiles can carry the same timestamp-based key (P1-14)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
    # The tool thread opens under its turn's profile override; the RPC side names the session's
    # profile home explicitly (None means the process home, the default profile).
    first, second = _op(), _op()
    token = set_hermes_home_override(str(homes[0]))
    try:
        live.open(first)
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(str(homes[1]))
    try:
        live.open(second)  # not OperationAlreadyOpen: another profile's session
    finally:
        reset_hermes_home_override(token)
    assert live.current("s1", profile_home=str(homes[0])) is first
    assert live.current("s1", profile_home=str(homes[1])) is second
    assert live.get("s1", first.op_id, profile_home=str(homes[1])) is None
    assert live.current("s1") is None  # the default profile holds nothing
    live.close(first)
    live.close(second)
    assert live.current("s1", profile_home=str(homes[0])) is None
    assert live.current("s1", profile_home=str(homes[1])) is None
