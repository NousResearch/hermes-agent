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
