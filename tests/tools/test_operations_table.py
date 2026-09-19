"""Operations table ownership, exclusion, close, cancellation, and waiting."""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from tools.operations import OperationAlreadyOpen, Operations, Owner


class WaitingEvent(threading.Event):
    def __init__(self) -> None:
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.waiting.set()
        return super().wait(timeout)


@dataclass(eq=False)
class FakeOperation:
    owner: Optional[Owner]
    op_id: str
    deadline_at: float = field(default_factory=lambda: time.time() + 10)
    wake: threading.Event = field(default_factory=WaitingEvent)
    _settled_by: Optional[str] = None

    @property
    def settled(self) -> bool:
        return self._settled_by is not None

    @property
    def settled_by(self) -> Optional[str]:
        return self._settled_by

    def settle(self, reason: str) -> bool:
        if self.settled:
            return False
        self._settled_by = reason
        self.wake.set()
        return True

    @property
    def key(self) -> str:
        return self.owner.key if self.owner else "s1"

    def snapshot(self) -> dict[str, str]:
        return {"op_id": self.op_id}


def _owner(key="s1", profile="profile-a"):
    return Owner(profile, key)


def _op(owner=None, op_id="op-1"):
    return FakeOperation(owner or _owner(), op_id)


def test_open_then_get_by_owner_and_op_id():
    table = Operations()
    operation = _op()

    table.open(operation)

    assert table.get(_owner(), operation.op_id) is operation
    assert table.current(_owner()) == [operation]
    assert table.get(_owner("s2"), operation.op_id) is None
    assert table.get(_owner(), "nope") is None


def test_exclusive_refuses_an_unsettled_operation_and_admits_after_close():
    table = Operations()
    first, second = _op(), _op(op_id="op-2")
    table.open(first, exclusive=True)

    with pytest.raises(OperationAlreadyOpen):
        table.open(second, exclusive=True)

    first.settle("continue")
    table.close(first)
    table.open(second, exclusive=True)

    assert table.current(_owner()) == [second]


def test_close_is_fenced_and_idempotent():
    table = Operations()
    first, replacement = _op(), _op()
    table.open(first)
    table.open(replacement)

    table.close(first)
    table.close(first)

    assert table.get(_owner(), replacement.op_id) is replacement


def test_two_profiles_sharing_a_key_do_not_see_each_other():
    table = Operations()
    first = _op(_owner(profile="profile-a"))
    second = _op(_owner(profile="profile-b"))
    table.open(first, exclusive=True)
    table.open(second, exclusive=True)

    assert table.current(_owner(profile="profile-a")) == [first]
    assert table.current(_owner(profile="profile-b")) == [second]
    assert table.get(_owner(profile="profile-b"), first.op_id) is second
    assert table.get(_owner(profile="profile-b"), first.op_id) is not first


def test_cancel_settles_owned_operations_and_wakes_waiters():
    table = Operations()
    operation = _op()
    table.open(operation)

    assert table.cancel(_owner(), "interrupt") == 1

    assert operation.settled_by == "interrupt"
    assert operation.wake.is_set()


def test_wait_returns_the_reason_that_settles_the_operation_mid_wait():
    table = Operations()
    operation = _op()
    table.open(operation)
    def settle() -> None:
        operation.wake.waiting.wait(1)
        operation.settle("continue")

    waiter = threading.Thread(target=settle)
    waiter.start()

    assert table.wait(operation, timeout=1, slice_seconds=0.01) == "continue"
    waiter.join()


def test_wait_timeout_returns_without_settling():
    """A tick timeout is the caller's turn to observe; only the deadline settles the operation."""
    table = Operations()
    operation = _op()
    operation.deadline_at = time.time() + 60
    table.open(operation)
    assert table.wait(operation, timeout=0.02, slice_seconds=0.01) == "timeout"
    assert not operation.settled
    operation.deadline_at = time.time() - 1
    assert table.wait(operation, timeout=5, slice_seconds=0.01) == "deadline"
    assert operation.settled
