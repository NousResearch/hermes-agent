"""Per-thread interrupt and yield signals: what each public call leaves behind for the thread."""

import threading

from tools import interrupt


def _other_tid() -> int:
    return threading.get_ident() + 7919


def test_interrupt_reason_is_kept_until_the_interrupt_is_raised_again_or_cleared():
    tid = threading.get_ident()
    try:
        interrupt.set_interrupt(True, reason="stopped by /stop")
        assert interrupt.is_interrupted()
        assert interrupt.get_interrupt_reason() == "stopped by /stop"
        interrupt.set_interrupt(True)
        assert interrupt.is_thread_interrupted(tid)
        assert interrupt.get_interrupt_reason() is None
    finally:
        interrupt.set_interrupt(False)
    assert not interrupt.is_interrupted()
    assert interrupt.get_interrupt_reason() is None


def test_a_yield_is_taken_exactly_once():
    tid = _other_tid()
    interrupt.request_yield(tid)
    interrupt.request_yield(tid)
    assert not interrupt.is_thread_interrupted(tid)
    assert interrupt.consume_yield(tid) is True
    assert interrupt.consume_yield(tid) is False
    assert interrupt.consume_yield(None) is False


def test_clearing_an_interrupt_drops_a_pending_yield():
    tid = _other_tid()
    interrupt.request_yield(tid)
    interrupt.set_interrupt(True, thread_id=tid)
    assert interrupt.is_thread_interrupted(tid)
    interrupt.set_interrupt(False, thread_id=tid)
    assert not interrupt.is_thread_interrupted(tid)
    assert interrupt.consume_yield(tid) is False


def test_a_yield_requested_while_interrupted_does_not_outlive_the_interrupt():
    tid = _other_tid()
    interrupt.set_interrupt(True, thread_id=tid)
    interrupt.request_yield(tid)
    assert interrupt.is_thread_interrupted(tid)
    interrupt.set_interrupt(False, thread_id=tid)
    assert interrupt.consume_yield(tid) is False


def test_run_if_not_interrupted_follows_the_current_threads_interrupt_only():
    calls = []
    interrupt.request_yield(threading.get_ident())
    try:
        assert interrupt.run_if_not_interrupted(lambda: calls.append("pending yield")) is True
        interrupt.set_interrupt(True)
        assert interrupt.run_if_not_interrupted(lambda: calls.append("interrupted")) is False
    finally:
        interrupt.set_interrupt(False)
    assert calls == ["pending yield"]
    assert interrupt.consume_yield(threading.get_ident()) is False
