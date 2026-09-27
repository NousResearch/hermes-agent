import threading
import time
from concurrent.futures import Future, TimeoutError
from types import SimpleNamespace

import pytest

from tools import delegate_tool
from tools.delegate_tool_child_run import _ChildRun
from tools.delegate_tool_deadline import ReviewedDeadline, wait_with_reviewed_deadline


def test_real_future_wait_observes_a_parent_renewal():
    future = Future()
    lease = ReviewedDeadline(1.0)
    def parent():
        lease.report(1)
        lease.review(1, approve=True)
    timers = [threading.Timer(0.6, parent), threading.Timer(1.3, lambda: future.set_result("finished"))]
    for timer in timers:
        timer.start()
    try:
        assert wait_with_reviewed_deadline(future, lease) == 'finished'
        assert lease.snapshot()['renewals'] == 1
    finally:
        for timer in timers:
            timer.join(2)


def test_unanswered_report_does_not_prevent_native_timeout(monkeypatch):
    monkeypatch.setattr(delegate_tool, '_get_child_timeout', lambda: 0.15)
    monkeypatch.setattr(delegate_tool, '_load_config', lambda: {'reviewed_timeout': True})
    ended = threading.Event()
    child = SimpleNamespace(_interrupt_requested=False, close=lambda: None,
                            get_activity_summary=lambda: {'api_call_count': 1})
    def worker(**kwargs):
        child._delegate_reviewed_deadline.report(1)
        while not child._interrupt_requested:
            time.sleep(0.01)
        ended.set()
        return {'interrupted': True}
    child.run_conversation = worker
    run = _ChildRun(child, None, 0, 'test', None, None)
    result, error, deferred = run.await_child()
    assert result is None and error['status'] == 'timeout'
    assert 'Parent-reviewed deadline expired' in error['error']
    assert child._delegate_reviewed_deadline.remaining() == 0
    assert ended.wait(2)


def test_worker_timeout_exception_is_not_swallowed():
    future = Future()
    future.set_exception(TimeoutError('upstream'))
    with pytest.raises(TimeoutError, match='upstream'):
        wait_with_reviewed_deadline(future, ReviewedDeadline(600))


def test_native_worker_timeout_is_not_a_review_deadline(monkeypatch):
    monkeypatch.setattr(delegate_tool, '_get_child_timeout', lambda: 600)
    monkeypatch.setattr(delegate_tool, '_load_config', lambda: {'reviewed_timeout': True})
    def worker(**kwargs):
        raise TimeoutError('worker network timeout')
    child = SimpleNamespace(run_conversation=worker, _interrupt_requested=False,
        close=lambda: None, get_activity_summary=lambda: {'api_call_count': 1})
    result, error, deferred = _ChildRun(child, None, 0, 'test', None, None).await_child()
    assert result is None and error['status'] == 'error'
    assert error['error'] == 'worker network timeout'
