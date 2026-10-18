"""Executor cancellation and submit failure settle both detached task reservations."""
from concurrent.futures import Future

from tools import async_delegation as delegation
from tools.approval_context import (approval_task_context, release_background_approval_task,
                                    reserve_background_approval_task)


def _dispatch(monkeypatch, tmp_path, executor):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(delegation, '_get_executor', lambda _size: executor)
    delegation._reset_for_tests()
    with approval_task_context('Edit only src/cli.py', identity='u', profile=str(tmp_path), session='route',
                               is_current=lambda: True):
        worker = reserve_background_approval_task()
        completion = reserve_background_approval_task()
        result = delegation.dispatch_async_delegation_batch(
            goals=['work'], context=None, toolsets=None, role='worker', model='fixture',
            session_key='route', parent_session_id='parent', runner=dict,
            delegation_id='deleg_cancelled', approval_reservation=completion,
            on_unstarted=lambda: release_background_approval_task(worker))
    return result, worker, completion


def test_cancelled_queued_executor_future_releases_reservations(monkeypatch, tmp_path):
    class Queued:
        future = Future()

        def submit(self, _call):
            return self.future

    executor = Queued()
    result, worker, completion = _dispatch(monkeypatch, tmp_path, executor)
    try:
        assert result['status'] == 'dispatched'
        assert worker.lease.live()
        assert executor.future.cancel()
        assert not worker.lease.live()
        assert 'deleg_cancelled' not in delegation._records
        assert delegation.get_durable_delegation('deleg_cancelled') is None
        release_background_approval_task(worker)
        release_background_approval_task(completion)
    finally:
        delegation._reset_for_tests()


def test_executor_submit_exception_releases_reservations(monkeypatch, tmp_path):
    class Broken:
        def submit(self, _call):
            raise RuntimeError('test executor unavailable')

    result, worker, completion = _dispatch(monkeypatch, tmp_path, Broken())
    try:
        assert result['status'] == 'rejected'
        assert not worker.lease.live()
        release_background_approval_task(worker)
        release_background_approval_task(completion)
    finally:
        delegation._reset_for_tests()
