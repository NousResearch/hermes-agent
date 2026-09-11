"""Task evidence is turn-local, revocable, and never inherited by children."""
import contextvars
from concurrent.futures import ThreadPoolExecutor

from tools import approval_task as task
from tools.approval_context import set_current_session_key, reset_current_session_key
from tools.thread_context import propagate_context_to_thread
from agent.delegation_context import delegated_child_context


def test_old_finalizer_preserves_successor():
    old = task.from_composer('s', {'kind': 'desktop_composer', 'raw_text': 'old'})
    new = task.from_composer('s', {'kind': 'desktop_composer', 'raw_text': 'new'})
    session = {'_approval_task_lease': new}
    task.release_task(session, old)
    assert not old.active and new.active
    assert session['_approval_task_lease'] is new


def test_explicit_contract_only_and_no_truncation():
    assert task.from_composer('s', {'text': 'edit'}) is None
    assert task.from_composer('s', {'kind': 'desktop_composer', 'raw_text': 'x' * 8193}) is None
    assert task.from_composer('s', {'kind': 'desktop_composer', 'raw_text': 'x\x00'}) is None
    lease = task.from_composer('s', {'kind': 'desktop_composer', 'raw_text': '  edit <x>  '})
    assert lease.record.raw_text == '  edit <x>  '


def test_revocation_threads_sessions_and_children():
    lease = task.from_composer('s', {'kind': 'desktop_composer', 'raw_text': 'edit'})
    session = set_current_session_key('s')
    try:
        with task.bind_task(lease):
            assert task.current_task() == lease.record
            with ThreadPoolExecutor(1) as pool:
                assert pool.submit(propagate_context_to_thread(task.current_task)).result() == lease.record
            with delegated_child_context():
                assert task.current_task() is None
            other = set_current_session_key('other')
            try:
                assert task.current_task() is None
            finally:
                reset_current_session_key(other)
            copied = contextvars.copy_context()
            lease.revoke()
            assert copied.run(task.current_task) is None
        assert task.current_task() is None
    finally:
        reset_current_session_key(session)
