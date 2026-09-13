import sys
import threading
import time
import pytest



@pytest.mark.parametrize('blocked', ['interrupt_callback', 'ledger_lock'])
def test_shutdown_budget_includes_synchronous_work(tmp_path, monkeypatch, blocked):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from tools import async_delegation as ad
    ad._reset_for_tests()
    release = threading.Event()
    record = {'delegation_id': 'review-bounded', 'goal': 'offline', 'dispatched_at': time.time(),
              'status': 'running', 'interrupt_fn': release.wait if blocked == 'interrupt_callback' else lambda: None}
    ad._persist_dispatch(record)
    with ad._records_lock:
        ad._records[record['delegation_id']] = record
    if blocked == 'ledger_lock':
        ad._DB_LOCK.acquire()
    def unblock():
        release.set()
        if blocked == 'ledger_lock':
            ad._DB_LOCK.release()
    timer = threading.Timer(3.2, unblock)
    timer.start()
    try:
        start = time.monotonic()
        result = ad.finalize_for_oneshot_shutdown(grace_seconds=0.02)
        elapsed = time.monotonic() - start
        assert elapsed < 2, f'{blocked}: grace=0.02s but elapsed={elapsed:.3f}s'
        assert result['unknown'] + result['pending'] == 1
        if blocked == 'ledger_lock':
            assert result['unknown'] == 0
            assert result['pending'] == 1
            assert record['status'] == 'finalizing'
    finally:
        timer.join(4)
        ad._reset_for_tests()
