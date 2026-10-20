"""Interrupt admission while minting or attaching a dispatch approval lease."""
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from tools import approval, async_delegation as ad, approval_notify_lease as leases


@pytest.mark.parametrize('phase', ['mint', 'attach'])
@pytest.mark.parametrize('scope', ['all', 'session'])
def test_interrupt_during_approval_handoff(monkeypatch, phase, scope):
    ad._reset_for_tests()
    paused, resume = threading.Event(), threading.Event()
    captured, submitted, results = [], [], []
    key = 'handoff-interrupt'
    def notify(data):
        pass
    def factory(lease, dispatch_id):
        captured.append(lease)
        if phase == 'mint':
            paused.set()
            assert resume.wait(3)
        return notify
    notify.acquire_background = factory
    approval.register_gateway_notify(key, notify)
    real_acquire = leases.acquire
    def acquire(*args, **kwargs):
        lease = real_acquire(*args, **kwargs)
        if phase == 'attach':
            paused.set()
            assert resume.wait(3)
        return lease
    monkeypatch.setattr(leases, 'acquire', acquire)
    def submit(fn):
        submitted.append(fn)
        return Future()
    monkeypatch.setattr(ad, '_get_executor', lambda n: SimpleNamespace(submit=submit))
    def dispatch():
        results.append(ad.dispatch_async_delegation(goal='fixture', context=None, toolsets=None,
            role='leaf', model=None, session_key=key, runner=dict))
    thread = threading.Thread(target=dispatch)
    try:
        thread.start()
        assert paused.wait(3)
        if scope == 'all':
            ad.interrupt_all()
        else:
            ad.interrupt_for_session(session_key=key)
        resume.set()
        thread.join(3)
        assert not thread.is_alive(), 'approval handoff deadlocked'
        assert not submitted, 'interrupted admission submitted a late worker'
        assert results[0]['status'] == 'rejected'
        assert len(captured) == 1
        assert not captured[0].active
        assert captured[0].callback is None
        assert not captured[0].permits('later-request')
        assert next(iter(ad._records.values()))['status'] == 'interrupted'
    finally:
        resume.set()
        thread.join(3)
        for lease in captured:
            lease.release()
        approval.unregister_gateway_notify(key)
        ad._reset_for_tests()
