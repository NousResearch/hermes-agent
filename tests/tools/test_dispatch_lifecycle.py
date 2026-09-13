"""Dispatch/shutdown seam: injected failures, real ledger and process exit."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fault", ["persist", "prune", "executor", "submit", "rollback", "stuck", "queued_submit"])
def test_failed_or_stuck_registration_does_not_block_following_child(tmp_path, fault):
    code = r'''
import contextlib, threading, time, sys
from tools import async_delegation as ad
mode = sys.argv[1]
release = threading.Event()
entered = threading.Event()
original_persist = ad._persist_dispatch
original_executor = ad._get_executor
original_prune = ad._prune_durable_records
original_transaction = ad._transaction
failed_ids = []
def persist(r):
    failed_ids.append(r['delegation_id'])
    ad._persist_dispatch = original_persist
    if mode == 'stuck':
        entered.set()
        release.wait()
    if mode == 'persist': raise RuntimeError('persist injected')
    original_persist(r)
def fail(*a, **k): raise RuntimeError('injected')
class BrokenExecutor:
    submit = fail
class RollbackConnection:
    def __init__(self, conn): self.conn = conn
    def execute(self, sql, *a):
        if sql.startswith('DELETE FROM async_delegations WHERE delegation_id='):
            raise RuntimeError('rollback injected')
        return self.conn.execute(sql, *a)
@contextlib.contextmanager
def transaction():
    with original_transaction() as conn: yield RollbackConnection(conn)
runner_calls = []
dispatch_calls = []
def dispatch():
    index = len(dispatch_calls)
    dispatch_calls.append(index)
    def runner():
        runner_calls.append(index)
        release.wait()
    return ad.dispatch_async_delegation(goal='offline', context=None, toolsets=None,
        role='leaf', model='fixture', session_key='owned', runner=runner,
        interrupt_fn=lambda: None)
ad._persist_dispatch = persist
if mode == 'prune': ad._prune_durable_records = fail
if mode == 'executor': ad._get_executor = fail
if mode in ('submit', 'rollback'): ad._get_executor = lambda *a: BrokenExecutor()
if mode == 'rollback': ad._transaction = transaction
if mode == 'queued_submit':
    executor = original_executor(3)
    adjust = executor._adjust_thread_count
    executor._adjust_thread_count = fail
errors = []
def first():
    try: dispatch()
    except Exception as exc: errors.append(str(exc))
if mode == 'stuck':
    threading.Thread(target=first, daemon=True).start()
    assert entered.wait(2)
else: first()
ad._prune_durable_records = original_prune
ad._get_executor = original_executor
ad._transaction = original_transaction
if mode == 'queued_submit': executor._adjust_thread_count = adjust
healthy = dispatch()
assert healthy['status'] == 'dispatched'
start = time.monotonic()
counts = ad.finalize_for_oneshot_shutdown(grace_seconds=.02)
assert time.monotonic() - start < .12 + .03
assert ad.get_durable_delegation(healthy['delegation_id'])['state'] == 'unknown', counts
assert counts['signaled'] == 1, counts
assert len([t for t in threading.enumerate() if t.name.startswith('oneshot-')]) <= 2
if mode == 'stuck':
    assert counts['pending'] == 1, counts
else:
    assert not errors, errors
    assert 0 not in runner_calls, runner_calls
    row = ad.get_durable_delegation(failed_ids[0])
    assert row['state'] == 'error', row
    assert counts['pending'] == 0, counts
print(counts, flush=True)
'''
    child = subprocess.run([sys.executable, "-c", code, fault], cwd=ROOT,
                           env={**os.environ, "HERMES_HOME": str(tmp_path)},
                           text=True, capture_output=True, timeout=8)
    assert child.returncode == 0, child.stdout + child.stderr
