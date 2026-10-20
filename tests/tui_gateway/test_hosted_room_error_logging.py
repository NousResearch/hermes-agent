"""Hosted task diagnostics persist fixed reason codes, never exception bodies."""
import logging
from types import SimpleNamespace

import pytest

from hermes_state_runtime import RuntimeStoreError
from tests.tui_gateway import test_hosted_room_driver_runtime as runtime_fixtures
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, _admit, _identity, _runtime,
)

db = runtime_fixtures.db


@pytest.mark.parametrize('reason', ['request_too_large', 'PRIVATE_SECRET_SENTINEL'])
def test_submit_error_log_contains_task_and_safe_reason_only(db, monkeypatch, caplog, reason):
    identity = _identity()
    _admit(db, identity, prompt='PRIVATE_BODY_SENTINEL')
    rpc = FakeSessionRPC()
    error = RuntimeStoreError(reason)
    error.args = ('PRIVATE_SECRET_SENTINEL PRIVATE_BODY_SENTINEL',)

    def refuse(**kwargs):
        raise error

    monkeypatch.setattr(rpc, 'submit', refuse)
    runtime = _runtime(db, rpc)
    with caplog.at_level(logging.WARNING, logger='tui_gateway.hosted_room_driver'):
        runtime._process_room(BINDING)
        runtime._record_error('worker cycle failed: PRIVATE_SECRET_SENTINEL')
        runtime._record_error('stop remains pending: PRIVATE_BODY_SENTINEL')
    assert identity.task_id in caplog.text
    assert ('request_too_large' if reason == 'request_too_large' else 'attempt_failed') in caplog.text
    assert 'PRIVATE_SECRET_SENTINEL' not in caplog.text
    assert 'PRIVATE_BODY_SENTINEL' not in caplog.text


@pytest.mark.parametrize('waiting_for_approval', [False, True])
def test_legal_long_wait_can_complete_without_stop_or_resubmit(db, monkeypatch, waiting_for_approval):
    """Simulated monotonic elapsed time, not a paid/real hour-long model run."""
    from gateway import hosted_room_driver as state
    from tui_gateway import hosted_room_driver as driver
    elapsed = [0.0]
    monkeypatch.setattr(driver, 'time', SimpleNamespace(monotonic=lambda: elapsed[0]))
    identity = _identity()
    _admit(db, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    sid = rpc.add_session()
    if waiting_for_approval:
        rpc.states[sid]['pending_approval'] = {
            'request_id': 'approval-1', 'command': 'read-only fixture', 'choices': ['once', 'deny']}
    actions = []
    runtime = _runtime(db, rpc, turn_timeout_seconds=7200,
                       pending_action=lambda room, member, action: actions.append(action))
    monkeypatch.setattr(runtime._wake, 'wait', lambda timeout: None)

    def progress():
        elapsed[0] += 1800
        if elapsed[0] == 3600:
            rpc.states[sid]['pending_approval'] = None
            rpc.complete(identity.task_id, content='Finished after a legal wait.')

    rpc.on_info = progress
    runtime._process_room(BINDING)
    assert elapsed[0] == 3600
    assert state.get_task(db, identity)['status'] == 'settled'
    assert len([call for call in rpc.calls if call[0] == 'submit']) == 1
    assert not [call for call in rpc.calls if call[0] == 'interrupt']
    if waiting_for_approval:
        assert any(action and action.get('request_id') == 'approval-1' for action in actions)
