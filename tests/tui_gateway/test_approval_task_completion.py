"""A direct TUI completion turn binds the producer's reserved request at review."""
from types import SimpleNamespace

import pytest

from hermes_constants import get_process_hermes_home
from tests.tui_gateway.test_failed_turn_retention import _session, turn_env
from tui_gateway import server
from tools.approval_context import (approval_task_context, current_approval_task_context,
                                    reserve_background_approval_task)
from tools.approval_smart import _smart_approve


@pytest.mark.usefixtures('turn_env')
def test_completion_prompt_turn_reviewer_uses_original_request(monkeypatch):
    profile = str(get_process_hermes_home())
    monkeypatch.setattr('agent.auxiliary_client.call_llm', lambda **_k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='APPROVE'))]))
    seen = []

    def run(*_args, **_kwargs):
        seen.append((current_approval_task_context(), _smart_approve('python -c "print(1)"', 'script')))
        return {'final_response': 'done', 'messages': []}

    agent = SimpleNamespace(session_id='session-key', run_conversation=run,
                            clear_interrupt=lambda: None)
    session = _session(agent)
    monkeypatch.setitem(server._sessions, 'approval-completion', session)
    with approval_task_context('Edit only alpha.py', identity='basic:alice', profile=profile,
                               session='session-key', task_id='original-admission',
                               is_current=lambda: True):
        reservation = reserve_background_approval_task()
    try:
        assert server._run_prompt_submit('completion-rid', 'approval-completion', session,
                                         '[finished] build', approval_reservations=(reservation,))
        assert seen and seen[0][0].text == 'Edit only alpha.py'
        assert seen[0][0].task_id == 'original-admission'
        assert seen[0][0].ancestry == ('completion:completion-rid',)
        assert seen[0][1] == 'approve'
    finally:
        reservation.release()
        server._sessions.pop('approval-completion', None)


@pytest.mark.usefixtures('turn_env')
def test_notification_dispatch_keeps_origin_until_continuation_finishes(monkeypatch):
    from tools import async_delegation as delegation
    profile = str(get_process_hermes_home())

    monkeypatch.setattr('agent.auxiliary_client.call_llm', lambda **_k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='APPROVE'))]))
    seen = []

    def run(*_args, **_kwargs):
        seen.append((current_approval_task_context(), _smart_approve('python -c "print(1)"', 'script')))
        return {'final_response': 'done', 'messages': []}

    agent = SimpleNamespace(session_id='session-key', run_conversation=run,
                            clear_interrupt=lambda: None)
    session = _session(agent, profile_home=profile, running=True)
    monkeypatch.setitem(server._sessions, 'approval-notification', session)
    pending = []
    monkeypatch.setattr(server, '_start_session_work',
                        lambda target, **_kwargs: pending.append(target) or object())
    monkeypatch.setattr('tools.async_delegation.claim_event_delivery', lambda *_a: 'claim')
    monkeypatch.setattr('tools.async_delegation.complete_event_delivery', lambda *_a: None)
    event = {'type': 'async_delegation', 'delegation_id': 'deleg_test_origin',
             'session_key': 'session-key'}
    with approval_task_context('Edit only alpha.py', identity='basic:alice', profile=profile,
                               session='session-key', task_id='original-admission',
                               is_current=lambda: True):
        reservation = reserve_background_approval_task()
    with delegation._records_lock:
        delegation._records[event['delegation_id']] = {
            'session_key': 'session-key', '_approval_reservation': reservation,
            'status': 'completed', 'completed_at': 1.0}
    try:
        server._notif_dispatch_event('approval-notification', session, event, '[finished] build')
        assert len(pending) == 1 and reservation.active()
        pending.pop()()
        assert seen and seen[0][0].text == 'Edit only alpha.py'
        assert seen[0][1] == 'approve'
        assert not reservation.active()
    finally:
        reservation.release()
        with delegation._records_lock:
            delegation._records.pop(event['delegation_id'], None)
        server._sessions.pop('approval-notification', None)


@pytest.mark.usefixtures('turn_env')
def test_queued_submit_generation_and_stop_fence_reviewer(monkeypatch):
    from tools import approval_context

    monkeypatch.setattr(approval_context, '_get_approval_mode', lambda: 'off')
    decisions = []
    session = _session(_queued_prompt_generation=2, running=True)

    def run(*_args, **_kwargs):
        lease = next(iter(session['_approval_task_leases']))
        decisions.append(approval_context.current_approval_task_context().text)
        session['_queued_prompt_generation'] = 3
        decisions.append(approval_context.approval_task_revoked())
        session['_queued_prompt_generation'] = 2
        lease.revoke()  # the local Stop path revokes the same owner lease
        decisions.append(approval_context.approval_task_revoked())
        return {'final_response': 'done', 'messages': []}

    session['agent'] = SimpleNamespace(session_id='session-key', run_conversation=run,
                                       clear_interrupt=lambda: None)
    monkeypatch.setitem(server._sessions, 'queued-fence', session)
    try:
        assert server._run_prompt_submit('queued-rid', 'queued-fence', session, 'Edit only alpha.py',
                                         approval_origin=('local:stdio', 'submit-id', None,
                                                          'Edit only alpha.py', 2))
        assert decisions == ['Edit only alpha.py', True, True]
    finally:
        server._sessions.pop('queued-fence', None)
