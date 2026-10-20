"""A direct TUI completion turn binds the producer's reserved request at review."""
import threading
from types import SimpleNamespace

import pytest

from hermes_constants import get_process_hermes_home
from tests.tui_gateway.test_failed_turn_retention import _session, turn_env
from tui_gateway import server
from tools.approval_context import (approval_task_context, current_approval_task_context,
                                    reserve_background_approval_task)
from tools.approval_smart import _smart_approve

_REAL_THREAD = threading.Thread


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


@pytest.mark.parametrize('origins', [1, 2], ids=['single-origin', 'coalesced-distinct-origins'])
@pytest.mark.usefixtures('turn_env')
def test_authenticated_desktop_process_completion_retains_command_authority(tmp_path, monkeypatch, origins):
    """The registry's process completion must carry the direct submit past turn exit."""
    import json

    from hermes_state import SessionDB
    from tests.tui_gateway.test_approval_request_origin import LoginSocket
    from tools import approval_context
    from tools.process_registry import ProcessSession, process_registry
    from tools.process_registry_notifications import format_process_notification
    from tools.terminal_tool_background import spawn_background_process
    from tools.terminal_tool import terminal_tool
    from tui_gateway.transport import bind_transport, reset_transport

    db = SessionDB(tmp_path / 'state.db')
    monkeypatch.setattr(server, '_get_db', lambda: db)
    monkeypatch.setattr(server, '_schedule_agent_build', lambda _sid: None)
    monkeypatch.setattr(server, '_schedule_session_cap_enforcement', lambda: None)
    monkeypatch.setattr(server, '_start_agent_build', lambda *_a: None)
    monkeypatch.setattr(server, '_restart_completed_failed_agent_build', lambda *_a: False)
    monkeypatch.setattr(server, '_wait_agent_for_prompt', lambda *_a: None)
    monkeypatch.setattr(server, '_start_session_services', lambda *_a: None)
    monkeypatch.setattr(approval_context, '_get_approval_mode', lambda: 'off')
    processes = [ProcessSession(id=f'proc_desktop_completion_{index}', command='printf done',
                                started_at=1700000000.0 + index) for index in range(origins)]
    spawned = []

    def spawn(_registry, **kwargs):
        proc = processes[len(spawned)]
        proc.session_key = kwargs['session_key']
        with process_registry._lock:
            process_registry._running[proc.id] = proc
        spawned.append(proc)
        return proc

    monkeypatch.setattr('tools.terminal_tool_background._spawn', spawn)
    decisions = []

    def run(*_args, **_kwargs):
        if len(spawned) < origins:
            result = json.loads(spawn_background_process(
                command='printf done', env=SimpleNamespace(env={}), env_type='local',
                effective_task_id='fixture-task', task_id='fixture-task',
                session_key=session['session_key'], workdir=None, cwd=str(tmp_path),
                effective_pty=False, notify_on_complete=True, watch_patterns=None,
                approval_note=None, pty_disabled_reason=None))
            assert result['session_id'] == spawned[-1].id, result
        else:
            inline_thread = server.threading.Thread
            server.threading.Thread = _REAL_THREAD
            try:
                result = json.loads(terminal_tool('printf ok', task_id=session['session_key']))
            finally:
                server.threading.Thread = inline_thread
            decisions.append((current_approval_task_context(), result))
        return {'final_response': 'done', 'messages': []}

    token = bind_transport(LoginSocket())
    try:
        created = server.handle_request({'id': 'create', 'method': 'session.create',
                                         'params': {'source': 'desktop', 'cwd': str(tmp_path)}})
        assert 'result' in created, created
        sid = created['result']['session_id']
        session = server._sessions[sid]
        session['agent'] = SimpleNamespace(session_id=session['session_key'],
                                           run_conversation=run, clear_interrupt=lambda: None)
        prompts = [f'Inspect build {index}' for index in range(origins)]
        for index, prompt in enumerate(prompts):
            submitted = server.handle_request({'id': f'prompt-{index}', 'method': 'prompt.submit',
                                               'params': {'session_id': sid, 'text': prompt}})
            assert submitted['result']['status'] == 'streaming', submitted
        assert len(spawned) == origins and decisions == []
        assert all(proc.approval_task_reservation is not None and
                   proc.approval_task_reservation.lease.live() for proc in processes)
        for proc in processes:
            process_registry._finish_exited(proc, 0)
        events = [process_registry.completion_queue.get(timeout=5) for _ in processes]
        assert {event['session_id'] for event in events} == {proc.id for proc in processes}
        server._notif_handle_ready(sid, session, events, set(), process_registry,
                                   format_process_notification, None)
        assert len(decisions) == origins
        assert {evidence.text for evidence, _verdict in decisions} == set(prompts)
        assert all(verdict['output'].strip() == 'ok' for _evidence, verdict in decisions), decisions
        assert all(proc.approval_task_reservation is None for proc in processes)
    finally:
        reset_transport(token)
        with process_registry._lock:
            for proc in processes:
                process_registry._running.pop(proc.id, None)
                process_registry._finished.pop(proc.id, None)
        if 'sid' in locals():
            server._sessions.pop(sid, None)
        db.close()


def test_unowned_process_completion_settles_its_reservation():
    from tools.process_registry import ProcessSession, process_registry
    from tools.approval_context import reserve_background_approval_task

    proc = ProcessSession(id='proc_orphaned_approval', command='printf done',
                          session_key='retired-session', started_at=1700000100.0,
                          exited=True, notify_on_complete=True)
    with approval_task_context('Inspect the build', identity='basic:alice', profile='owner',
                               session=proc.session_key, task_id='original-admission',
                               is_current=lambda: True):
        reservation = reserve_background_approval_task()
    proc.approval_task_reservation = reservation
    with process_registry._lock:
        process_registry._finished[proc.id] = proc
    event = {'type': 'completion', 'session_id': proc.id, 'session_key': proc.session_key,
             'started_at': proc.started_at}
    try:
        assert server._notif_handle_event('other', _session(session_key='other-session'), event,
                                          set(), process_registry, lambda _event: 'done', None)
        assert proc.approval_task_reservation is None
        assert not reservation.active()
    finally:
        reservation.release()
        with process_registry._lock:
            process_registry._finished.pop(proc.id, None)
