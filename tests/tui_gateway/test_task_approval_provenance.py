"""Real Desktop RPC -> turn invocation -> independent reviewer boundary (no inference)."""
from types import SimpleNamespace
import threading
import socket
import contextvars
import xml.etree.ElementTree as ET

import pytest

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport
from tui_gateway.ws import WSTransport
from tools.approval_task import current_task
from tools.approval_smart import _smart_approve


RealThread = threading.Thread


class InlineThread:
    def __init__(self, target=None, args=(), kwargs=None, **_):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        if self.target:
            self.target(*self.args, **self.kwargs)

    def is_alive(self):
        return False

    def join(self, *_, **__):
        pass


@pytest.fixture
def desktop_turn(monkeypatch, tmp_path):
    from agent import auxiliary_client
    def unexpected_network(*args, **kwargs):
        raise AssertionError('provenance test attempted a network request')
    monkeypatch.setattr(socket.socket, 'connect', unexpected_network)
    monkeypatch.setattr(socket.socket, 'connect_ex', unexpected_network)
    monkeypatch.setattr(socket, 'create_connection', unexpected_network)
    observed = []
    tasks = []

    def reviewer(**kwargs):
        observed.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVE"))])

    def conversation(*args, **kwargs):
        tasks.append(current_task())
        assert _smart_approve('echo foo#bar; touch /tmp/review-would-execute', 'script execution') == 'approve'
        return {"final_response": "done"}

    agent = SimpleNamespace(session_id="stored-1", run_conversation=conversation, clear_interrupt=lambda: None)
    transport = WSTransport(None, None)
    ready = threading.Event()
    ready.set()
    session = dict(agent=agent, session_key="stored-1", source="desktop", transport=transport,
                   history=[], history_lock=threading.Lock(), history_version=0, running=False,
                   attached_images=[], cols=80, slash_worker=None, show_reasoning=False,
                   tool_progress_mode="all", inflight_turn=None, agent_ready=ready)
    monkeypatch.setitem(server._sessions, "ui-1", session)
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(auxiliary_client, "call_llm", reviewer)
    for name in ("_emit", "_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_sync_session_key_after_compress", "_start_agent_build"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda a: {})
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *a: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a: False)
    token = bind_transport(transport)
    try:
        yield session, tasks, observed
    finally:
        reset_transport(token)


def test_explicit_composer_evidence_reaches_reviewer_unchanged(desktop_turn):
    session, tasks, observed = desktop_turn
    raw = '  Edit <instructions> & keep the rest.  '
    result = server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': {
        'session_id': 'ui-1', 'text': raw + '\nGENERATED ATTACHMENT CONTENT',
        'input_provenance': {'kind': 'desktop_composer', 'raw_text': raw}}})
    assert 'error' not in result, result  # Contract must admit the field before the handler.
    assert result['result']['status'] == 'streaming'
    assert tasks and tasks[0].raw_text == raw
    system, user = observed[0]
    assert raw not in system['content']
    block = user['content'].split('<task_evidence>')[1].split('</task_evidence>')[0]
    evidence = ET.fromstring('<task_evidence>' + block + '</task_evidence>')
    assert evidence.findtext('raw_input') == raw
    assert 'GENERATED ATTACHMENT' not in block
    assert 'echo foo#bar; touch /tmp/review-would-execute' in user['content']
    assert '_approval_task_lease' not in session
    assert current_task() is None


@pytest.mark.parametrize('provenance', [
    None, 42, [], {}, {'kind': 'desktop_composer', 'raw_text': 'valid', 'extra': True},
    {'kind': 'other', 'raw_text': 'valid'}, {'kind': 'desktop_composer', 'raw_text': True},
    {'kind': 'desktop_composer', 'raw_text': ' '},
    {'kind': 'desktop_composer', 'raw_text': 'x' * 8193},
    {'kind': 'desktop_composer', 'raw_text': 'bad\x00'},
    {'kind': 'desktop_composer', 'raw_text': 'bad\ud800'},
])
def test_malformed_optional_evidence_keeps_valid_submit(desktop_turn, provenance):
    _, tasks, observed = desktop_turn
    result = server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': {
        'session_id': 'ui-1', 'text': 'valid', 'input_provenance': provenance}})
    assert result['result']['status'] == 'streaming', result
    assert tasks == [None]
    assert '<task_evidence>' not in observed[0][1]['content']


def test_unrelated_unknown_parameter_still_refused_before_turn(desktop_turn):
    _, tasks, observed = desktop_turn
    result = server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': {
        'session_id': 'ui-1', 'text': 'valid', 'input_provenance': {
            'kind': 'desktop_composer', 'raw_text': 'valid'}, 'unknown_parameter': 1}})
    assert result['error']['code'] == 4000, result
    assert not tasks and not observed


def test_real_worker_keeps_lease_after_build_wrapper_returns(desktop_turn, monkeypatch):
    from agent import memory_provider

    session, _, _ = desktop_turn
    entered, release = threading.Event(), threading.Event()
    workers = []
    seen = []

    def conversation(*args, **kwargs):
        return {'final_response': 'done'}

    original_invoke = server._invoke_agent

    def gate_before_invoke(*args, **kwargs):
        seen.append(current_task())
        entered.set()
        assert release.wait(5), 'worker was never released'
        return original_invoke(*args, **kwargs)

    def spawn_worker(target, *, name=None, **kwargs):
        context = contextvars.copy_context()
        worker = RealThread(target=lambda: context.run(target), name=name, daemon=True)
        workers.append(worker)
        return worker

    session['agent'].run_conversation = conversation
    monkeypatch.setattr(server, '_invoke_agent', gate_before_invoke)
    monkeypatch.setattr(memory_provider, 'spawn_context_thread', spawn_worker)
    result = server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': {
        'session_id': 'ui-1', 'text': 'edit',
        'input_provenance': {'kind': 'desktop_composer', 'raw_text': 'edit'}}})
    assert result['result']['status'] == 'streaming', result
    try:
        assert entered.wait(5), 'worker did not enter the conversation'
        lease = session.get('_approval_task_lease')
        assert lease is not None and lease.active
        assert seen[0] == lease.record
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
    assert not lease.active and '_approval_task_lease' not in session


def test_failed_build_retires_before_idle_and_successor_correction(desktop_turn, monkeypatch):
    from tools.approval_task import from_composer, install_session_task, bind_task, task_revoked

    session, _, _ = desktop_turn
    old = []
    original = from_composer

    def capture(*args):
        lease = original(*args)
        old.append(lease)
        return lease

    monkeypatch.setattr('tools.approval_task.from_composer', capture)
    successor = original('stored-1', {'kind': 'desktop_composer', 'raw_text': 'new'})
    with bind_task(successor):
        copied = contextvars.copy_context()
    publications = []

    def publish(event, *args):
        if event == 'session.info':
            publications.append((old[0].active, session.get('_approval_task_lease')))
            install_session_task(session, successor)

    monkeypatch.setattr(server, '_emit', publish)
    monkeypatch.setattr(server, '_wait_agent_for_prompt', lambda *a: {'error': {'message': 'build failed'}})
    result = server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': {
        'session_id': 'ui-1', 'text': 'edit',
        'input_provenance': {'kind': 'desktop_composer', 'raw_text': 'edit'}}})
    assert result['result']['status'] == 'streaming', result
    assert publications == [(False, None)]
    assert not old[0].active and session['_approval_task_lease'] is successor and successor.active
    session['agent'].steer = lambda text: not successor.active
    corrected = server.handle_request({'id': 's', 'method': 'session.steer', 'params': {
        'session_id': 'ui-1', 'text': 'stop'}})
    assert corrected['result']['status'] == 'queued', corrected
    assert copied.run(task_revoked)
    assert not successor.active and '_approval_task_lease' not in session


@pytest.mark.parametrize('extra', [{}, {'display_kind': 'hidden'}, {'queued': True}])
def test_generated_identical_text_has_no_evidence(desktop_turn, extra):
    session, tasks, observed = desktop_turn
    params = {'session_id': 'ui-1', 'text': 'edit the instructions', **extra}
    # Even an accidentally forwarded field is ignored on unsupported paths.
    if extra:
        params['input_provenance'] = {'kind': 'desktop_composer', 'raw_text': params['text']}
    result = server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': params})
    assert result['result']['status'] == 'streaming'
    assert tasks == [None]
    assert '<task_evidence>' not in observed[0][1]['content']

@pytest.mark.parametrize('identity', [
    {'provider': 'server-internal'}, {'user_id': 'server-internal'},
])
def test_internal_transport_cannot_supply_task(desktop_turn, identity):
    session, tasks, observed = desktop_turn
    session['transport'].auth_identity = identity
    server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': {
        'session_id': 'ui-1', 'text': 'edit',
        'input_provenance': {'kind': 'desktop_composer', 'raw_text': 'edit'}}})
    assert tasks == [None]
    assert '<task_evidence>' not in observed[0][1]['content']


@pytest.mark.parametrize('excluded', ['other_source', 'non_ws', 'hidden', 'busy', 'truncated',
                                      'isolated', 'isolated_fallback'])
def test_dispatch_exclusions_never_mint_task(desktop_turn, monkeypatch, excluded):
    session, tasks, observed = desktop_turn
    if excluded == 'other_source':
        session['source'] = 'tui'
    elif excluded == 'non_ws':
        token = bind_transport(object())
    elif excluded == 'hidden':
        session['hidden'] = True
    elif excluded == 'busy':
        session['running'] = True
        monkeypatch.setattr(server, '_handle_busy_submit', lambda *a, **k: {'result': {'status': 'queued'}})
    elif excluded == 'truncated':
        monkeypatch.setattr(server, '_truncate_history_for_submit', lambda *a: (None, {}))
    else:
        monkeypatch.setattr(server, '_session_uses_compute_host', lambda *a: True)
        monkeypatch.setattr(server, '_submit_prompt_to_compute_host', lambda *a, **k: (
            {'result': {'status': 'streaming'}} if excluded == 'isolated'
            else {'error': {'message': 'offline'}}))
    params = {'session_id': 'ui-1', 'text': 'edit',
              'input_provenance': {'kind': 'desktop_composer', 'raw_text': 'edit'}}
    if excluded == 'truncated':
        params.update(truncate_before_user_ordinal=0, confirm_truncate=True)
    try:
        result = server.handle_request({'id': 'r', 'method': 'prompt.submit', 'params': params})
        assert 'result' in result, result
        assert tasks == ([] if excluded in {'busy', 'isolated'} else [None])
        assert all('<task_evidence>' not in messages[1]['content'] for messages in observed)
        assert '_approval_task_lease' not in session or session['_approval_task_lease'] is None
    finally:
        if excluded == 'non_ws':
            reset_transport(token)


@pytest.mark.parametrize('failure', ['persist', 'thread', 'wait', 'cancel', 'admission',
                                     'retirement', 'scheduling', 'prepare', 'prebody', 'finalizer'])
def test_early_exits_revoke_captured_generation(desktop_turn, monkeypatch, failure):
    from tools import approval_task
    session, tasks, observed = desktop_turn
    leases = []
    original = approval_task.from_composer
    def capture(*args):
        lease = original(*args)
        leases.append(lease)
        return lease
    monkeypatch.setattr(approval_task, 'from_composer', capture)
    if failure == 'persist':
        monkeypatch.setattr(server, '_persist_session_row_for_submit', lambda *a: {'error': 'failed'})
    if failure == 'thread':
        def fail_start(self):
            raise RuntimeError('thread unavailable')
        monkeypatch.setattr(InlineThread, 'start', fail_start)
    if failure == 'wait':
        monkeypatch.setattr(server, '_wait_agent_for_prompt', lambda *a: {'error': {'message': 'failed'}})
    if failure == 'cancel':
        def cancel(*args):
            session['_turn_cancel_requested'] = True
        monkeypatch.setattr(server, '_wait_agent_for_prompt', cancel)
    if failure == 'admission':
        monkeypatch.setattr(server, '_admit_prompt_turn', lambda *a: None)
    if failure == 'retirement':
        from contextlib import nullcontext
        monkeypatch.setattr(server, '_session_turn_admission', lambda *a: nullcontext(False))
    if failure == 'scheduling':
        monkeypatch.setattr(server, '_start_session_work', lambda *a, **k: None)
    if failure == 'prepare':
        monkeypatch.setattr(server, '_prepare_turn_input', lambda *a: None)
    if failure == 'prebody':
        monkeypatch.setattr(server, '_record_turn_marker', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('prebody')))
    if failure == 'finalizer':
        monkeypatch.setattr(server, '_finish_turn', lambda *a: (_ for _ in ()).throw(RuntimeError('finalizer')))
    params = {'session_id': 'ui-1', 'text': 'edit',
              'input_provenance': {'kind': 'desktop_composer', 'raw_text': 'edit'}}
    if failure in {'thread', 'prebody', 'finalizer'}:
        with pytest.raises(RuntimeError, match='thread unavailable|prebody|finalizer'):
            server._methods['prompt.submit']('r', params)
    else:
        server._methods['prompt.submit']('r', params)
    assert leases and not leases[0].active
    assert '_approval_task_lease' not in session
    assert not tasks or failure == 'finalizer'


@pytest.mark.parametrize("path", ["redirect", "steer", "build", "slash"])
@pytest.mark.parametrize("accepted", [True, False])
def test_correction_rpc_revokes_inflight_copied_reviewer(desktop_turn, monkeypatch, path, accepted):
    import contextvars
    from agent import auxiliary_client
    from tools.approval_context import set_current_session_key, reset_current_session_key
    from tools.approval_task import bind_task, from_composer, task_revoked

    session, _, _ = desktop_turn
    lease = from_composer("stored-1", {"kind": "desktop_composer", "raw_text": "original task"})
    session["_approval_task_lease"] = lease
    session["running"] = True
    publication_liveness = []

    def publish_correction(text):
        publication_liveness.append(lease.active)
        return accepted

    session["agent"].steer = publish_correction
    session["agent"].redirect = publish_correction
    enqueue = server._enqueue_prompt

    def publish_queue(*args, **kwargs):
        publication_liveness.append(lease.active)
        return enqueue(*args, **kwargs)

    monkeypatch.setattr(server, "_enqueue_prompt", publish_queue)
    session["agent"]._supports_active_turn_redirect = True
    if path == "build":
        session["agent"] = None
    responses = []

    def reviewer(**kwargs):
        assert "<task_evidence>" in kwargs["messages"][1]["content"]
        if path == "slash":
            response = server._methods["slash.exec"]("correction", {
                "session_id": "ui-1", "command": "/steer stop editing" if accepted else "/steer"})
        else:
            response = server._methods["session.steer" if path == "steer" else "session.redirect"](
                "correction", {"session_id": "ui-1", "text": "stop editing" if accepted or path != "build" else " "})
        responses.append(response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVE"))])

    monkeypatch.setattr(auxiliary_client, "call_llm", reviewer)
    token = set_current_session_key("stored-1")
    try:
        with bind_task(lease):
            copied = contextvars.copy_context()
        verdict = copied.run(_smart_approve, "echo harmless", "test review")
        assert responses
        dispatched = accepted or path in {"redirect", "steer"}
        assert publication_liveness == ([False] if dispatched else [])
        assert verdict == ("escalate" if dispatched else "approve"), responses
        assert copied.run(task_revoked) is dispatched
        assert lease.active is not dispatched
        assert (session.get("_approval_task_lease") is lease) is not dispatched
        assert copied.run(current_task) == (None if dispatched else lease.record)
    finally:
        reset_current_session_key(token)


@pytest.mark.parametrize('path', ['steer', 'redirect', 'build', 'slash'])
def test_throwing_correction_callback_sees_revoked_lease(desktop_turn, monkeypatch, path):
    from tools.approval_task import from_composer, task_revoked, bind_task

    session, _, _ = desktop_turn
    lease = from_composer('stored-1', {'kind': 'desktop_composer', 'raw_text': 'original'})
    session['_approval_task_lease'] = lease
    session['running'] = True
    observed = []

    def throw(*args, **kwargs):
        observed.append((lease.active, task_revoked()))
        raise RuntimeError('publication failed')

    if path == 'build':
        session['agent'] = None
        monkeypatch.setattr(server, '_enqueue_prompt', throw)
    else:
        session['agent'].steer = throw
        session['agent'].redirect = throw
        session['agent']._supports_active_turn_redirect = True
    method = 'slash.exec' if path == 'slash' else ('session.redirect' if path in {'redirect', 'build'} else 'session.steer')
    params = {'session_id': 'ui-1', **({'command': '/steer stop'} if path == 'slash' else {'text': 'stop'})}
    with bind_task(lease):
        if path == 'build':
            with pytest.raises(RuntimeError, match='publication failed'):
                server._methods[method]('r', params)
        else:
            server._methods[method]('r', params)
        assert task_revoked()
    assert observed == [(False, True)]
    assert '_approval_task_lease' not in session
