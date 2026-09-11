"""Real Desktop RPC -> turn invocation -> independent reviewer boundary (no inference)."""
from types import SimpleNamespace
import threading
import xml.etree.ElementTree as ET

import pytest

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport
from tui_gateway.ws import WSTransport
from tools.approval_task import current_task
from tools.approval_smart import _smart_approve


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
    result = server._methods['prompt.submit']('r', {
        'session_id': 'ui-1', 'text': raw + '\nGENERATED ATTACHMENT CONTENT',
        'input_provenance': {'kind': 'desktop_composer', 'raw_text': raw}})
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


@pytest.mark.parametrize('extra', [{}, {'display_kind': 'hidden'}, {'queued': True}])
def test_generated_identical_text_has_no_evidence(desktop_turn, extra):
    session, tasks, observed = desktop_turn
    params = {'session_id': 'ui-1', 'text': 'edit the instructions', **extra}
    # Even an accidentally forwarded field is ignored on unsupported paths.
    if extra:
        params['input_provenance'] = {'kind': 'desktop_composer', 'raw_text': params['text']}
    result = server._methods['prompt.submit']('r', params)
    assert result['result']['status'] == 'streaming'
    assert tasks == [None]
    assert '<task_evidence>' not in observed[0][1]['content']

@pytest.mark.parametrize('identity', [
    {'provider': 'server-internal'}, {'user_id': 'server-internal'},
])
def test_internal_transport_cannot_supply_task(desktop_turn, identity):
    session, tasks, observed = desktop_turn
    session['transport'].auth_identity = identity
    server._methods['prompt.submit']('r', {
        'session_id': 'ui-1', 'text': 'edit',
        'input_provenance': {'kind': 'desktop_composer', 'raw_text': 'edit'}})
    assert tasks == [None]
    assert '<task_evidence>' not in observed[0][1]['content']


@pytest.mark.parametrize('failure', ['persist', 'thread', 'wait', 'cancel', 'admission'])
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
    params = {'session_id': 'ui-1', 'text': 'edit',
              'input_provenance': {'kind': 'desktop_composer', 'raw_text': 'edit'}}
    if failure == 'thread':
        with pytest.raises(RuntimeError, match='thread unavailable'):
            server._methods['prompt.submit']('r', params)
    else:
        server._methods['prompt.submit']('r', params)
    assert leases and not leases[0].active
    assert '_approval_task_lease' not in session
    assert not tasks


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
