"""Busy input receipts from the registered RPC, with no model or live state."""
import threading
from types import SimpleNamespace

import pytest
from tui_gateway import server


@pytest.fixture
def rpc(monkeypatch):
    calls = []
    agent = SimpleNamespace(steer=lambda text: calls.append(text) or True)
    session = {'agent': agent, 'session_key': 'neutral-key', 'running': True,
               'history': [], 'history_lock': threading.RLock(), 'history_version': 0,
               'transport': None, 'attached_images': []}
    monkeypatch.setitem(server._sessions, 'neutral-sid', session)
    monkeypatch.setattr(server, '_session_compression_in_flight', lambda s: False)
    def send(text='neutral follow-up', **extra):
        return server.handle_request({'id':'neutral-rpc', 'method':'session.steer',
            'params': {'session_id':'neutral-sid', 'text':text, **extra}})
    return session, calls, send


def test_normal_receipt_names_steer_not_next_turn(rpc):
    session, calls, send = rpc
    response = send()
    assert calls == ['neutral follow-up']
    assert response['result']['status'] == 'queued'
    assert response['result']['disposition'] == 'steer'
    assert 'queued_prompt' not in session


def test_compression_receipt_names_existing_queue(rpc, monkeypatch):
    session, calls, send = rpc
    monkeypatch.setattr(server, '_session_compression_in_flight', lambda s: True)
    response = send()
    assert calls == []
    assert session['queued_prompt']['text'] == 'neutral follow-up'
    assert response['result']['status'] == 'queued'
    assert response['result']['disposition'] == 'queue'
    assert response['result']['reason'] == 'compression'


def test_turn_ended_is_explicit_refusal(rpc):
    session, calls, send = rpc
    session['running'] = False
    response = send()
    assert not calls
    assert response['result']['status'] == 'rejected'
    assert response['result']['reason'] == 'turn_ended'
    assert response['result']['delivered'] is False


def test_agent_false_does_not_claim_safe_non_delivery(rpc):
    session, _calls, send = rpc
    session['agent'].steer = lambda text: False
    response = send()
    assert response['result']['status'] == 'rejected'
    assert 'delivered' not in response['result']
    assert 'disposition' not in response['result']


def test_exception_after_accept_is_unknown_and_never_auto_replayed(rpc):
    session, calls, send = rpc
    def accept_then_raise(text):
        calls.append(text)
        raise RuntimeError('neutral ambiguous acknowledgement')
    session['agent'].steer = accept_then_raise
    response = send()
    assert response['error']['code'] == 5000
    assert calls == ['neutral follow-up']
    assert 'result' not in response
    assert 'queued_prompt' not in session


def test_legacy_steer_rejects_submission_id_without_delivery(rpc):
    _session, calls, send = rpc
    first = send(submission_id='neutral-duplicate')
    second = send(submission_id='neutral-duplicate')
    assert first['error']['code'] == second['error']['code'] == 4000
    assert calls == []


def test_repeated_legacy_request_has_no_admission_dedupe(rpc):
    _session, calls, send = rpc
    send()
    send()
    assert calls == ['neutral follow-up', 'neutral follow-up']


def test_compression_duplicate_does_not_claim_enqueued(rpc, monkeypatch):
    session, calls, send = rpc
    session['inflight_turn'] = {'user':'neutral follow-up'}
    monkeypatch.setattr(server, '_session_compression_in_flight', lambda s: True)
    response = send()
    assert not calls
    assert not session.get('queued_prompt')
    assert response['result']['status'] == 'queued'  # legacy compatibility
    assert 'disposition' not in response['result']  # existing duplicate guard did not enqueue


def test_registered_handler_reaches_real_agent_steer_slot_once(rpc):
    from agent.interrupt_control import InterruptControlMixin
    session, _calls, send = rpc
    agent = InterruptControlMixin()
    session['agent'] = agent
    response = send()
    assert response['result']['status'] == 'queued'
    assert agent._drain_pending_steer() == 'neutral follow-up'
    assert agent._drain_pending_steer() is None
    assert not session.get('queued_prompt')


def test_explicit_queue_bypasses_steer_even_if_default_steer(rpc, monkeypatch):
    session, calls, _send = rpc
    monkeypatch.setattr(server, '_load_busy_input_mode', lambda: 'steer')
    response = server._handle_busy_submit('queue-rpc', 'neutral-sid', session,
                                          'explicit next', None, queued=True)
    assert response['result']['status'] == 'queued'
    assert session['queued_prompt']['text'] == 'explicit next'
    assert not calls


def test_invalid_text_and_unknown_session_preserve_error(rpc):
    _session, calls, send = rpc
    assert send('')['error']['code'] == 4002
    response = server.handle_request({'id':'wrong', 'method':'session.steer',
                                     'params':{'session_id':'unknown-neutral', 'text':'neutral'}})
    assert 'error' in response
    assert not calls


def test_unsupported_agent_is_not_reported_as_accepted(rpc):
    session, calls, send = rpc
    session['agent'] = SimpleNamespace()
    response = send()
    assert response['error']['code'] == 4010
    assert not calls
