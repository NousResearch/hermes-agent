"""Check host presentation timing on the real native waiter without wall-clock delays.
These tests keep presentation separate from a human answer.
"""
from types import SimpleNamespace

import pytest

from tools import clarify_gateway as cm


@pytest.fixture
def clock(monkeypatch):
    now = SimpleNamespace(value=0.0)
    monkeypatch.setattr(cm, 'time', SimpleNamespace(monotonic=lambda: now.value))
    yield now
    cm.clear_session('presentation-route')


def test_first_receipt_gives_one_full_native_wait(clock, monkeypatch):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    receipts = []

    def wait(timeout):
        clock.value += timeout
        if clock.value == 9:
            receipts.append(cm.record_presentation(entry, 'viewer', 'presented'))
        if clock.value == 10:
            receipts.append(cm.record_presentation(entry, 'viewer', 'presented'))
        return False

    monkeypatch.setattr(entry.event, 'wait', wait)
    assert cm.wait_for_response(entry.clarify_id, 10) is None
    assert clock.value == 19
    assert receipts == ['recorded', 'already_recorded']
    assert entry.response is None
    assert entry.full_wait_elapsed is True


@pytest.mark.parametrize('state', ['interaction', 'hidden'])
def test_invalidation_is_sticky_and_never_creates_an_answer(clock, monkeypatch, state):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    assert cm.record_presentation(entry, 'viewer', 'presented') == 'recorded'
    assert cm.record_presentation(entry, 'viewer', state) == 'invalidated'
    assert cm.record_presentation(entry, 'viewer', 'presented') == 'ineligible'
    monkeypatch.setattr(entry.event, 'wait', lambda timeout: setattr(clock, 'value', clock.value + timeout) or False)
    assert cm.wait_for_response(entry.clarify_id, 10) is None
    assert entry.full_wait_elapsed is False


def test_nonpresenter_hidden_does_not_withdraw_first_receipt(clock):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    assert cm.record_presentation(entry, 'first', 'presented') == 'recorded'
    assert cm.record_presentation(entry, 'second', 'hidden') == 'ineligible'
    assert cm.record_presentation(entry, 'second', 'presented') == 'already_recorded'
    assert not entry.presentation_invalidated
    assert cm.record_presentation(entry, 'second', 'interaction') == 'invalidated'


def test_receipt_at_deadline_cannot_extend_or_resurrect(clock, monkeypatch):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    statuses = []

    def wait(timeout):
        clock.value += timeout
        if clock.value == 10:
            statuses.append(cm.record_presentation(entry, 'viewer', 'presented'))
        return False

    monkeypatch.setattr(entry.event, 'wait', wait)
    assert cm.wait_for_response(entry.clarify_id, 10) is None
    assert statuses == ['ineligible']
    assert clock.value == 10
    assert cm.record_presentation(entry, 'viewer', 'presented') == 'already_resolved'


@pytest.mark.parametrize('answer', ['a', cm.SKIPPED, cm.CANCELLED])
def test_real_resolution_wins_after_receipt(clock, answer):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    cm.record_presentation(entry, 'viewer', 'presented')
    assert cm.resolve_gateway_clarify(entry.clarify_id, answer)
    cm.clear_session(entry.session_key)
    assert entry.response == answer
    assert not entry.full_wait_elapsed


def test_receipt_before_wait_uses_receipt_time_not_wait_entry(clock, monkeypatch):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    clock.value = 5
    cm.record_presentation(entry, 'viewer', 'presented')
    clock.value = 8
    monkeypatch.setattr(entry.event, 'wait', lambda timeout: setattr(clock, 'value', clock.value + timeout) or False)
    assert cm.wait_for_response(entry.clarify_id, 10) is None
    assert clock.value == 15
    assert entry.full_wait_elapsed


def test_unlimited_wait_cannot_emit_full_expiry_evidence(clock, monkeypatch):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    cm.record_presentation(entry, 'viewer', 'presented')
    monkeypatch.setattr(entry.event, 'wait', lambda timeout: cm.resolve_gateway_clarify(entry.clarify_id, 'b'))
    assert cm.wait_for_response(entry.clarify_id, 0) == 'b'
    assert entry.deadline is None and not entry.full_wait_elapsed


def test_stop_cancels_presented_wait_without_full_wait_evidence(clock, monkeypatch):
    from tools import interrupt
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    cm.record_presentation(entry, 'viewer', 'presented')
    monkeypatch.setattr(entry.event, 'wait', lambda timeout: False)
    monkeypatch.setattr(interrupt, 'is_interrupted', lambda: True)
    assert cm.wait_for_response(entry.clarify_id, 10) == cm.CANCELLED
    assert not entry.full_wait_elapsed
    assert cm.record_presentation(entry, 'viewer', 'presented') == 'already_resolved'


def test_interaction_before_receipt_is_sticky(clock):
    entry = cm.register('presentation', 'presentation-route', 'Choose', ['a', 'b'])
    assert cm.record_presentation(entry, 'viewer', 'interaction') == 'invalidated'
    assert cm.record_presentation(entry, 'viewer', 'presented') == 'ineligible'
    assert entry.presented_at is None
