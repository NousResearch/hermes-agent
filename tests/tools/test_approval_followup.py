"""Offline real waiting-consumer regressions; no script is executed."""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace as NS
import threading

import pytest
from tests.tools.test_background_approval_routing import rig, wait_until
from tools import approval as ap, approval_context as ac
from tools.approval_delegation import owner_for_source
from tools.approval_gateway_wait import _ApprovalEntry
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin


@contextmanager
def foreground_wait(rig):
    sent = threading.Event()
    results = []
    ap.register_gateway_notify(rig.key, lambda data: sent.set())
    def foreground():
        token = ac.set_current_session_key(rig.key)
        try:
            results.append(ap.check_execute_code_guard('print("UNRELATED inert foreground")', 'local'))
        finally:
            ac.reset_current_session_key(token)
    worker = threading.Thread(target=foreground)
    worker.start()
    try:
        assert sent.wait(2)
        with ap._lock:
            entry = next(e for e in ap._gateway_queues[rig.key] if e.lease is None)
        yield entry, results
    finally:
        ap.unregister_gateway_notify(rig.key)
        worker.join(5)
        assert not worker.is_alive()


def malformed(rid):
    return [rid[:-1], rid.upper(), rid[:8] + '-' + rid[8:], 'z' + rid[1:],
            '<' + rid + '>', 'request:' + rid, 'id=' + rid, 'request ' + rid,
            'unknown-id', '0' * 32, rid + ' extra', 'all ' + rid,
            'session ' + rid, 'always ' + rid, 'sessionalways', 'once extra']


@pytest.mark.parametrize('suffix', ['', ' session', ' always'])
def test_malformed_approval_never_resolves_either_real_wait(rig, suffix):
    rig.spawn(); rig.start(); assert rig.notified.wait(2)
    rid = ap.list_gateway_approvals(rig.key)[0]['request_id']
    with foreground_wait(rig) as (entry, results):
        for arg in malformed(rid):
            rig.answer('once', arg + suffix)
            assert entry.result is None, arg + suffix
            assert not entry.event.is_set(), arg + suffix
            assert not rig.results and not results
        rig.answer('deny', rid)
        wait_until(lambda: bool(rig.results))
        assert not rig.results[0]['approved']
        assert entry.result is None
    assert results and not results[0]['approved']


@pytest.mark.parametrize('args,choice,all_', [
    ('', 'once', False), ('once', 'once', False), ('all once', 'once', True),
    ('all', 'once', True), ('session', 'session', False),
    ('ses', 'session', False), ('always', 'always', False),
    ('permanent', 'always', False), ('permanently', 'always', False),
    ('ALL SESSION', 'session', True), ('all always', 'always', True),
    ('session all', 'session', True), ('session always', 'always', False),
])
def test_explicit_legacy_approval_forms_remain_compatible(rig, args, choice, all_):
    entries = [_ApprovalEntry({'command': 'inert first'}), _ApprovalEntry({'command': 'inert second'})]
    ap._gateway_queues[rig.key] = entries.copy()
    rig.answer('once', args)
    assert entries[0].result == choice
    assert entries[1].result == (choice if all_ else None)


class StringTrap:
    def __str__(self):
        raise AssertionError('identity coercion executed')


@pytest.mark.parametrize('field', ['user_id', 'chat_id', 'thread_id', 'session_key'])
@pytest.mark.parametrize('value', [True, False, 123, ['owner'], {'id': 'owner'}, ' ', '\t', ' owner ', StringTrap()])
def test_malformed_owner_values_rejected_without_coercion(rig, field, value):
    key = rig.key
    if field == 'session_key':
        key = value
    else:
        setattr(rig.source, field, value)
    assert owner_for_source(rig.source, key) is None


@pytest.mark.parametrize('value', [True, 123, ' ', NS(value='telegram'), StringTrap()])
def test_platform_is_not_an_arbitrary_value_wrapper(rig, value):
    rig.source.platform = value
    assert owner_for_source(rig.source, rig.key) is None


@pytest.mark.parametrize('source', [None, True, 123, {}, NS(), NS(platform=Platform.TELEGRAM)])
def test_malformed_source_is_refused(rig, source):
    assert owner_for_source(source, rig.key) is None


@pytest.mark.parametrize('value', [None, 0, 1, '', 'false', [], {}])
def test_bot_flag_must_be_explicit_false(rig, value):
    rig.source.is_bot = value
    assert owner_for_source(rig.source, rig.key) is None


@pytest.mark.parametrize('thread', [None, '', '123'])
def test_canonical_source_strings_preserved(rig, thread):
    rig.source.thread_id = thread
    owner = owner_for_source(rig.source, rig.key)
    assert owner.actor == rig.source.user_id
    assert owner.chat == rig.source.chat_id
    assert owner.thread == (thread if thread is not None else '')
    assert owner.platform == 'telegram'


@pytest.mark.parametrize('request_id', ['', False, 0, [], {}, ' ', StringTrap()])
def test_explicit_invalid_core_selector_cannot_become_fifo(rig, request_id):
    entry = _ApprovalEntry({'command': 'inert'})
    ap._gateway_queues[rig.key] = [entry]
    assert ap.resolve_gateway_approval(rig.key, 'once', request_id=request_id) == 0
    assert entry.result is None and not entry.event.is_set()


@pytest.mark.parametrize('dimension', ['actor', 'chat', 'thread', 'session', 'profile', 'platform'])
def test_mixed_wait_exact_owner_mismatch_cannot_fall_back(rig, monkeypatch, dimension):
    rig.spawn(); rig.start(); assert rig.notified.wait(2)
    rid = ap.list_gateway_approvals(rig.key)[0]['request_id']
    with foreground_wait(rig) as (entry, results):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id=rig.source.chat_id, user_id=rig.source.user_id)
        key = rig.key
        if dimension == 'session': key = 'wrong-session'
        elif dimension == 'profile': monkeypatch.setenv('HERMES_HOME', str(rig.home / 'wrong-profile'))
        else: setattr(source, {'actor': 'user_id', 'chat': 'chat_id', 'thread': 'thread_id', 'platform': 'platform'}[dimension],
                      Platform.DISCORD if dimension == 'platform' else 'wrong')
        assert ap.resolve_gateway_approval(rig.key, 'once', request_id=rid, owner=owner_for_source(source, key)) == 0
        assert entry.result is None and not entry.event.is_set()
        assert not results and not rig.results
        monkeypatch.setenv('HERMES_HOME', str(rig.home))
        rig.answer('deny', rid)
    assert results and not results[0]['approved']
