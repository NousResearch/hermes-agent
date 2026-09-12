"""Owner-approved disjoint grammar, real waiting consumers, inert payloads."""
import pytest
from tests.tools.test_background_approval_routing import rig, wait_until
from tests.tools.test_approval_followup import foreground_wait, malformed
from tools import approval as ap
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.mark.parametrize('mixed', [False, True])
def test_invalid_deny_syntax_never_resolves_any_request(rig, mixed):
    if mixed:
        rig.spawn(); rig.start(); assert rig.notified.wait(2)
        with ap._lock:
            child = ap._gateway_queues[rig.key][0]
        rid = child.data['request_id']
    else:
        child = None
        rid = 'abcdef0123456789abcdef0123456789ab'
    with foreground_wait(rig) as (entry, results):
        invalid = malformed(rid) + [arg + ' --reason explicit text' for arg in malformed(rid)] + [
            '--reason', 'all --reason', rid + ' --reason', '--reason   ',
            '--unknown', '--Reason why', '--reason=why', 'all --unknown why',
            rid + ' --unknown why', rid + ' all', rid + ' reason without marker',
            'all all', 'all wrong directory', 'that path is still in use',
            'extra --reason text', rid + ' extra --reason text',
        ]
        for args in invalid:
            rig.answer('deny', args)
            assert entry.result is None and not entry.event.is_set(), args
            assert not results and not rig.results, args
            if child:
                assert child.result is None and not child.event.is_set(), args
                with ap._lock:
                    assert child in ap._gateway_queues[rig.key] and entry in ap._gateway_queues[rig.key]
        if child:
            rig.answer('deny', rid)
            wait_until(lambda: bool(rig.results))
            assert not rig.results[0]['approved']
            assert entry.result is None


@pytest.mark.parametrize('target', ['', 'all', 'child'])
@pytest.mark.parametrize('reason', [None, 'path is still in use', '--unknown is literal reason text'])
def test_valid_deny_selects_only_intended_real_wait(rig, target, reason):
    rig.spawn(); rig.start(); assert rig.notified.wait(2)
    with ap._lock:
        child = ap._gateway_queues[rig.key][0]
    rid = child.data['request_id']
    with foreground_wait(rig) as (entry, results):
        selector = rid if target == 'child' else target
        args = selector + ((' --reason ' + reason) if reason is not None else '')
        rig.answer('deny', args)
        selected, untouched = (child, entry) if target == 'child' else (entry, child)
        assert selected.result == 'deny' and selected.event.is_set()
        assert selected.reason == reason
        assert untouched.result is None and not untouched.event.is_set()
        if target != 'child':
            rig.answer('deny', rid)
        wait_until(lambda: bool(rig.results))
        assert not rig.results[0]['approved']
        if target == 'child':
            if reason is not None:
                assert reason in rig.results[0]['message']
            rig.answer('deny', args)  # stale exact ID cannot deny the foreground
            assert entry.result is None and not entry.event.is_set()
    assert results and not results[0]['approved']


@pytest.mark.parametrize('dimension', ['actor', 'profile'])
@pytest.mark.parametrize('suffix', ['', ' --reason explicit reason'])
def test_bound_deny_owner_mismatch_has_no_foreground_fallback(rig, monkeypatch, dimension, suffix):
    rig.spawn(); rig.start(); assert rig.notified.wait(2)
    with ap._lock:
        child = ap._gateway_queues[rig.key][0]
    with foreground_wait(rig) as (entry, results):
        if dimension == 'profile':
            monkeypatch.setenv('HERMES_HOME', str(rig.home / 'wrong-profile'))
        rig.answer('deny', child.data['request_id'] + suffix,
                   actor='wrong-actor' if dimension == 'actor' else 'fixture-owner')
        for pending in (entry, child):
            assert pending.result is None and not pending.event.is_set()
        assert not results and not rig.results
        monkeypatch.setenv('HERMES_HOME', str(rig.home))
        rig.answer('deny', child.data['request_id'])


def test_reason_is_explicit_single_line_capped_and_all_remains_foreground_only(rig):
    entries = [_ApprovalEntry({'command': 'inert first'}), _ApprovalEntry({'command': 'inert second'})]
    ap._gateway_queues[rig.key] = entries.copy()
    reason = 'word\n' + 'x' * 300
    rig.answer('deny', 'ALL --reason ' + reason)
    assert all(e.result == 'deny' and e.event.is_set() for e in entries)
    assert all(e.reason == ('word ' + 'x' * 300)[:280] for e in entries)
