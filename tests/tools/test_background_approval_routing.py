"""Offline lifecycle contract: real gateway turn -> dispatch -> child -> guard.

Only transport delivery, async scheduler admission and model conversation bodies
are fixtures. No actual script, producer, broker, gateway or message is executed.
"""
import asyncio
import contextvars
from concurrent.futures import Future
from types import SimpleNamespace as NS
import threading
import socket

import pytest

# Installed before runtime imports; fixture transport below never opens a socket.
def _no_network(*args, **kwargs):
    raise AssertionError('offline approval test attempted network access')
socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network

from tools import approval as ap
from tools import approval_context as ac
from tools import delegate_tool_dispatch as dispatch
from tools.delegate_tool_child_run import _ChildRun, _signal_child_stop
from gateway.run_turn_runner import TurnRunner
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.session import SessionSource
from gateway.config import Platform
import hermes_cli.config as hc


@pytest.fixture
def rig(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    home.mkdir()
    (home / 'config.yaml').write_text('approvals:\n  mode: manual\n  timeout: 3\nsecurity:\n  tirith_enabled: false\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'telegram')
    hc._LOAD_CONFIG_CACHE.clear()
    ap._gateway_queues.clear()
    ap._gateway_notify_cbs.clear()
    ap._pending.clear()
    ap._session_approved.clear()
    ap._permanent_approved.clear()
    monkeypatch.setattr(ac, '_fire_approval_hook', lambda *a, **kw: None)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture-chat', user_id='fixture-owner')
    key = 'fixture-conversation'
    messages = []
    notified = threading.Event()
    class Adapter:
        typed_command_prefix = '/'
        async def send(self, chat_id, text, **kw):
            messages.append((chat_id, text, kw))
            notified.set()
            return NS(success=True)
    ctx = NS(session_key=key, session_id='parent-model-session', source=source,
             _status_adapter=Adapter(), _status_chat_id=source.chat_id,
             _status_thread_metadata={}, message='fixture', persist_user_display_kind=None,
             persist_user_display_metadata=None, moa_config=None, inbound_message_id=None)
    turn = TurnRunner(NS(), ctx)
    monkeypatch.setattr(turn, '_native_image_run_message', lambda: 'fixture')
    def schedule(coro, *args, **kw):
        fut = Future()
        try:
            fut.set_result(asyncio.run(coro))
        except Exception as exc:
            fut.set_exception(exc)
        return fut
    monkeypatch.setattr(turn, '_schedule', schedule)
    jobs = []
    def enqueue(**kwargs):
        jobs.append((contextvars.copy_context(), kwargs))
        return {'status': 'dispatched', 'delegation_id': 'fixture-job'}
    monkeypatch.setattr('tools.async_delegation.dispatch_async_delegation_batch', enqueue)
    children = []
    results = []
    def aggregate(unit, **kwargs):
        from concurrent.futures import ThreadPoolExecutor
        def run(child):
            cr = _ChildRun(child, NS(), 0, 'fixture', None, None)
            results.append(cr.await_child()[0])
        with ThreadPoolExecutor(max_workers=len(unit.children)) as pool:
            futures = [pool.submit(contextvars.copy_context().run, run, child) for _, _, child in unit.children]
            for future in futures:
                future.result()
        return {}
    monkeypatch.setattr(dispatch, '_execute_and_aggregate', aggregate)
    def spawn(count=1, mismatch=False, live=False):
        for i in range(count):
            def conversation(**kwargs):
                token = ac.set_current_session_key('wrong-child-key') if mismatch else None
                try:
                    return ap.check_execute_code_guard('print("fixture only; never run")', 'local')
                finally:
                    if token is not None:
                        ac.reset_current_session_key(token)
            children.append(NS(session_id=f'child-{i}', run_conversation=conversation))
        unit = NS(children=[(i, {'goal': 'fixture'}, c) for i, c in enumerate(children)],
                  task_list=[{'goal': 'fixture'}] * len(children), context='', top_role='leaf', creds={'model': 'fixture'})
        def parent_conversation(*a, **kw):
            result = dispatch._dispatch_unit(unit, 'fixture-job', None, {})
            if live:
                start()
                assert notified.wait(2)
            return result
        agent = NS(run_conversation=parent_conversation)
        turn._run_conversation_with_approval(agent, [], None, None, None)
    threads = []
    def start():
        for context, job in jobs:
            thread = threading.Thread(target=context.run, args=(job['runner'],), daemon=True)
            threads.append(thread)
            thread.start()
    class Slash(GatewaySlashCommandsMixin):
        def _session_key_for_source(self, src):
            return key if src.chat_id == source.chat_id else 'different-conversation'
        async def _deliver_approval_confirmation(self, event, text, kind):
            return text
    slash = Slash()
    slash._pending_approvals = {}
    def answer(choice, request_id, actor='fixture-owner', platform=Platform.TELEGRAM):
        event = NS(source=SessionSource(platform=platform, chat_id=source.chat_id, user_id=actor),
                   get_command_args=lambda: request_id)
        handler = slash._handle_approve_command if choice == 'once' else slash._handle_deny_command
        return asyncio.run(handler(event))
    state = NS(home=home, key=key, turn=turn, spawn=spawn, start=start, messages=messages,
               notified=notified, jobs=jobs, results=results, children=children, answer=answer, source=source)
    yield state
    for child in children:
        _signal_child_stop(child, 'fixture cleanup')
    ap.unregister_gateway_notify(key)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive(), 'fixture worker did not terminate'
    hc._LOAD_CONFIG_CACHE.clear()


@pytest.mark.parametrize('mismatch', [False, True], ids=['notifier-teardown', 'child-key-mismatch'])
def test_background_approval_lifecycle(rig, mismatch):
    rig.spawn(mismatch=mismatch)
    assert ap._gateway_notify_cb(rig.key) is None, 'parent turn notifier must be removed'
    rig.start()
    if mismatch:
        # Explicit identity mismatch must fail closed, not claim delivery or borrow a route.
        for _ in range(100):
            if rig.results:
                break
            threading.Event().wait(.02)
        assert rig.results and not rig.results[0]['approved']
        assert rig.results[0].get('outcome') == 'route_unavailable'
        assert not rig.messages
        assert not ap._pending
    else:
        assert rig.notified.wait(2), 'live child lost its owner notifier after parent teardown'
        entries = ap.list_gateway_approvals(rig.key)
        assert len(entries) == 1
        rid = entries[0]['request_id']
        assert rid in rig.messages[0][1]
        rig.answer('once', rid)
        for _ in range(100):
            if rig.results:
                break
            threading.Event().wait(.02)
        assert rig.results and rig.results[0]['approved']
        assert not ap.has_blocking_approval(rig.key)
        assert ap.resolve_gateway_approval(rig.key, 'once', request_id=rid) == 0
        assert not ap._session_approved and not ap._permanent_approved


def wait_until(predicate):
    for _ in range(250):
        if predicate():
            return
        threading.Event().wait(.02)
    assert predicate(), 'fixture lifecycle did not reach expected state'


@pytest.mark.parametrize('end', ['deny', 'timeout', 'cancel', 'termination', 'session-clear',
                                 'notify-refused', 'loop-missing', 'cancel-before-start'])
def test_background_request_fails_closed(rig, monkeypatch, end):
    if end == 'notify-refused':
        async def refused(*a, **kw):
            return NS(success=False, error='fixture destination refused')
        monkeypatch.setattr(rig.turn._ctx._status_adapter, 'send', refused)
    if end == 'loop-missing':
        def missing(coro, *a, **kw):
            coro.close()
            return None
        monkeypatch.setattr(rig.turn, '_schedule', missing)
    rig.spawn()
    if end == 'cancel-before-start':
        rig.jobs[0][1]['interrupt_fn']()
    rig.start()
    no_prompt = {'notify-refused', 'loop-missing', 'cancel-before-start'}
    rid = None
    if end not in no_prompt:
        assert rig.notified.wait(2)
        entry = ap.list_gateway_approvals(rig.key)[0]
        rid = entry['request_id']
        if end == 'deny':
            rig.answer('deny', rid)
        if end == 'cancel':
            rig.jobs[0][1]['interrupt_fn']()
        if end == 'termination':
            _signal_child_stop(rig.children[0], 'fixture child termination')
        if end == 'session-clear':
            ap.clear_session(rig.key)
    wait_until(lambda: bool(rig.results))
    assert not rig.results[0]['approved']
    assert not ap.has_blocking_approval(rig.key)
    assert not ap._pending
    if rid:
        assert ap.resolve_gateway_approval(rig.key, 'once', request_id=rid) == 0


@pytest.mark.parametrize('wrong', ['actor', 'platform', 'profile', 'request', 'unqualified', 'always', 'all'])
def test_bound_concurrent_requests_do_not_cross_authority(rig, monkeypatch, wrong):
    rig.spawn(count=2)
    rig.start()
    wait_until(lambda: len(ap.list_gateway_approvals(rig.key)) == 2)
    entries = ap.list_gateway_approvals(rig.key)
    first, second = [e['request_id'] for e in entries]
    assert first != second
    if wrong == 'actor':
        rig.answer('once', first, actor='unrelated-authorized-user')
    elif wrong == 'platform':
        rig.answer('once', first, platform=Platform.DISCORD)
    elif wrong == 'profile':
        with monkeypatch.context() as m:
            m.setenv('HERMES_HOME', str(rig.home / 'other-profile'))
            rig.answer('once', first)
    elif wrong == 'request':
        rig.answer('once', '0' * 32)
    elif wrong == 'unqualified':
        assert ap.resolve_gateway_approval(rig.key, 'once') == 0
    else:
        rig.answer('once', f'{first} {wrong}')
    assert len(ap.list_gateway_approvals(rig.key)) == 2
    rig.answer('once', second)
    wait_until(lambda: len(rig.results) == 1)
    assert rig.results[0]['approved']
    assert [e['request_id'] for e in ap.list_gateway_approvals(rig.key)] == [first]
    rig.answer('deny', first)
    wait_until(lambda: len(rig.results) == 2)
    assert sum(bool(r['approved']) for r in rig.results) == 1
    assert not ap._session_approved and not ap._permanent_approved


def test_other_profile_cannot_clear_child_request(rig, monkeypatch):
    rig.spawn()
    rig.start()
    assert rig.notified.wait(2)
    rid = ap.list_gateway_approvals(rig.key)[0]['request_id']
    with monkeypatch.context() as m:
        m.setenv('HERMES_HOME', str(rig.home / 'other-profile'))
        assert not ap.list_gateway_approvals(rig.key)
        assert not ap.has_blocking_approval(rig.key)
        assert ap.get_pending_gateway_approval(rig.key) is None
        assert not ap.ack_gateway_approval(rig.key, rid)
        ap.clear_session(rig.key)
    assert [e['request_id'] for e in ap.list_gateway_approvals(rig.key)] == [rid]
    rig.answer('deny', rid)
    wait_until(lambda: bool(rig.results))


@pytest.mark.parametrize('second', ['same-script', 'per-call-tool'])
def test_one_shot_does_not_authorize_next_operation(rig, second):
    rig.spawn()
    first_results = []
    def conversation(**kwargs):
        first_results.append(ap.check_execute_code_guard('print("fixture")', 'local'))
        if second == 'same-script':
            return ap.check_execute_code_guard('print("fixture")', 'local')
        return ap.request_tool_approval('fixture-broker-call', 'fixture per-call consent; no broker invoked')
    rig.children[0].run_conversation = conversation
    rig.start()
    assert rig.notified.wait(2)
    rid = ap.list_gateway_approvals(rig.key)[0]['request_id']
    rig.answer('once', rid)
    wait_until(lambda: bool(first_results) and bool(ap.list_gateway_approvals(rig.key)))
    next_id = ap.list_gateway_approvals(rig.key)[0]['request_id']
    assert next_id != rid
    rig.answer('once', rid)
    assert ap.has_blocking_approval(rig.key)
    rig.answer('deny', next_id)
    wait_until(lambda: bool(rig.results))
    assert first_results[0]['approved'] and not rig.results[0]['approved']


def test_parent_teardown_leaves_only_live_child_wait(rig):
    from tools.approval_gateway_wait import _ApprovalEntry
    parent_entry = _ApprovalEntry({'command': 'fixture parent wait'})
    ap._gateway_queues[rig.key] = [parent_entry]
    rig.spawn(live=True)
    assert parent_entry.event.is_set()
    pending = ap.list_gateway_approvals(rig.key)
    assert len(pending) == 1 and pending[0]['background_approval']
    rig.answer('once', pending[0]['request_id'])
    wait_until(lambda: bool(rig.results))
    assert rig.results[0]['approved']


@pytest.mark.parametrize('missing', ['actor', 'notifier'])
def test_missing_route_never_borrows_parent_callback(rig, monkeypatch, missing):
    if missing == 'actor':
        rig.source.user_id = None
    else:
        monkeypatch.setattr(rig.turn, '_background_approval_notifier', lambda: None)
    rig.spawn()
    rig.start()
    wait_until(lambda: bool(rig.results))
    assert rig.results[0]['outcome'] == 'route_unavailable'
    assert not rig.messages and not ap._pending and not ap.has_blocking_approval(rig.key)


def test_synchronous_grandchild_keeps_live_owner_route(rig):
    rig.spawn()
    grandchild = NS(session_id='fixture-grandchild', run_conversation=lambda **kw:
                    ap.check_execute_code_guard('print("grandchild fixture")', 'local'))
    def conversation(**kwargs):
        return _ChildRun(grandchild, rig.children[0], 0, 'fixture', None, None).await_child()[0]
    rig.children[0].run_conversation = conversation
    rig.start()
    assert rig.notified.wait(2), 'synchronous grandchild lost background owner route'
    entry = ap.list_gateway_approvals(rig.key)[0]
    assert entry['child_id'] == grandchild.session_id
    rig.answer('once', entry['request_id'])
    wait_until(lambda: bool(rig.results))
    assert rig.results[0]['approved']
