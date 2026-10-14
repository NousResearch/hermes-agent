"""Pinned CDP response budgets and ambiguous-reply handling."""
import json

import pytest

from tools.browser_tab_lifecycle import PinnedBrowser


class Socket:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []
        self.timeouts = []

    def send(self, payload):
        self.sent.append(json.loads(payload))

    def recv(self, timeout):
        self.timeouts.append(timeout)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)


def connect_socket(monkeypatch, socket):
    import websockets.sync.client
    monkeypatch.setattr(websockets.sync.client, 'connect', lambda *a, **kw: socket)


def test_fixture_budget_does_not_change_default_cleanup_budget(monkeypatch):
    socket = Socket([{'id': 1, 'result': {}}, {'id': 1, 'result': {}}])
    connect_socket(monkeypatch, socket)
    ws = 'ws://fixture/devtools/browser/exact'
    PinnedBrowser(ws).command('Target.getTargets')
    PinnedBrowser(ws, response_timeout=10).command('Target.createTarget', url='about:blank')
    assert 0 < socket.timeouts[0] <= 1
    assert 9 < socket.timeouts[1] <= 10


def test_late_close_reply_cannot_prove_next_request_absence(monkeypatch):
    socket = Socket([TimeoutError('ambiguous close'),
                     {'id': 1, 'result': {'success': True}},
                     {'method': 'Target.targetDestroyed', 'params': {'targetId': 'owned'}},
                     {'id': 2, 'result': {'targetInfos': [{'targetId': 'owned'}]}}])
    connect_socket(monkeypatch, socket)
    browser = PinnedBrowser('ws://fixture/devtools/browser/exact')
    with pytest.raises(TimeoutError):
        browser.close_target('owned')
    assert browser.targets() == {'owned'}
    assert [request['method'] for request in socket.sent] == ['Target.closeTarget', 'Target.getTargets']
    assert [request['id'] for request in socket.sent] == [1, 2]


def test_unrelated_frames_share_one_deadline(monkeypatch):
    from tools import browser_tab_lifecycle as lifecycle
    socket = Socket([{'method': 'Target.targetCreated'}, {'id': 99, 'result': {}}])
    connect_socket(monkeypatch, socket)
    browser = PinnedBrowser('ws://fixture/devtools/browser/exact')
    ticks = iter([0, .2, .3, .6, .7, 1.1])
    monkeypatch.setattr(lifecycle.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='CDP cleanup response'):
        browser.targets()
    assert len(socket.sent) == 1
    assert socket.timeouts == pytest.approx([.7, .3])
