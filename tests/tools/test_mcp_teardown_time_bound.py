"""A transport teardown that never finishes must not hold the MCP run loop forever (#145292).

After a keepalive failure ``_serve_session`` returns ``"reconnect"``; ``_serve_transport`` then exits
the ``ClientSession`` and the HTTP transport. Those exits had no time bound, so a close that hung
on a dead socket left the run loop waiting: no reconnect, no park, no self-probe, and the tools
already deregistered never came back until the process restarted.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from tools import mcp_tool_transport as T


class _HangingExit:
    async def __aenter__(self):
        return (object(), object())

    async def __aexit__(self, *exc):
        await asyncio.Event().wait()  # the HTTP session close that never completes


class _QuickExit:
    exited = False

    async def __aenter__(self):
        return (object(), object())

    async def __aexit__(self, *exc):
        type(self).exited = True
        return False


class _Session:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Server:
    name = "stalled-http-standin"

    def __init__(self, outcome: str | BaseException = "reconnect"):
        self._outcome = outcome

    def _session_kwargs(self):
        return {}

    async def _serve_session(self, session, connect_timeout, label):
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome

    def _reconnect_or_reraise_group(self, eg):
        return "reconnect"


@pytest.fixture
def fake_session(monkeypatch):
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "ClientSession", _Session)
    monkeypatch.setattr(T, "_TEARDOWN_TIMEOUT_S", 0.2, raising=False)


def _serve(server, transport):
    async def run():
        return await asyncio.wait_for(T.MCPServerTransportMixin._serve_transport(server, transport, "HTTP", 5.0),
                                      timeout=3)

    return asyncio.run(run())


def test_a_stalled_http_teardown_still_returns_reconnect(fake_session, caplog):
    with caplog.at_level(logging.WARNING):
        outcome = _serve(_Server(), _HangingExit())
    assert outcome == "reconnect"
    assert any("teardown" in r.getMessage() and "stalled-http-standin" in r.getMessage() for r in caplog.records)


def test_a_stalled_teardown_does_not_swallow_the_body_error(fake_session):
    with pytest.raises(RuntimeError, match="session failed"):
        _serve(_Server(RuntimeError("session failed")), _HangingExit())


def test_a_normal_teardown_is_unchanged(fake_session, caplog):
    _QuickExit.exited = False
    with caplog.at_level(logging.WARNING):
        outcome = _serve(_Server(), _QuickExit())
    assert outcome == "reconnect" and _QuickExit.exited is True
    assert not any("teardown" in r.getMessage() for r in caplog.records)
