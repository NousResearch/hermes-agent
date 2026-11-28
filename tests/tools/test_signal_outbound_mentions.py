"""Signal outbound @mentions: the ``mentions`` param reaches the JSON-RPC send.

Signal renders a real, notifying mention only when the body carries U+FFFC at the
tag position AND the message carries a matching ``start:length:recipient`` range.
The adapter already parses mentions inbound (``_render_mentions``); these cover
the outbound half.
"""

import asyncio
import sys
from types import ModuleType

import pytest

MENTION = "0:1:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _make_httpx_mock(sent):
    """Mock httpx that records the JSON payload of every post."""

    class AsyncBaseTransport:
        pass

    class Proxy:
        pass

    class MockResp:
        status_code = 200

        def json(self):
            return {"timestamp": 1234567890}

        def raise_for_status(self):
            pass

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, *args, **kwargs):
            sent.append(kwargs.get("json") or {})
            return MockResp()

    httpx_mock = ModuleType("httpx")
    httpx_mock.AsyncClient = lambda timeout=None: MockClient()
    httpx_mock.AsyncBaseTransport = AsyncBaseTransport
    httpx_mock.Proxy = Proxy
    return httpx_mock


@pytest.fixture
def sent(monkeypatch):
    posted = []
    monkeypatch.setitem(sys.modules, "httpx", _make_httpx_mock(posted))
    return posted


EXTRA = {"http_url": "http://localhost:8080", "account": "+155****4567"}


def _params(sent):
    return [p.get("params", {}) for p in sent]


class TestSignalOutboundMentions:
    def test_mentions_reach_the_rpc_params(self, sent):
        from plugins.platforms.signal.standalone import _send_signal

        result = asyncio.run(
            _send_signal(EXTRA, "group:Zm9v", "￼ ping", mentions=[MENTION])
        )

        assert result["success"] is True
        assert [p for p in _params(sent) if p.get("mentions") == [MENTION]]

    def test_absent_when_not_requested(self, sent):
        """No mentions argument must not add the key: signal-cli rejects a malformed one."""
        from plugins.platforms.signal.standalone import _send_signal

        asyncio.run(_send_signal(EXTRA, "group:Zm9v", "no tags here"))

        assert all("mentions" not in p for p in _params(sent))

    def test_standalone_send_forwards_mentions(self, sent):
        """The registered ``standalone_sender_fn`` is what send_message calls."""
        from plugins.platforms.signal.standalone import standalone_send

        class _PConfig:
            extra = EXTRA

        result = asyncio.run(
            standalone_send(_PConfig(), "group:Zm9v", "￼ ping", mentions=[MENTION])
        )

        assert result["success"] is True
        assert [p for p in _params(sent) if p.get("mentions") == [MENTION]]

    def test_signal_policy_declares_mention_support(self):
        """send_message drops the mentions argument unless the platform's policy declares
        support (gateway/outbound/send.py), so threading it through the sender is not
        enough on its own -- the declaration is half the fix."""
        from plugins.platforms.signal.register import register

        captured = {}

        class _Ctx:
            def register_platform(self, **kwargs):
                captured.update(kwargs)

        register(_Ctx())

        assert captured["standalone_send"].mentions is True
