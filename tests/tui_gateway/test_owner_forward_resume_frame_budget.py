"""#145215: the owner-forward dial accepts the whole ``session.resume`` answer.

``owner_forward.connect_owner`` dials the owner gateway itself, and its first RPC is
``session.resume``, whose answer carries the entire transcript. With the library's default
``max_size`` (1 MiB) a long Bot Chat's resume frame closes the dial with code 1009 ("frame
exceeds limit of 1048576 bytes"), so the forward fails and the user's message is lost — where
a direct client on ``gateway_client.connect_gateway`` (8 MiB) would have succeeded. The dial
must ask for the same frame budget as the normal client path, and keep the loopback dial off
the user's HTTP(S) proxy."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tui_gateway import owner_forward

FRAME_BUDGET = 8 * 1024 * 1024


@pytest.mark.asyncio
async def test_owner_dial_uses_the_client_frame_budget(monkeypatch):
    from hermes_cli import gateway_client
    import websockets.asyncio.client as ws_client
    from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL

    seen = {}
    fake_ws = SimpleNamespace(subprotocol=CANONICAL_GATEWAY_PROTOCOL)

    class _FakeClient:
        def __init__(self, ws):
            self.ws = ws

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

    @contextlib.asynccontextmanager
    async def fake_connect(url, subprotocols=None, **kwargs):
        seen.update(kwargs, url=url, subprotocols=subprotocols)
        yield fake_ws

    monkeypatch.setattr(
        gateway_client, "_session_ticket", lambda home, endpoint, **_kw: "ticket-1"
    )
    monkeypatch.setattr(
        gateway_client,
        "gateway_ws_target",
        lambda endpoint, ticket: ("ws://owner", [CANONICAL_GATEWAY_PROTOCOL]),
    )
    monkeypatch.setattr(ws_client, "connect", fake_connect)
    monkeypatch.setattr(gateway_client, "GatewayClient", _FakeClient)

    async with owner_forward.connect_owner(
        Path("/tmp/hermes-home"), "endpoint-1"
    ) as client:
        assert client.ws is fake_ws

    assert seen.get("max_size") == FRAME_BUDGET, (
        "the owner-forward dial must accept the client path's frame budget; "
        "the websockets default (1 MiB) drops long sessions' resume answers"
    )
    # Same contract as connect_gateway: loopback peer, no user proxy, bounded open.
    assert seen.get("proxy") is None
    assert seen.get("open_timeout") == 10
    assert seen["subprotocols"] == [CANONICAL_GATEWAY_PROTOCOL]
