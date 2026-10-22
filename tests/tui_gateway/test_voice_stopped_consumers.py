"""TUI-26: every Python caller of the owner's canonical ``prompt.submit`` treats the declared
``{voice_stopped: true}`` reply (a bare stop phrase typed into a LOCAL voice chat: the chat ends and
nothing is admitted) as settled with no turn, never as a malformed admission receipt."""
import asyncio
import contextlib
from types import SimpleNamespace

import pytest


class _StopClient:
    def __init__(self):
        self.calls, self.events = [], asyncio.Queue()

    async def rpc(self, method, **params):
        self.calls.append(method)
        if method == "session.resume":
            return {"session_id": params["session_id"]}
        if method == "prompt.submit":
            return {"voice_stopped": True}
        raise AssertionError(f"unexpected {method}")


def test_owner_forward_relays_nothing_after_a_voice_stopped_reply(monkeypatch, tmp_path):
    from tui_gateway import owner_forward, server
    client = _StopClient()

    @contextlib.asynccontextmanager
    async def connect(_home, _endpoint):
        yield client
    monkeypatch.setattr(owner_forward, "discover_owner", lambda _home: "ready-owner")
    monkeypatch.setattr(owner_forward, "connect_owner", connect)
    monkeypatch.setattr(owner_forward, "forwardable", lambda *_a: True)
    emitted = []
    session = {"session_key": "stored-a", "profile_home": str(tmp_path)}
    receipt = owner_forward.forward_prompt("sid", session, "stop", emit=lambda *a: emitted.append(a))
    assert receipt == {"voice_stopped": True}
    assert client.calls == ["session.resume", "prompt.submit"]
    assert emitted == [], "no turn was admitted, so nothing is relayed or settled"
    monkeypatch.setattr(owner_forward, "forward_prompt", lambda *_a, **_k: {"voice_stopped": True})
    refusal = SimpleNamespace(ledger_status="started")
    reply = server._submit_to_ledger_owner("r", "sid", session, "stop", {}, refusal)
    assert reply["result"] == {"voice_stopped": True}, reply


@pytest.mark.asyncio
async def test_attach_view_submit_and_oneshot_return_on_a_voice_stopped_reply(capsys):
    from agent.i18n import t
    from hermes_cli.gateway_chat_view import GatewayChatView
    view = GatewayChatView(_StopClient(), {"stored_session_id": "s"})
    assert await view.submit("stop") == {"voice_stopped": True}
    assert view.mine == {}
    assert t("cli.voice.stop_phrase_typed") in capsys.readouterr().out
