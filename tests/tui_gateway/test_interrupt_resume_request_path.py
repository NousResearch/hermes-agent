"""Which contract checks a pausing ``session.interrupt``, by the socket it arrives on.

The caller that pauses a turn (``resume: true``) is a host plugin's launcher. It writes the frame on the socket it
relays to the backend it launched, a ``hermes serve`` (``--isolated``). That app has no session authority, so
``tui_gateway.ws`` hands every frame to ``server.dispatch`` and the standalone contract (``SessionInterruptParams``)
checks it: ``resume`` is taken there and the turn's ``message.complete`` says ``paused``.

A gateway-owned socket (``hermes-gateway-v1``, the Desktop's own dial to the local gateway) checks the same method
name against the canonical contract (``CanonicalControlParams``), which is a closed key set without ``resume``:
it answers ``4001 invalid_params`` and stops nothing. Nothing sends ``resume`` there; a caller that pauses a
gateway-owned turn needs the canonical contract extended first.
"""

from __future__ import annotations

import asyncio
import json
import threading
import types

import pytest

from tests.tui_gateway import test_auto_continue as turns
from tests.waiting import async_wait_until
from tui_gateway import server
from tui_gateway import ws as ws_mod

turn_env = turns.turn_env
marker_home = turns.marker_home


def pause_frame(sid: str, **extra) -> dict:
    """The launcher's frame: its own request id, the live session id and ``resume``; nothing else."""
    return {"jsonrpc": "2.0", "id": "sandbox-call-1", "method": "session.interrupt",
            "params": {"session_id": sid, "resume": True, **extra}}


class Socket:
    """A WebSocket of an app without a session authority (``hermes serve``)."""

    def __init__(self):
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.scope: dict = {}
        self.app = types.SimpleNamespace(state=types.SimpleNamespace(session_authority=None))

    async def accept(self):
        pass

    async def send_text(self, line):
        self.sent.append(json.loads(line))

    async def receive_text(self):
        item = await self.inbound.get()
        if item is None:
            raise ws_mod._WebSocketDisconnect()
        return json.dumps(item)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_a_serve_socket_takes_the_pause(monkeypatch):
    monkeypatch.setattr(ws_mod, "_start_backend_liveness", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "_ensure_lease_watcher", lambda: None)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    sid = "pause-serve-socket"
    session = server._sessions[sid] = {
        "agent": None, "agent_ready": threading.Event(), "agent_error": None, "history": [],
        "history_lock": threading.RLock(), "session_key": sid, "agent_build_started": True, "running": True,
        "inflight_turn": {"user": "read the plan"}, "_run_thread": types.SimpleNamespace(is_alive=lambda: True),
    }
    sock = Socket()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    try:
        sock.inbound.put_nowait(pause_frame(sid))
        await async_wait_until(lambda: any(f.get("id") == "sandbox-call-1" for f in sock.sent), timeout=20,
                               what="the reply to the pause frame")
    finally:
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 20)
        server._sessions.pop(sid, None)

    ready = next(f["params"]["payload"] for f in sock.sent if (f.get("params") or {}).get("type") == "gateway.ready")
    assert "session_authority" not in ready  # so the Desktop speaks the standalone wire on this socket
    reply = next(f for f in sock.sent if f.get("id") == "sandbox-call-1")
    assert reply.get("result", {}).get("status") == "interrupted", reply
    assert session["_turn_cancel_requested"] is True  # the turn stops; the next test runs one to its completion


def test_a_turn_paused_on_a_serve_socket_completes_paused(turn_env, marker_home, monkeypatch):
    frames, replies = [], []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setattr(server, "_start_usage_ticker",
                        lambda *a: (threading.Event(), types.SimpleNamespace(join=lambda: None)))
    transport = types.SimpleNamespace(write=lambda frame: frames.append(frame) or True)
    assert ws_mod._authority_connection(Socket(), transport, None, False) is None  # frames go to server.dispatch
    agent = types.SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
                                  interrupt=lambda *a, **k: None)

    def run(message, **kwargs):
        # Mid-turn, as when the launcher answers a grant while the tool call waits. The fixture's inline thread
        # reports itself dead while it runs the turn; a real turn thread is alive here.
        session["_run_thread"] = types.SimpleNamespace(is_alive=lambda: True)
        replies.append(server.dispatch(pause_frame("pause-turn"), transport))
        return {"final_response": "", "interrupted": True, "messages": []}

    agent.run_conversation = run
    session = turns._session(agent=agent, running=True)
    server._sessions["pause-turn"] = session
    try:
        server._run_prompt_submit("request", "pause-turn", session, "read the plan")
    finally:
        server._sessions.pop("pause-turn", None)

    assert replies and replies[0].get("result", {}).get("status") == "interrupted", replies
    completes = [f["params"]["payload"] for f in frames if (f.get("params") or {}).get("type") == "message.complete"]
    assert [(p.get("status"), p.get("paused")) for p in completes] == [("interrupted", True)]


@pytest.mark.parametrize("extra", [{}, {"execution_generation": 3}],
                         ids=["the launcher's frame", "with the Desktop adapter's generation"])
def test_a_gateway_owned_socket_refuses_resume_as_an_unknown_key(extra):
    from gateway.session_controls import AuthorityConnection

    connection = object.__new__(AuthorityConnection)
    connection.authority = types.SimpleNamespace(profile_id="home", instance_id="instance")
    connection.actor = types.SimpleNamespace(profile_id="home")
    connection._operator, connection._identity, connection._siblings = False, {}, {}

    reply = asyncio.run(connection.dispatch(pause_frame("gateway-owned", **extra)))

    assert reply["error"]["code"] == 4001, reply
    assert reply["error"]["data"] == {"reason": "invalid_params", "fields": ["resume"]}
