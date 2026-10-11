"""Real host launcher for conditional IPC tests; only provider/bootstrap work is substituted."""
from __future__ import annotations

import time
import threading
from pathlib import Path
from unittest.mock import MagicMock

from agent import conversation_loop, process_bootstrap, relay_cwd
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from run_agent import AIAgent
from tui_gateway import compute_host, server, server_requests
import model_tools


def make_agent(sid, key, **kwargs):
    db = kwargs.get("session_db") or SessionDB(Path(get_hermes_home()) / "state.db")
    db.create_session(key, source="tui")
    return AIAgent(model="test-model", provider="openai", api_key="test-key",
                   base_url="https://example.invalid/v1", quiet_mode=True, skip_memory=True,
                   skip_context_files=True, skip_background_review=True, save_trajectories=False,
                   platform="gui", session_id=key, session_db=db)


def loop(agent, text, *args, **kwargs):
    agent.__dict__.setdefault("_test_calls", []).append(text)
    if text == "hold":
        server._emit("message.delta", next(sid for sid, s in server._sessions.items() if s["agent"] is agent),
                     {"text": "holding"})
        deadline = time.monotonic() + 20
        while not agent._interrupt_requested and time.monotonic() < deadline:
            time.sleep(.01)
    return {"final_response": "done", "messages": [{"role": "user", "content": text},
             {"role": "assistant", "content": "done"}], "interrupted": bool(agent._interrupt_requested)}


def rotate(host, frame, session):
    agent = session["agent"]
    old = agent.session_id
    agent.session_id = "away"
    agent.session_id = old
    return {}


def replace(host, frame, session):
    with server._sessions_lock:
        server._sessions[frame["sid"]] = dict(session)
    return {}


def request(host, frame, session, *, foreign=False):
    req = server_requests.ServerRequest("foreign-session" if foreign else frame["sid"], "clarify", {"prompt": "choose"}, qids=["one", "two"])
    with server_requests._lock:
        server_requests._open[req.id] = req
    return {"id": req.id}


def notice(host, frame, session):
    host._transport.write({"jsonrpc": "2.0", "method": "event", "params": {
        "type": "notice", "session_id": frame["sid"], "payload": {"text": "reserved observation", "level": "info"}}})
    return {}


def lease_barrier(host, frame, session):
    entered, release = threading.Event(), threading.Event()
    session["_test_lease_entered"], session["_test_lease_release"] = entered, release
    db = session["agent"]._session_db
    acquire = db.acquire_session_turn_lease
    def acquiring(*a, **k):
        entered.set()
        assert release.wait(15)
        return acquire(*a, **k)
    db.acquire_session_turn_lease = acquiring
    return {}


def lease_release(host, frame, session):
    session["_test_lease_release"].set()
    return {}


def inspect(host, frame, session):
    return {"members": len(host._conditional._members), "logical_live": sum(
                type(peer).__name__ == "HostBoundPeer" for peer in server._session_live_transports(session)),
            "running": session["running"],
            "messages": session["agent"]._session_db.get_messages(session["session_key"]),
            "requests": {rid: req.result for rid, req in server_requests._open.items()},
            "lease_wait": bool(session.get("_test_lease_entered") and session["_test_lease_entered"].is_set()),
            "calls": session["agent"].__dict__.get("_test_calls", []),
            "holder": getattr(session["agent"], "_active_session_turn_lease_holder", None)}


def membership_freeze(host, frame, session):
    manager = getattr(host._conditional, "_membership", None)
    if manager is not None:
        manager.stop.set()
    return {}


def membership_expire(host, frame, session):
    # Advance the lease boundary without wall-clock sleeps, including on the old candidate.
    with host._conditional._lock:
        for peer, _record in host._conditional._members.values():
            with getattr(peer, "_lease_lock", host._conditional._lock):
                peer._expires_at = 0
    manager = getattr(host._conditional, "_membership", None)
    if manager is not None:
        manager.poll()
    return inspect(host, frame, session)


def test_control(host, frame):
    from functools import partial
    handlers = {"rotate": rotate, "replace": replace, "request": request, "foreign_request": partial(request, foreign=True),
                "notice": notice, "lease_barrier": lease_barrier, "lease_release": lease_release, "inspect": inspect, "membership_freeze": membership_freeze, "membership_expire": membership_expire}
    result = handlers[frame["test_action"]](host, frame, server._sessions[frame["sid"]])
    host._reply("control.ack", frame["sid"], frame["request_id"], result=result)


def main():
    model_tools.get_tool_definitions = lambda *a, **k: []
    model_tools.check_toolset_requirements = lambda *a, **k: {}
    process_bootstrap.OpenAI = MagicMock()
    relay_cwd.resolve_relay_scope_cwds = lambda *a: ("", "")
    conversation_loop.run_conversation = loop
    server._make_agent = make_agent
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_tts_stream_begin",
                 "_register_session_cwd", "_session_todo_state", "_hydrate_session_cwd",
                 "_wire_session_agent", "_start_session_services", "_schedule_mcp_late_refresh"):
        setattr(server, name, lambda *a, **k: None)
    server._ensure_active_session_slot = lambda *a, **k: None
    server._get_usage = lambda *a: {}
    compute_host.ComputeHost._FRAME_HANDLERS["test.control"] = "_test_control"
    compute_host.ComputeHost._test_control = test_control
    compute_host.run_host(stdout=server._real_stdout)


if __name__ == "__main__":
    main()
