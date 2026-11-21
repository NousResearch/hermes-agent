"""Compute-host dispatch failures retain the inline fallback (#147198)."""

import threading

from tests.tui_gateway.test_tui_gateway_server import _session
from tui_gateway import server


def test_prompt_submit_fails_open_inline_when_compute_host_dispatch_breaks(monkeypatch):
    class _BrokenSupervisor:
        def submit_turn(self, frame, *, on_complete=None):
            if on_complete is not None:
                on_complete(
                    {
                        "type": "turn.error",
                        "request_id": frame["request_id"],
                        "reason": "send_failed",
                        "message": "broken pipe",
                    }
                )
            raise BrokenPipeError("broken pipe")

    class _ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            assert self._target is not None
            self._target()

    session = _session(agent=None, agent_ready=threading.Event())
    server._sessions["iso-fallback"] = session
    inline_calls = []
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _BrokenSupervisor())
    # This fallback test has no persistent session, even if an earlier test opened a store.
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)
    # The deferred inline-fallback thread now waits via the patient variant.
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda _session, _rid, _sid: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **_kwargs: inline_calls.append((rid, sid, text)),
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    try:
        resp = server.handle_request(
            {
                "id": "fallback-turn",
                "method": "prompt.submit",
                "params": {"session_id": "iso-fallback", "text": "hello"},
            }
        )
    finally:
        server._sessions.pop("iso-fallback", None)

    assert resp == {
        "jsonrpc": "2.0",
        "id": "fallback-turn",
        "result": {"status": "streaming"},
    }
    assert inline_calls == [("fallback-turn", "iso-fallback", "hello")]
    assert session.get("_compute_host_active") is not True
