import asyncio
import time

from tui_gateway import server
from tui_gateway import ws as ws_mod
from tui_gateway.session_state import SessionState


def _fresh_session(**fields):
    """Session seeded like every real creation site: a SessionState with timestamps.

    Two reasons this must be a SessionState (not a bare dict), each a real
    contract the production disconnect path now relies on:

    * Timestamps — importing tui_gateway.server starts a module-level
      idle-reaper thread (_start_idle_reaper) that scans _sessions every 300s
      and evicts sessions whose transport is dead and whose
      created_at/last_active are older than the 6h TTL. A seed missing those
      keys reads as 0.0 ("idle since the epoch") and becomes evictable the
      instant handle_ws's finally closes the transport; in a process that
      outlives the reaper's first tick, a tick in the close→assert window
      pops the session (KeyError). Real creation sites always stamp both.
    * Attribute access — the repoint path sets ``session.transport =
      _detached_ws_transport`` (server.py, Phase 3 Step 3 attribute style);
      that assignment raises AttributeError on a plain dict, silently
      skipping the repoint so the session keeps its live WSTransport. Real
      sessions are SessionState, so the test must be too.
    """
    now = time.time()
    return SessionState({"created_at": now, "last_active": now, **fields})


def _run_disconnect(monkeypatch, seed):
    """Drive handle_ws to its disconnect `finally`, seeding sessions against the
    live WSTransport the moment it exists. Returns nothing; inspect _sessions."""
    # Disable the grace-reap Timer: detached sessions normally schedule a
    # threading.Timer via _schedule_ws_orphan_reap, which would outlive the test
    # and fire _reap during interpreter teardown — touching _sessions/DB and
    # producing spurious post-run errors under the per-file CI runner. Grace=0
    # short-circuits the Timer (see _schedule_ws_orphan_reap) so the test leaves
    # no lingering thread.
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)

    # Mirror the real _finalize_session chokepoint: it is the single place that
    # closes the slash-worker (#38095). Stub it but keep that behavior so the
    # disconnect-reap path still exercises worker teardown.
    def _fake_finalize(s, end_reason="tui_close"):
        w = s.get("slash_worker")
        if w:
            w.close()

    monkeypatch.setattr(server, "_finalize_session", _fake_finalize)

    created = []
    real_transport = ws_mod.WSTransport
    monkeypatch.setattr(
        ws_mod, "WSTransport",
        lambda ws, loop, **kw: created.append(real_transport(ws, loop, **kw)) or created[-1],
    )

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            seed(created[0])  # transport now exists; attach it to sessions
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))


def test_ws_disconnect_reaps_flagged_session_and_closes_worker(monkeypatch):
    closed = []

    class FakeWorker:
        def close(self):
            closed.append(True)

    server._sessions.clear()
    try:
        _run_disconnect(
            monkeypatch,
            lambda t: server._sessions.update(
                flagged=_fresh_session(
                    transport=t,
                    close_on_disconnect=True,
                    slash_worker=FakeWorker(),
                    session_key="k",
                )
            ),
        )
        assert "flagged" not in server._sessions
        assert closed == [True]
    finally:
        server._sessions.clear()


def test_ws_disconnect_preserves_and_repoints_reconnectable_session(monkeypatch):
    server._sessions.clear()
    try:
        _run_disconnect(
            monkeypatch,
            lambda t: server._sessions.update(
                plain=_fresh_session(
                    transport=t, close_on_disconnect=False, session_key="k"
                )
            ),
        )
        assert server._sessions["plain"]["transport"] is server._detached_ws_transport
    finally:
        server._sessions.clear()
