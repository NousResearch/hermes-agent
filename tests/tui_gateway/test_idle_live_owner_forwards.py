"""Local fix: a Dashboard/legacy backend attached to a session another live surface (e.g. the
gateway's own Telegram delivery) owns, with NO durable-ledger work pending (the session is
idle, not mid-turn), used to fall straight through ``_ensure_active_session_slot`` to a flat,
silent ``SESSION_NOT_OWNED`` refusal that ``methods_prompt._slot_refusal`` never forwards
(only a canonical-ledger refusal ever reached ``owner_forward``). That left typed Dashboard
text for a Telegram-linked session dropped with no server-side trace at all. A live foreign
owner (its transport is not dead -- a dead one already takes the detached-runtime-takeover
branch) must be forwarded the same way a mid-turn canonical-ledger refusal is.

Real: the SessionDB, the legacy ``prompt.submit`` and its ledger gate. Replaced: the active-
session slot claim itself (``_claim_active_session_slot`` / ``_take_over_detached_runtime_lease``,
which depend on process-table liveness and the on-disk lease registry -- not worth wiring up
for this unit) and the owner's discovery/socket dial (same substitution the existing
``test_legacy_submit_forwards_to_owner.py`` suite uses).
"""
import asyncio
import contextlib
from types import SimpleNamespace

from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions
from tui_gateway import server
from tests.waiting import wait_until


class _OwnerClient:
    def __init__(self, authority):
        from gateway.session_controls import AuthorityConnection
        self.loop, self.events = asyncio.get_running_loop(), asyncio.Queue()
        self.conn = AuthorityConnection(authority, self, {"user_id": "desktop"})

    def write(self, frame):
        self.loop.call_soon_threadsafe(self.events.put_nowait, frame)
        return True

    def close(self):
        return None

    async def rpc(self, method, **params):
        reply = await self.conn.dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        if "error" in reply:
            from hermes_cli.gateway_client import rpc_error
            raise rpc_error(reply["error"])
        return reply["result"]


def _legacy(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_register_session_cwd",
                 "_start_agent_build"):
        monkeypatch.setattr(server, name, lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: False)
    turns, emitted = [], []
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *a: turns.append(a[1]))
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)) or True)
    created = server.handle_request({"id": "c", "method": "session.create", "params": {"source": "desktop"}})
    return created["result"]["session_id"], created["result"]["stored_session_id"], turns, emitted


def _idle_live_foreign_owner(monkeypatch, db, key, epoch):
    """A different live surface (e.g. Telegram) holds the session's slot, but it is idle: no
    admission is pending in the durable ledger, so ``_canonical_ledger_refusal`` sees nothing."""
    from hermes_cli.active_sessions import SESSION_NOT_OWNED, ActiveSessionRefusal
    monkeypatch.setattr(server, "_claim_active_session_slot",
                         lambda *_a, **_k: (None, ActiveSessionRefusal("owned elsewhere", reason=SESSION_NOT_OWNED)))
    monkeypatch.setattr(server, "_take_over_detached_runtime_lease", lambda *_a, **_k: False)

    from gateway.session_authority import LiveSession, SessionAuthority
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="owned", instance_id="gateway",
                                 db=db, epoch=epoch)
    authority.sessions[key] = LiveSession(SimpleNamespace(platform=None, user_id="desktop"), "route")
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    from tui_gateway import owner_forward

    @contextlib.asynccontextmanager
    async def connect(_home, endpoint):
        assert endpoint == "ready-owner"
        yield _OwnerClient(authority)
    monkeypatch.setattr(owner_forward, "discover_owner", lambda _home: "ready-owner")
    monkeypatch.setattr(owner_forward, "connect_owner", connect)
    return authority


def test_dashboard_submit_on_an_idle_telegram_owned_session_is_forwarded_not_dropped(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        epoch = begin_runtime_epoch(db, instance_id="gateway")
        _idle_live_foreign_owner(monkeypatch, db, key, epoch)

        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "hallo"}})

        assert "error" not in reply, reply  # before the fix: 4090 SESSION_NOT_OWNED, dropped silently
        assert reply["result"]["status"] == "queued" and reply["result"]["forwarded"] is True
        assert turns == [] and db.get_messages_as_conversation(key) == []  # nothing ran in this (losing) backend
        rows = list_session_admissions(db, session_id=key)
        assert len(rows) == 1 and rows[0]["payload"]["text"] == "hallo"
        wait_until(lambda: any(e == "session.info" for e, *_ in emitted), timeout=10)
        assert not any(e == "error" for e, *_ in emitted)
    finally:
        server._sessions.pop(sid, None)
        db.close()
