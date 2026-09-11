"""The plugins.hermes_bots.hide_bot_chats contract (#102625).

Bot Mode plumbing (the Desktop sweep, canonical-chat creation) pushes
``hidden=True`` through three seams. With the pref false (Bot Chats
visible), every seam must suppress a hide while an explicit un-hide still
passes; with the pref at its True default, hides behave exactly as before.
A chat made visible is always recoverable; a chat silently hidden is lost —
so the gate fails toward VISIBLE when the config is unreadable.
"""

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session  # noqa: F401  (registers the RPC methods)
from hermes_state import SessionDB

from tui_gateway import bot_hide_pref
from tui_gateway.bot_hide_pref import bot_hide_pref as pref, effective_hidden_flag


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def hide_pref(monkeypatch):
    """Point the pref cache at a value without touching the real config."""
    bot_hide_pref._CACHE.clear()
    monkeypatch.setattr(bot_hide_pref, "_CACHE", {"v": False})
    yield
    bot_hide_pref._CACHE.clear()


def _call(method: str, params: dict) -> dict:
    return srv._methods[method](1, params)


def _seed(db, sid: str) -> None:
    db.create_session(sid, source="desktop")
    db._conn.execute("UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,))
    db._conn.commit()


# ── pref resolution ──────────────────────────────────────────────────────────

def test_pref_defaults_to_hide_when_key_absent(monkeypatch, tmp_path):
    """No plugins.hermes_bots key in the config -> old behavior (hide allowed)."""
    bot_hide_pref._CACHE.clear()
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"plugins": {}}
    )
    assert pref() is True
    assert effective_hidden_flag(True) is True


def test_pref_unreadable_config_fails_toward_visible(monkeypatch):
    """An unreadable config must NEVER hide a chat (review, #102625)."""
    bot_hide_pref._CACHE.clear()

    def _boom():
        raise OSError("config disk unavailable")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
    assert pref() is False
    assert effective_hidden_flag(True) is False


def test_explicit_unhide_passes_even_when_pref_disallows_hide(monkeypatch):
    monkeypatch.setattr(bot_hide_pref, "_CACHE", {"v": False})
    assert effective_hidden_flag(False) is False


# ── session.set_hidden RPC default + gate ────────────────────────────────────

def test_set_hidden_rpc_omitted_hidden_defaults_to_visible(db):
    """A caller omitting ``hidden`` must not silently hide (#102625 footgun)."""
    _seed(db, "plain-chat")
    envelope = _call("session.set_hidden", {"session_id": "plain-chat"})
    assert "error" not in envelope, envelope
    assert envelope["result"]["hidden"] is False
    assert db.get_session("plain-chat")["hidden"] == 0


def test_set_hidden_rpc_explicit_hide_still_hides(db, hide_pref):
    """The pref suppresses generic hides, but the RPC surface keeps its contract."""
    _seed(db, "plain-chat")
    envelope = _call("session.set_hidden", {"session_id": "plain-chat", "hidden": True})
    assert "error" not in envelope, envelope
    # Bot Mode plumbing hides THIS way — the pref gate must catch it.
    assert db.get_session("plain-chat")["hidden"] == 0


def test_set_hidden_rpc_explicit_hide_hides_when_pref_default(db):
    """Pref at its default (True): behavior unchanged from before the PR."""
    _seed(db, "plain-chat")
    envelope = _call("session.set_hidden", {"session_id": "plain-chat", "hidden": True})
    assert "error" not in envelope, envelope
    assert db.get_session("plain-chat")["hidden"] == 1


# ── session.create pending_hidden gate ──────────────────────────────────────

def test_create_hidden_true_suppressed_when_pref_false(monkeypatch):
    """The Desktop bots panel's session.create {hidden:true} must not hide."""
    monkeypatch.setattr(bot_hide_pref, "_CACHE", {"v": False})
    srv._sessions.clear()
    try:
        envelope = _call(
            "session.create", {"title": "from bots panel", "hidden": True}
        )
        assert "error" not in envelope, envelope
        sessions = [s for s in srv._sessions.values() if s.get("pending_title") == "from bots panel"]
        assert sessions, "session not registered"
        assert sessions[0]["pending_hidden"] is False
    finally:
        srv._sessions.clear()


def test_create_hidden_true_sticks_when_pref_default():
    srv._sessions.clear()
    try:
        envelope = _call("session.create", {"title": "bot scratch", "hidden": True})
        assert "error" not in envelope, envelope
        sessions = [s for s in srv._sessions.values() if s.get("pending_title") == "bot scratch"]
        assert sessions, "session not registered"
        assert sessions[0]["pending_hidden"] is True
    finally:
        srv._sessions.clear()


# ── REST PATCH gate (the seam the Desktop sweep writes through) ─────────────

class _FakeDB:
    def __init__(self):
        self.hidden_calls = []

    def set_session_hidden(self, sid, hidden):
        self.hidden_calls.append((sid, hidden))
        return True

    def resolve_session_id(self, session_id):
        return session_id

    def get_session_title(self, sid):
        return "t"

    def close(self):
        pass


@pytest.mark.asyncio
async def test_patch_hidden_true_suppressed_when_pref_false(monkeypatch):
    from hermes_cli.web_routers import sessions as sessions_router
    from hermes_cli.web_models import SessionRename
    import hermes_cli.web_server_sessions as _wss

    monkeypatch.setattr(bot_hide_pref, "_CACHE", {"v": False})
    fake = _FakeDB()
    monkeypatch.setattr(
        _wss, "_open_session_db_for_profile", lambda profile, *, read_only: fake
    )
    result = await sessions_router.rename_session_endpoint(
        "bot-chat-1", SessionRename(hidden=True)
    )
    assert result["ok"] is True
    assert fake.hidden_calls == [], "hide write must be suppressed by the pref"


@pytest.mark.asyncio
async def test_patch_hidden_false_unhides_even_when_pref_false(monkeypatch):
    from hermes_cli.web_routers import sessions as sessions_router
    from hermes_cli.web_models import SessionRename
    import hermes_cli.web_server_sessions as _wss

    monkeypatch.setattr(bot_hide_pref, "_CACHE", {"v": False})
    fake = _FakeDB()
    monkeypatch.setattr(
        _wss, "_open_session_db_for_profile", lambda profile, *, read_only: fake
    )
    result = await sessions_router.rename_session_endpoint(
        "bot-chat-1", SessionRename(hidden=False)
    )
    assert result["ok"] is True
    assert fake.hidden_calls == [("bot-chat-1", False)]
