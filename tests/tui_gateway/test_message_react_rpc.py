"""message.react RPC: session-key fallback for rehydrated / rotated sessions.

Regression coverage for #80670 — reacting in a resumed conversation failed
with 4040 because the registry entry's ``session_key`` was missing (desktop
rows predating the column) or stale (compression rotation), while the durable
rows live under the routed session id / lineage tip.
"""

import contextlib
import threading
from unittest.mock import MagicMock, patch

import pytest

_HEART = "\u2764\ufe0f"
_THUMBS = "\U0001f44d"
_LAUGH = "\U0001f602"


@pytest.fixture()
def server():
    # Mirror tests/tui_gateway/test_protocol.py: mock the heavy modules for
    # the *initial* import only, then snapshot/restore the RPC registry.
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._live_transports.clear()


class _ReactionDB:
    """Minimal SessionDB stand-in that records every write it receives."""

    def __init__(self, results=None, tips=None, latest=None):
        self.results = results or {}  # session key -> reactions list (or None)
        self.tips = tips or {}        # session key -> lineage tip
        self.latest = latest or {}    # role -> row_id
        self.calls = []

    def set_message_reaction(self, session_id, row_id, emoji, author="user"):
        self.calls.append((session_id, row_id, emoji, author))
        return self.results.get(session_id)

    def latest_message_row_id(self, session_id, *, role="user", offset=0, require_text=True):
        return self.latest.get(role)

    def resolve_resume_session_id(self, key):
        return self.tips.get(key, key)


def _register(server, sid, session_key):
    server._sessions[sid] = {"session_key": session_key, "history_lock": threading.Lock()}


def _call(server, **params):
    return server._methods["message.react"]("rid-1", params)


def test_react_falls_back_to_routed_sid_when_session_key_missing(server, monkeypatch):
    """Resumed legacy desktop session: session_key=None, rows under the sid."""
    db = _ReactionDB(results={"sess-abc": [{"emoji": _HEART, "author": "user"}]})
    monkeypatch.setattr(server, "_session_db", lambda session: contextlib.nullcontext(db))
    _register(server, "sess-abc", None)

    resp = _call(server, session_id="sess-abc", row_id=7, emoji=_HEART)

    assert db.calls == [("sess-abc", 7, _HEART, "user")]
    assert resp["result"]["row_id"] == 7


def test_react_retries_lineage_tip_after_rotated_key_miss(server, monkeypatch):
    """Compaction rotated the live key: write misses under the parent, lands
    under the continuation tip. Row ids are globally unique, so the retry
    targets the exact row the client addressed."""
    db = _ReactionDB(
        results={"child-2": [{"emoji": _THUMBS, "author": "user"}]},
        tips={"parent-1": "child-2"},
    )
    monkeypatch.setattr(server, "_session_db", lambda session: contextlib.nullcontext(db))
    _register(server, "parent-1", "parent-1")

    resp = _call(server, session_id="parent-1", row_id=7, emoji=_THUMBS)

    assert db.calls == [("parent-1", 7, _THUMBS, "user"), ("child-2", 7, _THUMBS, "user")]
    assert resp["result"]["row_id"] == 7


def test_react_newest_role_uses_fallback_key(server, monkeypatch):
    """The role-based (no row_id) path resolves through the fallback key too."""
    db = _ReactionDB(results={"sess-abc": [{"emoji": _LAUGH, "author": "user"}]}, latest={"user": 42})
    monkeypatch.setattr(server, "_session_db", lambda session: contextlib.nullcontext(db))
    _register(server, "sess-abc", None)

    resp = _call(server, session_id="sess-abc", newest_role="user", emoji=_LAUGH)

    assert db.calls == [("sess-abc", 42, _LAUGH, "user")]
    assert resp["result"]["row_id"] == 42


def test_react_still_4040_when_row_exists_nowhere(server, monkeypatch, caplog):
    """A genuinely stale row id still fails loudly (with a diagnostic log)."""
    db = _ReactionDB(results={}, tips={"parent-1": "child-2"})
    monkeypatch.setattr(server, "_session_db", lambda session: contextlib.nullcontext(db))
    _register(server, "parent-1", "parent-1")

    resp = _call(server, session_id="parent-1", row_id=999, emoji=_HEART)

    assert resp["error"]["code"] == 4040
    assert db.calls == [("parent-1", 999, _HEART, "user"), ("child-2", 999, _HEART, "user")]
    assert "message.react: no row" in caplog.text
