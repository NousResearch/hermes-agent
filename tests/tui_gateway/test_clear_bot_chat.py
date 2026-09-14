"""End-to-end RPC invariant for clearing the canonical Bot Chat (#110229)."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from hermes_state import SessionDB
from tools.bot_live_delivery import claim_pending_delivery, complete_delivery, deliver_to_live_owner
from tui_gateway import server


def test_clear_bot_chat_refuses_busy_then_clears_same_named_runtime(monkeypatch, tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="desktop")
    db.set_session_title("root", "Bot Chat")
    db.append_message("root", "user", "old question")
    db.end_session("root", "compression")
    db.create_session("tip", source="desktop", parent_session_id="root")
    db.append_message("tip", "assistant", "old summary", _compressed_summary=True)

    compressor = SimpleNamespace(on_session_reset=Mock())
    codex_session = SimpleNamespace(close=Mock())
    agent = SimpleNamespace(
        context_compressor=compressor,
        model="keep-model",
        _codex_session=codex_session,
        _session_messages=[{"role": "assistant", "content": "old summary"}],
        _last_flushed_db_idx=1,
        _flushed_db_message_ids={1},
        _db_flush_scan_prefix=[{"role": "assistant", "content": "old summary"}],
        _codex_reasoning_replay_enabled=False,
    )
    session = {
        "agent": agent,
        "history": list(agent._session_messages),
        "display_history_prefix": [{"role": "user", "content": "old question"}],
        "history_lock": threading.RLock(),
        "history_version": 4,
        "running": True,
        "session_key": "tip",
        "source": "desktop",
        "profile_home": None,
    }
    server._sessions["live"] = session
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *_args: False)
    monkeypatch.setattr(server, "_session_info", lambda current, _session=None: {"model": current.model})
    emitted = Mock()
    broadcast = Mock()
    monkeypatch.setattr(server, "_emit", emitted)
    monkeypatch.setattr(server, "_broadcast_global_event", broadcast)

    try:
        refused = server.handle_request({"id": "busy", "method": "session.clear_bot_chat", "params": {}})
        assert refused["error"]["code"] == 4023
        assert db.get_messages_as_conversation("tip", include_ancestors=True, include_compacted=True)

        session["running"] = False
        owner = {
            "profile_home": str(tmp_path.resolve()),
            "session_id": "tip",
            "lease_id": "lease",
            "live_session_id": "live",
        }
        delivery = deliver_to_live_owner(tmp_path, owner, "late result")
        pending = server.handle_request({"id": "pending", "method": "session.clear_bot_chat", "params": {}})
        assert pending["error"]["code"] == 4023
        assert db.get_messages_as_conversation("tip", include_ancestors=True, include_compacted=True)
        assert claim_pending_delivery(tmp_path, owner)["delivery_id"] == delivery["delivery_id"]
        complete_delivery(tmp_path, delivery["delivery_id"], status="cancelled")

        cleared = server.handle_request({"id": "clear", "method": "session.clear_bot_chat", "params": {}})
    finally:
        server._sessions.pop("live", None)
        db.close()

    assert cleared["result"] == {"cleared": True, "messages_cleared": 2}
    assert session["history"] == []
    assert session["display_history_prefix"] == []
    assert session["history_version"] == 5
    assert agent.model == "keep-model"
    assert agent._session_messages == []
    assert agent._last_flushed_db_idx == 0
    assert agent._flushed_db_message_ids == set()
    assert agent._db_flush_scan_prefix is None
    assert agent._codex_reasoning_replay_enabled is True
    assert agent._codex_session is None
    codex_session.close.assert_called_once_with()
    compressor.on_session_reset.assert_called_once_with()
    emitted.assert_called_once_with("session.info", "live", {"model": "keep-model"})
    broadcast.assert_called_once_with("sessions.changed", {})
