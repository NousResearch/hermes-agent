"""Ownership and runtime-safety tests for desktop message reactions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from state_store_runtime_readiness import trap_state_db_opens
from tools import react_to_message_tool as reactions


_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


def _selected_pg_profile(tmp_path, monkeypatch, *, with_dsn=True):
    root = tmp_path / ".hermes"
    root.mkdir()
    profile = root / "profiles" / "selected-pg"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    if with_dsn:
        monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    else:
        monkeypatch.delenv("HERMES_STATE_STORE_TEST_DSN", raising=False)
    return root, profile


def test_reaction_database_closes_when_write_fails(monkeypatch):
    db = MagicMock()
    db.latest_message_row_id.return_value = 42
    db.set_message_reaction.side_effect = RuntimeError("write failed")
    monkeypatch.setattr(reactions, "_postgresql_runtime_activation_error", lambda: None)
    monkeypatch.setattr(reactions, "_open_session_db", lambda: db)
    monkeypatch.setattr(
        reactions,
        "get_session_env",
        lambda _name, _default="": "session-1",
    )

    result = reactions.react_to_message_tool("👍")

    assert "write failed" in result
    db.close.assert_called_once()


def test_selected_postgresql_reaction_refuses_before_sqlite_acquisition_or_emit(tmp_path, monkeypatch):
    """Direct handler calls must not bypass the selected-PG legacy-runtime boundary."""
    root, profile = _selected_pg_profile(tmp_path, monkeypatch)
    root_db = SessionDB(db_path=root / "state.db")
    root_db.create_session("root-session", "test")
    root_db.close()
    root_size = (root / "state.db").stat().st_size
    emit = MagicMock(side_effect=AssertionError("desktop emit must not run"))
    open_session_db = MagicMock(side_effect=AssertionError("SQLite acquire must not run"))
    monkeypatch.setattr(reactions.desktop_ui, "emit", emit)
    monkeypatch.setattr(reactions, "_open_session_db", open_session_db)
    monkeypatch.setattr(reactions, "get_session_env", lambda _name, _default="": "reaction-session")

    token = set_hermes_home_override(str(profile))
    try:
        with trap_state_db_opens(root, profile) as opens:
            payload = json.loads(reactions.react_to_message_tool("👍"))
    finally:
        reset_hermes_home_override(token)

    assert payload["error_type"] == "postgresql_runtime_activation"
    assert "legacy session runtime support" in payload["error"]
    assert "tui-api-session-runtime" in payload["missing_capabilities"]
    assert "postgresql://" not in json.dumps(payload)
    assert opens == []
    assert open_session_db.call_count == 0
    assert emit.call_count == 0
    assert (root / "state.db").stat().st_size == root_size
    assert not (profile / "state.db").exists()


def test_selected_postgresql_missing_dsn_returns_bounded_error_before_acquisition_or_emit(tmp_path, monkeypatch):
    root, profile = _selected_pg_profile(tmp_path, monkeypatch, with_dsn=False)
    emit = MagicMock(side_effect=AssertionError("desktop emit must not run"))
    open_session_db = MagicMock(side_effect=AssertionError("SQLite acquire must not run"))
    monkeypatch.setattr(reactions.desktop_ui, "emit", emit)
    monkeypatch.setattr(reactions, "_open_session_db", open_session_db)
    monkeypatch.setattr(reactions, "get_session_env", lambda _name, _default="": "reaction-session")

    token = set_hermes_home_override(str(profile))
    try:
        with trap_state_db_opens(root, profile) as opens:
            payload = json.loads(reactions.react_to_message_tool("👍"))
    finally:
        reset_hermes_home_override(token)

    assert payload == {
        "error": "Message reactions cannot access the selected state-store configuration.",
        "error_type": "state_store_configuration_error",
    }
    assert opens == []
    assert open_session_db.call_count == 0
    assert emit.call_count == 0
    assert not (root / "state.db").exists()
    assert not (profile / "state.db").exists()


def test_default_sqlite_reaction_persists_and_emits_once(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_key = "reaction-session"
    seed = SessionDB(db_path=home / "state.db")
    seed.create_session(session_key, "test")
    seed.append_message(session_key, "user", "hello")
    row_id = seed.latest_message_row_id(session_key, role="user")
    emit = MagicMock()
    monkeypatch.setattr(reactions.desktop_ui, "emit", emit)
    monkeypatch.setattr(reactions, "_open_session_db", lambda: seed)
    monkeypatch.setattr(reactions, "get_session_env", lambda _name, _default="": session_key)

    assert row_id is not None
    payload = json.loads(reactions.react_to_message_tool("👍"))

    assert payload.get("success") is True, payload
    assert payload["row_id"] == row_id
    assert payload["reactions"] == [{"emoji": "👍", "author": "agent", "at": payload["reactions"][0]["at"]}]
    verify = SessionDB(db_path=home / "state.db")
    try:
        assert verify.get_message_reactions(session_key, row_id) == payload["reactions"]
    finally:
        verify.close()
        seed.close()
    emit.assert_called_once_with(
        "message.reaction",
        {"row_id": row_id, "reactions": payload["reactions"], "role": "user"},
    )


def test_malformed_reaction_row_id_preserves_registry_tool_error(monkeypatch):
    db = MagicMock()
    monkeypatch.setattr(reactions, "_postgresql_runtime_activation_error", lambda: None)
    monkeypatch.setattr(reactions, "_open_session_db", lambda: db)
    monkeypatch.setattr(reactions, "get_session_env", lambda _name, _default="": "session-1")

    result = reactions.registry.dispatch("react_to_message", {"emoji": "👍", "message_row_id": "bad"})
    assert isinstance(result, str)
    payload = json.loads(result)

    assert "invalid literal for int()" in payload["error"]
    db.close.assert_called_once()
