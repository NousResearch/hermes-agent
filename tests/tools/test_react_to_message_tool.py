"""Ownership tests for desktop message reactions."""

from unittest.mock import MagicMock

from tools import react_to_message_tool as reactions


def test_reaction_database_closes_when_write_fails(monkeypatch):
    db = MagicMock()
    db.latest_message_row_id.return_value = 42
    db.set_message_reaction.side_effect = RuntimeError("write failed")
    monkeypatch.setattr(reactions, "_open_session_db", lambda: db)
    monkeypatch.setattr(
        reactions,
        "get_session_env",
        lambda _name, _default="": "session-1",
    )

    result = reactions.react_to_message_tool("👍")

    assert "write failed" in result
    db.close.assert_called_once()


def test_native_telegram_reaction_uses_current_message(monkeypatch):
    calls = []

    class Adapter:
        async def _set_reaction(self, chat_id, message_id, emoji):
            calls.append((chat_id, message_id, emoji))
            return True

    env = {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "7221776739",
        "HERMES_SESSION_MESSAGE_ID": "123",
    }
    monkeypatch.setattr(reactions, "get_session_env", lambda name, default="": env.get(name, default))
    monkeypatch.setattr(reactions, "_live_adapter_for_current_platform", lambda _platform: Adapter())
    monkeypatch.setattr("model_tools._run_async", lambda coro: __import__("asyncio").run(coro))

    result = reactions.react_to_message_tool("👍")

    assert '"native": true' in result
    assert calls == [("7221776739", "123", "👍")]


def test_check_requirements_allows_native_platform(monkeypatch):
    monkeypatch.setattr(reactions, "_is_desktop_reactions_session", lambda: False)
    monkeypatch.setattr(reactions, "_native_reactions_available", lambda: True)

    assert reactions.check_react_requirements() is True
