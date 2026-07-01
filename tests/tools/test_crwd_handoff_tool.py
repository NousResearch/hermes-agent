"""Tests for the crwd_handoff tool module (no live Chatwoot required)."""

import json
from unittest.mock import patch

import pytest

from tools import crwd_handoff_tool as t


@pytest.fixture
def chatwoot_env(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("CHATWOOT_AGENT_TOKEN", "agent-tok")
    monkeypatch.setenv("CHATWOOT_ACCOUNT_ID", "1")
    monkeypatch.delenv("CHATWOOT_TOKEN", raising=False)


class TestAvailability:
    def test_unavailable_without_base_url(self, monkeypatch):
        monkeypatch.delenv("CHATWOOT_BASE_URL", raising=False)
        monkeypatch.setenv("CHATWOOT_AGENT_TOKEN", "x")
        assert t.check_crwd_handoff_requirements() is False

    def test_unavailable_without_token(self, monkeypatch):
        monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
        monkeypatch.delenv("CHATWOOT_AGENT_TOKEN", raising=False)
        monkeypatch.delenv("CHATWOOT_TOKEN", raising=False)
        assert t.check_crwd_handoff_requirements() is False

    def test_available_with_creds(self, chatwoot_env):
        assert t.check_crwd_handoff_requirements() is True

    def test_falls_back_to_plain_token(self, monkeypatch):
        monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
        monkeypatch.delenv("CHATWOOT_AGENT_TOKEN", raising=False)
        monkeypatch.setenv("CHATWOOT_TOKEN", "plain-tok")
        assert t.check_crwd_handoff_requirements() is True
        assert t._agent_token() == "plain-tok"


class TestResolveConversation:
    def _with_session(self, platform, chat_id):
        return patch(
            "gateway.session_context.get_session_env",
            side_effect=lambda name, default="": {
                "HERMES_SESSION_PLATFORM": platform,
                "HERMES_SESSION_CHAT_ID": chat_id,
            }.get(name, default),
        )

    def test_parses_account_and_conversation(self, chatwoot_env):
        with self._with_session("chatwoot", "7:42"):
            assert t._resolve_conversation() == ("7", "42")

    def test_bare_id_uses_account_env(self, chatwoot_env):
        with self._with_session("chatwoot", "42"):
            assert t._resolve_conversation() == ("1", "42")

    def test_wrong_platform_returns_none(self, chatwoot_env):
        with self._with_session("telegram", "7:42"):
            assert t._resolve_conversation() == (None, None)

    def test_missing_chat_id_returns_none(self, chatwoot_env):
        with self._with_session("chatwoot", ""):
            assert t._resolve_conversation() == (None, None)


class TestComposeNote:
    def test_includes_reason_and_summary(self):
        note = t._compose_note("frustrated member", "Tried refresh + incognito, still stuck.")
        assert "frustrated member" in note
        assert "still stuck" in note
        assert "human agent" in note.lower()

    def test_defaults_reason_when_blank(self):
        note = t._compose_note("", "")
        assert "handoff requested" in note

    def test_truncates_long_summary(self):
        note = t._compose_note("x", "a" * 5000)
        assert len(note) < 5000


class TestHandler:
    def test_noop_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("CHATWOOT_BASE_URL", raising=False)
        out = json.loads(t.crwd_handoff_tool({"reason": "angry"}))
        assert out["notified"] is False
        assert out["error"] is None

    def test_noop_when_no_conversation(self, chatwoot_env):
        with patch.object(t, "_resolve_conversation", return_value=(None, None)):
            out = json.loads(t.crwd_handoff_tool({"reason": "angry"}))
        assert out["notified"] is False
        assert out["error"] is None

    def test_success_path(self, chatwoot_env):
        with patch.object(t, "_resolve_conversation", return_value=("1", "42")), patch.object(
            t, "_post_private_note", return_value=(True, "")
        ) as post:
            out = json.loads(t.crwd_handoff_tool({"reason": "rejected submission", "summary": "s"}))
        assert out["notified"] is True
        post.assert_called_once()
        assert post.call_args.args[0] == "1"
        assert post.call_args.args[1] == "42"

    def test_post_failure_degrades_gracefully(self, chatwoot_env):
        with patch.object(t, "_resolve_conversation", return_value=("1", "42")), patch.object(
            t, "_post_private_note", return_value=(False, "HTTP 403")
        ):
            out = json.loads(t.crwd_handoff_tool({"reason": "angry"}))
        # Never a hard error — the coach still hands off to the member.
        assert out["notified"] is False
        assert out["error"] is None
