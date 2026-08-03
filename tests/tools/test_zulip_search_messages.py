"""Tests for the zulip_search_messages tool."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.zulip.search_tool import (
    _check_zulip_search_requirements,
    zulip_search_messages,
)
from plugins.platforms.zulip.attachments import (
    DownloadedUpload,
    zulip_download_attachment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_zulip_env(monkeypatch):
    """Set the Zulip env vars needed for the tool."""
    monkeypatch.setenv("ZULIP_SITE_URL", "https://test.zulipchat.com")
    monkeypatch.setenv("ZULIP_BOT_EMAIL", "bot@test.zulipchat.com")
    monkeypatch.setenv("ZULIP_API_KEY", "test-api-key")


def _clear_zulip_env(monkeypatch):
    for name in ("ZULIP_SITE_URL", "ZULIP_BOT_EMAIL", "ZULIP_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _config_with_zulip(api_key="config-api-key"):
    from gateway.config import Platform

    platform_config = SimpleNamespace(
        token=None,
        api_key=api_key,
        extra={
            "site_url": "https://config.zulipchat.com",
            "bot_email": "bot@config.zulipchat.com",
        },
    )
    return SimpleNamespace(platforms={Platform("zulip"): platform_config})


def _make_messages(count: int = 3, start_id: int = 100):
    """Create synthetic Zulip messages for testing."""
    return [
        {
            "id": start_id + i,
            "sender_full_name": f"User {i + 1}",
            "sender_email": f"user{i + 1}@example.com",
            "content": f"Message content {i + 1}",
            "timestamp": 1700000000 + i * 60,
        }
        for i in range(count)
    ]


def _set_session_vars(**kwargs):
    """Set gateway session context for a test and return reset tokens."""
    from gateway.session_context import set_session_vars

    return set_session_vars(**kwargs)


def _reset_session_vars(tokens):
    """Restore the pre-test context so env fallback keeps working elsewhere."""
    for token in reversed(tokens):
        token.var.reset(token)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestZulipSearchMessages:
    """Verify the zulip_search_messages tool."""

    def test_search_by_stream_and_topic(self, monkeypatch):
        """Basic search with stream+topic narrowing returns formatted messages."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(3),
            "found_newest": True,
            "found_oldest": False,
        }

        with patch("zulip.Client", return_value=mock_client):
            result = zulip_search_messages(stream="general", topic="database")

        data = json.loads(result)
        assert data["count"] == 3
        assert data["messages"][0]["sender"] == "User 1"
        assert data["messages"][0]["id"] == 100
        assert data["messages"][0]["is_bot"] is False
        assert data["oldest_message_id"] == 100
        assert data["newest_message_id"] == 102
        assert "pagination_hint" in data

        # Verify the correct narrow was passed to the Zulip client.
        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["narrow"] == [
            ["stream", "general"],
            ["topic", "database"],
        ]

    def test_search_with_text_query(self, monkeypatch):
        """Full-text search with a query string."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(1),
            "found_newest": True,
            "found_oldest": True,
        }

        with patch("zulip.Client", return_value=mock_client):
            result = zulip_search_messages(query="postgresql", stream="general")

        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["narrow"] == [
            ["stream", "general"],
            ["search", "postgresql"],
        ]

    def test_anchor_pagination(self, monkeypatch):
        """Pagination with a numeric anchor."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(5, start_id=50),
            "found_newest": False,
            "found_oldest": False,
        }

        with patch("zulip.Client", return_value=mock_client):
            result = zulip_search_messages(
                stream="general",
                topic="database",
                anchor="42",
                num_before=5,
            )

        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["anchor"] == "42"
        assert call_args["num_before"] == 5
        assert call_args["num_after"] == 0

        data = json.loads(result)
        assert data["count"] == 5
        assert data["oldest_message_id"] == 50
        assert data["newest_message_id"] == 54

    def test_no_results(self, monkeypatch):
        """Empty result set returns descriptive note."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": [],
            "found_newest": True,
            "found_oldest": True,
        }

        with patch("zulip.Client", return_value=mock_client):
            result = zulip_search_messages(stream="nonexistent")

        data = json.loads(result)
        assert data["count"] == 0
        assert "No messages matched" in data.get("note", "")

    def test_api_error(self, monkeypatch):
        """Zulip API errors are returned as error JSON."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "error",
            "msg": "Invalid API key",
        }

        with patch("zulip.Client", return_value=mock_client):
            result = zulip_search_messages(stream="general")

        data = json.loads(result)
        assert "error" in data

    def test_missing_credentials(self, monkeypatch):
        """Missing credentials return a helpful error unless config has Zulip."""
        _clear_zulip_env(monkeypatch)

        with patch("gateway.config.load_gateway_config", return_value=SimpleNamespace(platforms={})):
            result = zulip_search_messages(stream="general")

        data = json.loads(result)
        assert "error" in data
        assert "credentials" in data["error"].lower()

    def test_uses_config_credentials(self, monkeypatch):
        """Config-backed Zulip credentials work without ZULIP_* env vars."""
        _clear_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(1),
            "found_newest": True,
            "found_oldest": True,
        }

        with patch("gateway.config.load_gateway_config", return_value=_config_with_zulip()), \
             patch("zulip.Client", return_value=mock_client) as client_cls:
            result = zulip_search_messages(stream="general")

        data = json.loads(result)
        assert data["count"] == 1
        client_cls.assert_called_once_with(
            site="https://config.zulipchat.com",
            email="bot@config.zulipchat.com",
            api_key="config-api-key",
        )

    def test_check_fn_session_platform_zulip(self, monkeypatch):
        """When session platform is 'zulip', config-backed creds make the tool available."""
        _clear_zulip_env(monkeypatch)
        tokens = _set_session_vars(platform="zulip")
        try:
            with patch("gateway.config.load_gateway_config", return_value=_config_with_zulip()):
                assert _check_zulip_search_requirements() is True
        finally:
            _reset_session_vars(tokens)

    def test_check_fn_other_platform_no_creds(self, monkeypatch):
        """On non-Zulip platforms without env vars, tool is unavailable."""
        _clear_zulip_env(monkeypatch)
        tokens = _set_session_vars(platform="telegram")
        try:
            with patch("gateway.config.load_gateway_config", return_value=SimpleNamespace(platforms={})):
                assert _check_zulip_search_requirements() is False
        finally:
            _reset_session_vars(tokens)

    def test_check_fn_allows_config_credentials_without_gateway(self, monkeypatch):
        """The search tool calls Zulip directly, so credentials are sufficient."""
        _clear_zulip_env(monkeypatch)

        with patch("gateway.config.load_gateway_config", return_value=_config_with_zulip()), \
             patch("gateway.status.is_gateway_running", return_value=False):
            assert _check_zulip_search_requirements() is True

    def test_bot_messages_flagged(self, monkeypatch):
        """Messages from the bot itself are flagged with is_bot=True."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": [
                {
                    "id": 1,
                    "sender_full_name": "Hermes Bot",
                    "sender_email": "bot@test.zulipchat.com",
                    "content": "I can help with that",
                    "timestamp": 1700000000,
                },
            ],
            "found_newest": True,
            "found_oldest": True,
        }

        with patch("zulip.Client", return_value=mock_client):
            result = zulip_search_messages(stream="general")

        data = json.loads(result)
        assert data["messages"][0]["is_bot"] is True

    def test_default_parameters(self, monkeypatch):
        """Default anchor='newest', num_before=20, num_after=0."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": [],
            "found_newest": True,
            "found_oldest": True,
        }

        with patch("zulip.Client", return_value=mock_client):
            zulip_search_messages()

        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["anchor"] == "newest"
        assert call_args["num_before"] == 20
        assert call_args["num_after"] == 0

    def test_num_after_for_surrounding_context(self, monkeypatch):
        """num_after > 0 fetches messages after the anchor for context."""
        _set_zulip_env(monkeypatch)

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(10),
            "found_newest": True,
            "found_oldest": True,
        }

        with patch("zulip.Client", return_value=mock_client):
            zulip_search_messages(anchor="500", num_before=5, num_after=5)

        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["anchor"] == "500"
        assert call_args["num_before"] == 5
        assert call_args["num_after"] == 5


class TestZulipDownloadAttachment:
    """Verify the tool that materializes files found in Zulip history."""

    @pytest.mark.asyncio
    async def test_downloads_selected_attachment(self, monkeypatch):
        _set_zulip_env(monkeypatch)
        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": [{
                "id": 123,
                "content": "[report.pdf](/user_uploads/2/a/token/report.pdf)",
            }],
        }
        downloaded = DownloadedUpload(
            path="/tmp/hermes/doc_report.pdf",
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=42,
        )

        with patch("zulip.Client", return_value=mock_client), \
             patch(
                 "plugins.platforms.zulip.attachments.download_zulip_upload",
                 new=AsyncMock(return_value=downloaded),
             ) as download:
            result = await zulip_download_attachment(message_id=123)

        data = json.loads(result)
        assert data == {
            "message_id": 123,
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 42,
            "path": "/tmp/hermes/doc_report.pdf",
        }
        download.assert_awaited_once_with(
            site_url="https://test.zulipchat.com",
            bot_email="bot@test.zulipchat.com",
            api_key="test-api-key",
            path="/user_uploads/2/a/token/report.pdf",
        )

    @pytest.mark.asyncio
    async def test_session_restricts_lookup_to_current_conversation(self, monkeypatch):
        _set_zulip_env(monkeypatch)
        tokens = _set_session_vars(
            platform="zulip",
            chat_id="123:database",
            chat_name="engineering",
        )
        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": [{
                "id": 456,
                "content": "[data.csv](/user_uploads/2/a/token/data.csv)",
            }],
        }
        downloaded = DownloadedUpload(
            path="/tmp/hermes/doc_data.csv",
            filename="data.csv",
            mime_type="text/csv",
            size_bytes=12,
        )
        try:
            with patch("zulip.Client", return_value=mock_client), \
                 patch(
                     "plugins.platforms.zulip.attachments.download_zulip_upload",
                     new=AsyncMock(return_value=downloaded),
                 ):
                result = await zulip_download_attachment(message_id=456)
        finally:
            _reset_session_vars(tokens)

        assert json.loads(result)["path"] == "/tmp/hermes/doc_data.csv"
        params = mock_client.get_messages.call_args[0][0]
        assert params["narrow"] == [["stream", "engineering"], ["topic", "database"]]

    @pytest.mark.asyncio
    async def test_rejects_message_outside_restricted_conversation(self, monkeypatch):
        _set_zulip_env(monkeypatch)
        tokens = _set_session_vars(
            platform="zulip",
            chat_id="123:database",
            chat_name="engineering",
        )
        mock_client = MagicMock()
        # The API did not return the requested message inside this narrow.
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": [{"id": 999, "content": "outside message"}],
        }
        try:
            with patch("zulip.Client", return_value=mock_client), \
                 patch(
                     "plugins.platforms.zulip.attachments.download_zulip_upload",
                     new=AsyncMock(),
                 ) as download:
                result = await zulip_download_attachment(message_id=456)
        finally:
            _reset_session_vars(tokens)

        assert "allowed conversation" in json.loads(result)["error"]
        download.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_filename_selects_matching_attachment(self, monkeypatch):
        _set_zulip_env(monkeypatch)
        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": [{
                "id": 123,
                "content": (
                    "[one.txt](/user_uploads/2/a/token/one.txt) "
                    "[two.pdf](/user_uploads/2/a/token/two.pdf)"
                ),
            }],
        }
        downloaded = DownloadedUpload(
            path="/tmp/hermes/doc_two.pdf",
            filename="two.pdf",
            mime_type="application/pdf",
            size_bytes=2,
        )
        with patch("zulip.Client", return_value=mock_client), \
             patch(
                 "plugins.platforms.zulip.attachments.download_zulip_upload",
                 new=AsyncMock(return_value=downloaded),
             ) as download:
            result = await zulip_download_attachment(message_id=123, filename="two.pdf")

        assert json.loads(result)["filename"] == "two.pdf"
        assert download.await_args.kwargs["path"].endswith("/two.pdf")


class TestZulipSearchSessionRestriction:
    """Verify that searches from Zulip sessions are restricted to the current conversation.

    This prevents a user in a private DM from asking the bot to exfiltrate
    messages from streams or other DMs the bot is subscribed to.
    """

    def test_stream_session_restricts_to_current_topic(self, monkeypatch):
        """When in a stream session, search is restricted to that stream+topic."""
        _set_zulip_env(monkeypatch)
        tokens = _set_session_vars(
            platform="zulip",
            chat_id="123:database",
            chat_name="engineering",
        )

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(2),
            "found_newest": True,
            "found_oldest": True,
        }

        try:
            with patch("zulip.Client", return_value=mock_client):
                # Agent tries to search a different stream — should be overridden.
                result = zulip_search_messages(stream="general", topic="other")
        finally:
            _reset_session_vars(tokens)

        data = json.loads(result)
        assert data["count"] == 2

        call_args = mock_client.get_messages.call_args[0][0]
        # Should be restricted to current session's stream+topic, not agent's params.
        assert call_args["narrow"] == [
            ["stream", "engineering"],
            ["topic", "database"],
        ]

    def test_dm_session_restricts_to_current_dm(self, monkeypatch):
        """When in a DM session, search is restricted to that DM."""
        _set_zulip_env(monkeypatch)
        tokens = _set_session_vars(
            platform="zulip",
            chat_id="dm:alice@example.com",
            chat_name="alice@example.com",
        )

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(1),
            "found_newest": True,
            "found_oldest": True,
        }

        try:
            with patch("zulip.Client", return_value=mock_client):
                # Agent tries to search a stream — should be overridden to DM.
                result = zulip_search_messages(stream="general", query="hello")
        finally:
            _reset_session_vars(tokens)

        data = json.loads(result)
        assert data["count"] == 1

        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["narrow"] == [
            ["pm-with", "alice@example.com"],
            ["search", "hello"],
        ]

    def test_group_dm_session_restricts_to_current_group(self, monkeypatch):
        """When in a group DM session, search is restricted to that group DM."""
        _set_zulip_env(monkeypatch)
        tokens = _set_session_vars(
            platform="zulip",
            chat_id="group_dm:alice@example.com,bob@example.com",
            chat_name="alice@example.com, bob@example.com",
        )

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(1),
            "found_newest": True,
            "found_oldest": True,
        }

        try:
            with patch("zulip.Client", return_value=mock_client):
                result = zulip_search_messages()
        finally:
            _reset_session_vars(tokens)

        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["narrow"] == [
            ["pm-with", "alice@example.com,bob@example.com"],
        ]

    def test_non_zulip_session_allows_full_search(self, monkeypatch):
        """When called from CLI (no Zulip session), agent's params are respected."""
        _set_zulip_env(monkeypatch)
        # No HERMES_SESSION_PLATFORM set — simulates CLI usage.

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(1),
            "found_newest": True,
            "found_oldest": True,
        }

        with patch("zulip.Client", return_value=mock_client):
            result = zulip_search_messages(stream="general", topic="database")

        data = json.loads(result)
        assert data["count"] == 1

        call_args = mock_client.get_messages.call_args[0][0]
        assert call_args["narrow"] == [
            ["stream", "general"],
            ["topic", "database"],
        ]

    def test_query_sanitization_strips_scope_operators(self, monkeypatch):
        """stream: and pm-with: operators are stripped in restricted sessions."""
        _set_zulip_env(monkeypatch)
        tokens = _set_session_vars(
            platform="zulip",
            chat_id="123:database",
            chat_name="engineering",
        )

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(1),
            "found_newest": True,
            "found_oldest": True,
        }

        try:
            with patch("zulip.Client", return_value=mock_client):
                result = zulip_search_messages(
                    query="stream:general pm-with:alice@example.com postgresql"
                )
        finally:
            _reset_session_vars(tokens)

        call_args = mock_client.get_messages.call_args[0][0]
        # Should have session narrow + sanitized query (scope operators removed).
        assert call_args["narrow"] == [
            ["stream", "engineering"],
            ["topic", "database"],
            ["search", "postgresql"],
        ]

    def test_query_sanitization_all_operators_removed(self, monkeypatch):
        """If query contains only scope operators, no search narrow is added."""
        _set_zulip_env(monkeypatch)
        tokens = _set_session_vars(
            platform="zulip",
            chat_id="123:database",
            chat_name="engineering",
        )

        mock_client = MagicMock()
        mock_client.get_messages.return_value = {
            "result": "success",
            "messages": _make_messages(1),
            "found_newest": True,
            "found_oldest": True,
        }

        try:
            with patch("zulip.Client", return_value=mock_client):
                result = zulip_search_messages(query="stream:general pm-with:bob@example.com")
        finally:
            _reset_session_vars(tokens)

        call_args = mock_client.get_messages.call_args[0][0]
        # Only session narrow — no search narrow since query was fully sanitized.
        assert call_args["narrow"] == [
            ["stream", "engineering"],
            ["topic", "database"],
        ]
