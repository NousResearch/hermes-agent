"""Tests for email MEDIA delivery in send_message_tool."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform


def _run(coro):
    return asyncio.run(coro)


class TestSendToPlatformEmailMedia:
    def test_email_media_routes_to_attachment_helper(self, tmp_path):
        attachment = tmp_path / "report.pdf"
        attachment.write_bytes(b"%PDF-1.4 fake")

        helper = AsyncMock(return_value={"success": True, "platform": "email", "attachments": 1})
        plain_send = AsyncMock(return_value={"success": True, "platform": "email"})

        with patch("tools.send_message_tool._send_email_with_media", helper, create=True), patch(
            "tools.send_message_tool._send_email", plain_send
        ):
            result = _run(
                _send_to_platform(
                    Platform.EMAIL,
                    SimpleNamespace(extra={}),
                    "user@example.com",
                    "See attached.",
                    media_files=[(str(attachment), False)],
                )
            )

        assert result["success"] is True
        helper.assert_awaited_once_with(
            {},
            "user@example.com",
            "See attached.",
            media_files=[(str(attachment), False)],
        )
        plain_send.assert_not_awaited()

    def test_email_media_only_message_is_deliverable(self, tmp_path):
        attachment = tmp_path / "report.pdf"
        attachment.write_bytes(b"%PDF-1.4 fake")

        helper = AsyncMock(return_value={"success": True, "platform": "email", "attachments": 1})

        with patch("tools.send_message_tool._send_email_with_media", helper, create=True):
            result = _run(
                _send_to_platform(
                    Platform.EMAIL,
                    SimpleNamespace(extra={}),
                    "user@example.com",
                    "",
                    media_files=[(str(attachment), False)],
                )
            )

        assert result["success"] is True
        helper.assert_awaited_once_with(
            {},
            "user@example.com",
            "",
            media_files=[(str(attachment), False)],
        )


class TestSendEmailWithMedia:
    def test_builds_multipart_message_with_attachments(self, tmp_path, monkeypatch):
        from tools.send_message_tool import _send_email_with_media

        monkeypatch.setenv("EMAIL_ADDRESS", "bot@example.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("EMAIL_SMTP_PORT", "587")

        attachment = tmp_path / "report.pdf"
        attachment.write_bytes(b"%PDF-1.4 fake")

        with patch("smtplib.SMTP") as smtp_cls:
            server = MagicMock()
            smtp_cls.return_value = server

            result = _run(
                _send_email_with_media(
                    {},
                    "user@example.com",
                    "See attached.",
                    media_files=[(str(attachment), False)],
                )
            )

        assert result["success"] is True
        assert result["platform"] == "email"
        assert result["attachments"] == 1

        server.send_message.assert_called_once()
        msg = server.send_message.call_args.args[0]
        assert msg.is_multipart()
        attachment_parts = [
            part for part in msg.walk()
            if part.get_content_disposition() == "attachment"
        ]
        assert len(attachment_parts) == 1
        assert attachment_parts[0].get_filename() == "report.pdf"
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("bot@example.com", "secret")
