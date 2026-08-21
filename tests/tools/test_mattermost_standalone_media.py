"""Regression tests for Mattermost standalone media item shapes (#90403).

The shared media extraction path (``send_message_tool``) yields
``(file_path, is_voice)`` tuples — the shape every other standalone sender
unpacks directly. The Mattermost sender treated every non-dict item as a
bare path, so a tuple reached ``os.path.exists`` and raised ``TypeError``,
dropping the attachment instead of uploading it.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_aiohttp_resp(status, json_data=None):
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value="")
    return resp


def _make_aiohttp_session(resp):
    request_ctx = MagicMock()
    request_ctx.__aenter__ = AsyncMock(return_value=resp)
    request_ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=request_ctx)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, session


async def _send_media(media_files):
    from plugins.platforms.mattermost.adapter import _standalone_send

    pconfig = SimpleNamespace(
        token="tok-abc", extra={"url": "https://mm.example.com"}
    )
    return await _standalone_send(
        pconfig, "channel1", "see attachment", media_files=media_files
    )


class TestStandaloneSendMediaShapes:
    def test_upload_accepts_media_path_voice_flag_tuple(self, tmp_path):
        """The shared-extraction shape ``(path, is_voice)`` uploads and the
        returned file_id rides on the post payload."""
        target = tmp_path / "report.html"
        target.write_text("<html>report</html>", encoding="utf-8")

        # One resp serves both the upload POST and the final post POST.
        resp = _make_aiohttp_resp(
            201, json_data={"id": "post123", "file_infos": [{"id": "file9"}]}
        )
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATTERMOST_URL": "", "MATTERMOST_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_media([(str(target), False)]))

        assert result.get("success") is True
        assert session.post.call_count == 2
        upload_url = session.post.call_args_list[0][0][0]
        post_url = session.post.call_args_list[1][0][0]
        assert upload_url.endswith("/api/v4/files")
        assert post_url.endswith("/api/v4/posts")
        post_payload = session.post.call_args_list[1][1]["json"]
        assert post_payload["file_ids"] == ["file9"]

    def test_bare_path_string_still_uploads(self, tmp_path):
        target = tmp_path / "plain.txt"
        target.write_text("hello", encoding="utf-8")
        resp = _make_aiohttp_resp(
            201, json_data={"id": "post124", "file_infos": [{"id": "file10"}]}
        )
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATTERMOST_URL": "", "MATTERMOST_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_media([str(target)]))

        assert result.get("success") is True
        assert session.post.call_count == 2

    def test_legacy_dict_item_still_uploads(self, tmp_path):
        target = tmp_path / "dict_item.txt"
        target.write_text("hello", encoding="utf-8")
        resp = _make_aiohttp_resp(
            201, json_data={"id": "post125", "file_infos": [{"id": "file11"}]}
        )
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATTERMOST_URL": "", "MATTERMOST_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_media([{"path": str(target)}]))

        assert result.get("success") is True
        assert session.post.call_count == 2

    def test_voice_flagged_tuple_still_uploads_as_attachment(self, tmp_path):
        """Mattermost has no native voice bubble distinction (uploads are
        generic attachments), so the flag is ignored — the file must still
        be delivered rather than crash."""
        target = tmp_path / "note.ogg"
        target.write_bytes(b"\x4f\x67\x67\x53")
        resp = _make_aiohttp_resp(
            201, json_data={"id": "post126", "file_infos": [{"id": "file12"}]}
        )
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATTERMOST_URL": "", "MATTERMOST_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_media([(str(target), True)]))

        assert result.get("success") is True
        assert session.post.call_count == 2

    def test_missing_tuple_path_is_skipped_not_fatal(self):
        """An unresolvable path is skipped silently (pre-existing behavior);
        the send must still deliver the text."""
        resp = _make_aiohttp_resp(201, json_data={"id": "post127"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATTERMOST_URL": "", "MATTERMOST_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_media([("/nonexistent/path.png", False)]))

        assert result.get("success") is True
        assert session.post.call_count == 1
        assert "file_ids" not in session.post.call_args[1]["json"]
