"""QQ Bot proactive MEDIA delivery via send_message / _send_to_platform (#37315)."""

from __future__ import annotations

import asyncio
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from plugins.platforms.qqbot.constants import (
    MEDIA_TYPE_FILE,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    MEDIA_TYPE_VOICE,
    MSG_TYPE_MEDIA,
)
from plugins.platforms.qqbot.send import _qqbot_media_file_type
from plugins.platforms.qqbot.send import send_qqbot as _send_qqbot
from tools.send_message_tool import _send_to_platform


def _pconfig():
    return SimpleNamespace(
        token="client-secret",
        extra={"app_id": "app-123", "client_secret": "client-secret"},
        enabled=True,
    )


class TestQqbotMediaFileType:
    def test_image_ext(self):
        assert _qqbot_media_file_type("/tmp/a.PNG", False) == MEDIA_TYPE_IMAGE

    def test_video_ext(self):
        assert _qqbot_media_file_type("/tmp/a.mp4", False) == MEDIA_TYPE_VIDEO

    def test_voice_flag_or_ext(self):
        assert _qqbot_media_file_type("/tmp/a.ogg", True) == MEDIA_TYPE_VOICE
        assert _qqbot_media_file_type("/tmp/a.opus", False) == MEDIA_TYPE_VOICE

    def test_generic_file(self):
        assert _qqbot_media_file_type("/tmp/report.pdf", False) == MEDIA_TYPE_FILE


class TestSendToPlatformQqbotMediaRouting:
    def test_media_only_routes_to_send_qqbot_not_allowlist_error(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        media = [(str(img), False)]

        mock_send = AsyncMock(
            return_value={
                "success": True,
                "platform": "qqbot",
                "chat_id": "openid-1",
            }
        )
        with patch(
            "tools.send_message_tool._plugin_standalone_sender",
            return_value=(mock_send, None),
        ):
            result = asyncio.run(
                _send_to_platform(
                    Platform.QQBOT,
                    _pconfig(),
                    "openid-1",
                    "",
                    media_files=media,
                )
            )

        assert result["success"] is True
        assert "only supported for" not in str(result)
        mock_send.assert_awaited_once()
        assert mock_send.await_args.kwargs.get("media_files") == media

    def test_text_plus_media_uses_caption_not_omission_warning(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"jpeg-bytes")
        media = [(str(img), False)]

        mock_send = AsyncMock(
            return_value={"success": True, "platform": "qqbot", "chat_id": "oid"}
        )
        with patch(
            "tools.send_message_tool._plugin_standalone_sender",
            return_value=(mock_send, None),
        ):
            result = asyncio.run(
                _send_to_platform(
                    Platform.QQBOT,
                    _pconfig(),
                    "oid",
                    "hello caption",
                    media_files=media,
                )
            )

        assert result.get("success") is True
        assert not any("omitted" in w for w in result.get("warnings", []))
        assert mock_send.await_count == 1
        call = mock_send.await_args
        assert call.kwargs.get("media_files") == media
        assert call.kwargs.get("caption") == "hello caption"
        assert call.args[2] == ""


def _token_client(post_handler):
    """httpx.AsyncClient fake whose ``post`` delegates to ``post_handler(url, json)``."""

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            return post_handler(url, json)

    class _HttpxMod:
        AsyncClient = _Client

    return _HttpxMod()


def _sdk_module(texts):
    """``QQApiClient`` stub. The class name is the send path's SDK check."""

    class QQApiClient:
        def __init__(self, app_id, client_secret, log_tag=None):
            return None

        def setup(self, http):
            return None

        async def send_text(self, kind, chat_id, message, *, markdown=False, retries=1):
            # Guild is tried first and fails; only a delivered message is recorded.
            if kind != "c2c":
                raise RuntimeError(kind)
            texts.append(message)
            return {"id": "text-mid"}

    module = types.ModuleType("qqbot_agent_sdk")
    module.QQApiClient = QQApiClient
    module.configure = lambda **_kwargs: None
    return module


class TestSendQqbotMedia:
    def test_media_path_calls_deliver(self, tmp_path):
        img = tmp_path / "logo.png"
        img.write_bytes(b"png-data")

        deliver = AsyncMock(
            return_value={
                "success": True,
                "platform": "qqbot",
                "chat_id": "user-openid",
                "message_id": "m-1",
                "chat_type": "c2c",
            }
        )

        with patch.dict("sys.modules", {"httpx": _token_client(lambda url, json: None), "qqbot_agent_sdk": _sdk_module([])}), patch(
            "plugins.platforms.qqbot.send._qqbot_deliver_one_media",
            new=deliver,
        ):
            result = asyncio.run(
                _send_qqbot(
                    _pconfig(),
                    "user-openid",
                    "",
                    media_files=[(str(img), False)],
                    caption="see this",
                )
            )

        assert result["success"] is True
        deliver.assert_awaited_once()
        _args, kwargs = deliver.await_args
        assert kwargs.get("caption") == "see this"
        assert "user-openid" in _args
        assert str(img) in _args

    def test_deliver_one_media_falls_back_c2c_to_group(self, tmp_path):
        from plugins.platforms.qqbot.send import _qqbot_deliver_one_media

        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        calls = []

        class _Uploader:
            async def upload(self, chat_type, chat_id, source, file_type, file_name=None):
                calls.append(("upload", chat_type, file_type))
                if chat_type == "c2c":
                    raise RuntimeError("c2c rejected")
                return "FI-99"

        async def _fake_send_media(client, chat_type, chat_id, file_info, caption=None):
            calls.append(("send", chat_type, file_info, caption))
            return {"id": "mid-2"}

        with patch(
            "plugins.platforms.qqbot.send._media_uploader",
            return_value=_Uploader(),
        ), patch(
            "plugins.platforms.qqbot.send._qqbot_send_media_message",
            new=AsyncMock(side_effect=_fake_send_media),
        ):
            result = asyncio.run(
                _qqbot_deliver_one_media(
                    object(),
                    "gid",
                    str(img),
                    False,
                    caption="cap",
                )
            )

        assert result["success"] is True
        assert result["chat_type"] == "group"
        assert ("upload", "c2c", MEDIA_TYPE_IMAGE) in calls
        assert ("upload", "group", MEDIA_TYPE_IMAGE) in calls
        assert ("send", "group", "FI-99", "cap") in calls


class TestQqbotMediaFailureFallsBackToText:
    """Screen finding: media-delivery failure must not lose the caption/message text
    (main degraded to text + omission warning; the media path returned a bare error)."""

    def test_missing_media_file_still_delivers_caption_as_text(self, tmp_path):
        missing = tmp_path / "gone.png"  # never written: stale path
        texts = []

        with patch.dict("sys.modules", {"httpx": _token_client(lambda url, json: None), "qqbot_agent_sdk": _sdk_module(texts)}):
            result = asyncio.run(
                _send_qqbot(
                    _pconfig(),
                    "user-openid",
                    "",
                    media_files=[(str(missing), False)],
                    caption="the caption text",
                )
            )

        assert result.get("success") is True, result
        assert texts.count("the caption text") == 1, texts
        assert any("omitted" in w or "skipping" in w for w in result.get("warnings", [])), result

    def test_upload_quota_error_still_delivers_caption_as_text(self, tmp_path):
        img = tmp_path / "big.png"
        img.write_bytes(b"png")

        async def _deliver(client, chat_id, media_path, is_voice, caption=None):
            return {"error": "QQBot media send failed (guild/channel native media is unsupported; "
                             "tried c2c and group): c2c: QQ Bot API error [400] "
                             "/v2/users/user-openid/files: daily upload quota exceeded"}

        texts = []

        with patch.dict("sys.modules", {"httpx": _token_client(lambda url, json: None), "qqbot_agent_sdk": _sdk_module(texts)}), patch(
            "plugins.platforms.qqbot.send._qqbot_deliver_one_media", new=AsyncMock(side_effect=_deliver)
        ):
            result = asyncio.run(
                _send_qqbot(
                    _pconfig(),
                    "user-openid",
                    "",
                    media_files=[(str(img), False)],
                    caption="urgent notice",
                )
            )

        assert result.get("success") is True, result
        assert texts.count("urgent notice") == 1, texts
        assert result.get("warnings"), "media failure must be surfaced as a warning"

    def test_non_caption_message_text_sent_before_failed_media_is_not_duplicated(self, tmp_path):
        img = tmp_path / "x.png"
        img.write_bytes(b"x")
        texts = []

        with patch.dict("sys.modules", {"httpx": _token_client(lambda url, json: None), "qqbot_agent_sdk": _sdk_module(texts)}), patch(
            "plugins.platforms.qqbot.send._qqbot_deliver_one_media",
            new=AsyncMock(return_value={"error": "QQBot media send failed: too large"}),
        ):
            result = asyncio.run(
                _send_qqbot(
                    _pconfig(),
                    "user-openid",
                    "read this text",
                    media_files=[(str(img), False)],
                )
            )

        # Main's contract: the already-delivered text survives; the media failure becomes a
        # warning on the result, not a bare error that discards the send.
        assert result.get("success") is True, result
        assert texts.count("read this text") == 1, texts
        assert result.get("warnings"), "media failure must be surfaced as a warning"


class TestQqbotSendMediaMessageBody:
    def test_body_uses_msg_type_media(self):
        from plugins.platforms.qqbot.send import _qqbot_send_media_message

        captured = {}

        class _Client:
            async def post_c2c_message(self, chat_id, msg, *, keyboard=None):
                captured["chat_id"] = chat_id
                captured["body"] = msg.to_dict()
                return {"id": "x"}

        asyncio.run(
            _qqbot_send_media_message(
                _Client(),
                "c2c",
                "u1",
                "FILEINFO",
                caption="hi",
            )
        )

        assert captured["chat_id"] == "u1"
        assert captured["body"]["msg_type"] == MSG_TYPE_MEDIA
        assert captured["body"]["media"] == {"file_info": "FILEINFO"}
        assert captured["body"]["content"] == "hi"
        assert "msg_seq" in captured["body"]
