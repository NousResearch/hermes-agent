"""QQ Bot proactive MEDIA delivery via send_message / _send_to_platform (#37315)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from gateway.platforms.qqbot.constants import (
    MEDIA_TYPE_FILE,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    MEDIA_TYPE_VOICE,
    MSG_TYPE_MEDIA,
)
from tools.send_message_tool import (
    _qqbot_media_file_type,
    _send_qqbot,
    _send_to_platform,
)


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

        with patch(
            "tools.send_message_tool._send_qqbot",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "platform": "qqbot",
                    "chat_id": "openid-1",
                }
            ),
        ) as mock_send:
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

        with patch(
            "tools.send_message_tool._send_qqbot",
            new=AsyncMock(
                return_value={"success": True, "platform": "qqbot", "chat_id": "oid"}
            ),
        ) as mock_send:
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


class TestSendQqbotMedia:
    def test_media_path_calls_deliver_after_token(self, tmp_path):
        img = tmp_path / "logo.png"
        img.write_bytes(b"png-data")

        token_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"access_token": "at-1"},
        )

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json=None, headers=None):
                assert "getAppAccessToken" in url or url.endswith("getAppAccessToken")
                return token_resp

        class _HttpxMod:
            AsyncClient = _Client

        deliver = AsyncMock(
            return_value={
                "success": True,
                "platform": "qqbot",
                "chat_id": "user-openid",
                "message_id": "m-1",
                "chat_type": "c2c",
            }
        )

        with patch.dict("sys.modules", {"httpx": _HttpxMod()}), patch(
            "tools.send_message_tool._qqbot_deliver_one_media",
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
        assert kwargs.get("caption") == "see this" or (
            len(_args) >= 6 and _args[5] == "see this"
        )
        # chat_id + path
        assert "user-openid" in _args
        assert str(img) in _args

    def test_deliver_one_media_falls_back_c2c_to_group(self, tmp_path):
        from tools.send_message_tool import _qqbot_deliver_one_media

        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        calls = []

        async def _fake_upload_local(
            client, headers, chat_type, chat_id, media_path, file_type
        ):
            calls.append(("upload", chat_type, file_type))
            if chat_type == "c2c":
                raise RuntimeError("c2c rejected")
            return {"file_info": "FI-99"}

        async def _fake_send_media(
            client, headers, chat_type, chat_id, file_info, caption=None
        ):
            calls.append(("send", chat_type, file_info, caption))
            return {"id": "mid-2"}

        with patch(
            "tools.send_message_tool._qqbot_upload_local_file",
            new=AsyncMock(side_effect=_fake_upload_local),
        ), patch(
            "tools.send_message_tool._qqbot_send_media_message",
            new=AsyncMock(side_effect=_fake_send_media),
        ):
            result = asyncio.run(
                _qqbot_deliver_one_media(
                    client=None,
                    headers={"Authorization": "QQBot t"},
                    chat_id="gid",
                    media_path=str(img),
                    is_voice=False,
                    caption="cap",
                )
            )

        assert result["success"] is True
        assert result["chat_type"] == "group"
        assert ("upload", "c2c", MEDIA_TYPE_IMAGE) in calls
        assert ("upload", "group", MEDIA_TYPE_IMAGE) in calls
        assert ("send", "group", "FI-99", "cap") in calls


class TestQqbotSendMediaMessageBody:
    def test_body_uses_msg_type_media(self):
        from tools.send_message_tool import _qqbot_send_media_message

        captured = {}

        async def _fake_api(client, headers, method, path, body=None, timeout=30.0):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {"id": "x"}

        with patch(
            "tools.send_message_tool._qqbot_api_json",
            new=AsyncMock(side_effect=_fake_api),
        ):
            asyncio.run(
                _qqbot_send_media_message(
                    None,
                    {},
                    "c2c",
                    "u1",
                    "FILEINFO",
                    caption="hi",
                )
            )

        assert captured["method"] == "POST"
        assert captured["path"] == "/v2/users/u1/messages"
        assert captured["body"]["msg_type"] == MSG_TYPE_MEDIA
        assert captured["body"]["media"] == {"file_info": "FILEINFO"}
        assert captured["body"]["content"] == "hi"
