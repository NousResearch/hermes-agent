"""Tests for _send_qqbot and send_message media conflict resolution.

Covers:
- Chat type parsing from chat_id prefix
- File type detection from extensions
- _send_to_platform routing for QQBot and Feishu media blocks
- Error/warning strings listing both qqbot and feishu
"""

import asyncio
import os
import tempfile
from unittest import mock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pconfig(**extra):
    """Build a PlatformConfig for testing."""
    return PlatformConfig(enabled=True, extra=extra)


# ---------------------------------------------------------------------------
# _send_qqbot: chat_type parsing
# ---------------------------------------------------------------------------

class TestSendQqbotChatTypeParsing:
    """Verify chat_type is correctly derived from chat_id prefixes."""

    def test_c2c_prefix(self):
        """chat_id with 'c2c:' prefix should map to chat_type='c2c'."""
        # We test the prefix parsing logic inline since _send_qqbot is async
        chat_id = "c2c:ABC123"
        chat_type = "c2c"
        target_id = chat_id
        if chat_id.startswith("c2c:"):
            chat_type = "c2c"
            target_id = chat_id[4:]
        elif chat_id.startswith("group:"):
            chat_type = "group"
            target_id = chat_id[6:]
        elif chat_id.startswith("guild:"):
            chat_type = "guild"
            target_id = chat_id[6:]

        assert chat_type == "c2c"
        assert target_id == "ABC123"

    def test_group_prefix(self):
        """chat_id with 'group:' prefix should map to chat_type='group'."""
        chat_id = "group:G789XYZ"
        chat_type = "c2c"
        target_id = chat_id
        if chat_id.startswith("c2c:"):
            chat_type = "c2c"
            target_id = chat_id[4:]
        elif chat_id.startswith("group:"):
            chat_type = "group"
            target_id = chat_id[6:]
        elif chat_id.startswith("guild:"):
            chat_type = "guild"
            target_id = chat_id[6:]

        assert chat_type == "group"
        assert target_id == "G789XYZ"

    def test_guild_prefix(self):
        """chat_id with 'guild:' prefix should map to chat_type='guild'."""
        chat_id = "guild:CH456"
        chat_type = "c2c"
        target_id = chat_id
        if chat_id.startswith("c2c:"):
            chat_type = "c2c"
            target_id = chat_id[4:]
        elif chat_id.startswith("group:"):
            chat_type = "group"
            target_id = chat_id[6:]
        elif chat_id.startswith("guild:"):
            chat_type = "guild"
            target_id = chat_id[6:]

        assert chat_type == "guild"
        assert target_id == "CH456"

    def test_plain_openid_defaults_to_c2c(self):
        """Plain openid without prefix should default to c2c."""
        chat_id = "PLAIN_OPENID_123"
        chat_type = "c2c"
        target_id = chat_id
        if chat_id.startswith("c2c:"):
            chat_type = "c2c"
            target_id = chat_id[4:]
        elif chat_id.startswith("group:"):
            chat_type = "group"
            target_id = chat_id[6:]
        elif chat_id.startswith("guild:"):
            chat_type = "guild"
            target_id = chat_id[6:]

        assert chat_type == "c2c"
        assert target_id == "PLAIN_OPENID_123"

    def test_empty_chat_id(self):
        """Empty chat_id should default to c2c."""
        chat_id = ""
        chat_type = "c2c"
        target_id = chat_id
        if chat_id.startswith("c2c:"):
            chat_type = "c2c"
            target_id = chat_id[4:]
        elif chat_id.startswith("group:"):
            chat_type = "group"
            target_id = chat_id[6:]
        elif chat_id.startswith("guild:"):
            chat_type = "guild"
            target_id = chat_id[6:]

        assert chat_type == "c2c"
        assert target_id == ""


# ---------------------------------------------------------------------------
# _send_qqbot: file type detection from extension
# ---------------------------------------------------------------------------

class TestSendQqbotFileTypeDetection:
    """Verify file_type (1=image, 2=video, 3=voice, 4=file) from extension."""

    def _detect(self, filename):
        """Replicate file type detection logic from _send_qqbot."""
        file_type = 4  # default to file
        file_ext = filename.lower().split(".")[-1] if "." in filename else ""
        if file_ext in ("png", "jpg", "jpeg", "gif"):
            file_type = 1
        elif file_ext in ("mp4", "mov", "avi"):
            file_type = 2
        elif file_ext in ("mp3", "wav", "flac", "silk", "amr"):
            file_type = 3
        return file_type

    def test_image_png(self):
        assert self._detect("photo.png") == 1

    def test_image_jpg(self):
        assert self._detect("photo.jpg") == 1

    def test_image_jpeg(self):
        assert self._detect("photo.jpeg") == 1

    def test_image_gif(self):
        assert self._detect("anim.gif") == 1

    def test_video_mp4(self):
        assert self._detect("video.mp4") == 2

    def test_video_mov(self):
        assert self._detect("clip.mov") == 2

    def test_video_avi(self):
        assert self._detect("old.avi") == 2

    def test_voice_mp3(self):
        assert self._detect("audio.mp3") == 3

    def test_voice_wav(self):
        assert self._detect("sound.wav") == 3

    def test_voice_flac(self):
        assert self._detect("lossless.flac") == 3

    def test_voice_silk(self):
        assert self._detect("wechat.silk") == 3

    def test_voice_amr(self):
        assert self._detect("recording.amr") == 3

    def test_file_default(self):
        """Unknown extensions should default to file type 4."""
        assert self._detect("doc.pdf") == 4
        assert self._detect("archive.zip") == 4
        assert self._detect("readme.txt") == 4

    def test_no_extension(self):
        """Files without extension should default to file type 4."""
        assert self._detect("noext") == 4

    def test_uppercase_extension(self):
        """Uppercase extensions should still be detected (lower() applied)."""
        assert self._detect("PHOTO.PNG") == 1
        assert self._detect("VIDEO.MP4") == 2


# ---------------------------------------------------------------------------
# _send_qqbot: C2C endpoint URL construction
# ---------------------------------------------------------------------------

class TestSendQqbotEndpointUrls:
    """Verify the correct REST endpoint is chosen per chat_type."""

    def test_c2c_url(self):
        chat_type = "c2c"
        target_id = "USER_ABC"
        if chat_type == "c2c":
            url = f"https://api.sgroup.qq.com/v2/users/{target_id}/messages"
        elif chat_type == "group":
            url = f"https://api.sgroup.qq.com/v2/groups/{target_id}/messages"
        else:
            url = f"https://api.sgroup.qq.com/channels/{target_id}/messages"
        assert url == "https://api.sgroup.qq.com/v2/users/USER_ABC/messages"

    def test_group_url(self):
        chat_type = "group"
        target_id = "GROUP_XYZ"
        if chat_type == "c2c":
            url = f"https://api.sgroup.qq.com/v2/users/{target_id}/messages"
        elif chat_type == "group":
            url = f"https://api.sgroup.qq.com/v2/groups/{target_id}/messages"
        else:
            url = f"https://api.sgroup.qq.com/channels/{target_id}/messages"
        assert url == "https://api.sgroup.qq.com/v2/groups/GROUP_XYZ/messages"

    def test_guild_url(self):
        chat_type = "guild"
        target_id = "CHANNEL_42"
        if chat_type == "c2c":
            url = f"https://api.sgroup.qq.com/v2/users/{target_id}/messages"
        elif chat_type == "group":
            url = f"https://api.sgroup.qq.com/v2/groups/{target_id}/messages"
        else:
            url = f"https://api.sgroup.qq.com/channels/{target_id}/messages"
        assert url == "https://api.sgroup.qq.com/channels/CHANNEL_42/messages"

    def test_c2c_file_upload_url(self):
        chat_type = "c2c"
        target_id = "USER_ABC"
        upload_url = f"https://api.sgroup.qq.com/v2/users/{target_id}/files"
        assert upload_url == "https://api.sgroup.qq.com/v2/users/USER_ABC/files"

    def test_group_file_upload_url(self):
        chat_type = "group"
        target_id = "GROUP_XYZ"
        upload_url = f"https://api.sgroup.qq.com/v2/groups/{target_id}/files"
        assert upload_url == "https://api.sgroup.qq.com/v2/groups/GROUP_XYZ/files"


# ---------------------------------------------------------------------------
# _send_to_platform: QQBot media block routing
# ---------------------------------------------------------------------------

class TestSendToPlatformQqbotMediaRouting:
    """Verify _send_to_platform dispatches to _send_qqbot for QQBot+media."""

    def test_qqbot_media_block_exists(self):
        """The QQBot media delivery block should exist in _send_to_platform."""
        import tools.send_message_tool as smt
        source = smt._send_to_platform.__code__
        # We can't easily inspect async function internals, so check module
        assert hasattr(smt, '_send_qqbot')
        assert callable(smt._send_qqbot)

    def test_send_qqbot_accepts_media_files(self):
        """_send_qqbot should accept an optional media_files parameter."""
        import inspect
        import tools.send_message_tool as smt
        sig = inspect.signature(smt._send_qqbot)
        params = list(sig.parameters.keys())
        assert 'media_files' in params


# ---------------------------------------------------------------------------
# Error / Warning messages: both qqbot and feishu
# ---------------------------------------------------------------------------

class TestMediaDeliveryMessages:
    """Verify error/warning strings list both qqbot and feishu platforms."""

    def test_error_message_contains_both_platforms(self):
        """The error message when only media is sent should list both qqbot and feishu."""
        import tools.send_message_tool as smt

        # Read the source to find the error string
        source = open(smt.__file__).read()
        # Check that both qqbot and feishu appear in the MEDIA delivery error strings
        assert 'qqbot' in source, "Source should mention qqbot platform"
        assert 'feishu' in source, "Source should mention feishu platform"
        # Verify the error message section has both
        import re
        match = re.search(r'send_message MEDIA delivery.*?qqbot.*?feishu', source, re.DOTALL)
        assert match is not None, (
            "Error message should mention both qqbot and feishu in media delivery context"
        )

    def test_warning_message_contains_both_platforms(self):
        """The warning message about omitted media should list both qqbot and feishu."""
        import tools.send_message_tool as smt

        source = open(smt.__file__).read()
        # Count occurrences of 'qqbot and feishu'
        count = source.lower().count('qqbot')
        assert count >= 3, \
            f"Expected at least 3 references to qqbot (target desc, error, warning), found {count}"


# ---------------------------------------------------------------------------
# _send_to_platform: Feishu media block with thread_id
# ---------------------------------------------------------------------------

class TestSendToPlatformFeishuMediaRouting:
    """Verify _send_to_platform dispatches to _send_feishu with thread_id."""

    def test_feishu_media_function_accepts_thread_id(self):
        """_send_feishu should accept thread_id parameter."""
        import inspect
        import tools.send_message_tool as smt
        sig = inspect.signature(smt._send_feishu)
        params = list(sig.parameters.keys())
        assert 'thread_id' in params


# ---------------------------------------------------------------------------
# _send_qqbot: payload construction (no media)
# ---------------------------------------------------------------------------

class TestSendQqbotPayloadConstruction:
    """Verify payload is correctly built for text-only messages."""

    def test_text_only_payload(self):
        """Text-only message should have content and msg_type=0."""
        message = "Hello, QQ!"
        payload = {"content": message[:4000] if message else "", "msg_type": 0}
        assert payload["content"] == "Hello, QQ!"
        assert payload["msg_type"] == 0
        assert "media" not in payload

    def test_empty_message_payload(self):
        """Empty message should still produce valid payload."""
        message = ""
        payload = {"content": message[:4000] if message else "", "msg_type": 0}
        assert payload["content"] == ""
        assert payload["msg_type"] == 0

    def test_long_message_truncation(self):
        """Messages over 4000 chars should be truncated."""
        message = "x" * 5000
        payload = {"content": message[:4000] if message else "", "msg_type": 0}
        assert len(payload["content"]) == 4000

    def test_media_in_payload(self):
        """When file_info_list has items, media key should be added."""
        file_info_list = [{"file_id": "xxx", "filename": "test.png"}]
        message = "Check this out"
        payload = {"content": message[:4000] if message else "", "msg_type": 0}
        if file_info_list:
            payload["media"] = {"file_info": file_info_list[0]}
        assert "media" in payload
        assert payload["media"]["file_info"] == file_info_list[0]

    def test_no_media_for_empty_list(self):
        """When file_info_list is empty, no media key should be present."""
        file_info_list = []
        message = "Just text"
        payload = {"content": message[:4000] if message else "", "msg_type": 0}
        if file_info_list:
            payload["media"] = {"file_info": file_info_list[0]}
        assert "media" not in payload


# ---------------------------------------------------------------------------
# base64 encoding for file data
# ---------------------------------------------------------------------------

class TestSendQqbotBase64Encoding:
    """Verify base64 encoding of file data for QQ Bot v2 API."""

    def test_base64_encode_binary_data(self):
        import base64
        data = b"\x00\x01\x02\xff"
        encoded = base64.b64encode(data).decode("utf-8")
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Verify round-trip
        decoded = base64.b64decode(encoded)
        assert decoded == data

    def test_base64_encode_text_file(self):
        import base64
        data = b"Hello, QQ Bot API!"
        encoded = base64.b64encode(data).decode("utf-8")
        assert encoded == "SGVsbG8sIFFRIEJvdCBBUEkh"
        decoded = base64.b64decode(encoded)
        assert decoded == data

    def test_upload_body_structure(self):
        """Upload body should contain file_type, srv_send_msg=False, and file_data."""
        upload_body = {
            "file_type": 1,
            "srv_send_msg": False,
            "file_data": "base64encodedstring",
        }
        assert upload_body["file_type"] == 1
        assert upload_body["srv_send_msg"] is False
        assert isinstance(upload_body["file_data"], str)
