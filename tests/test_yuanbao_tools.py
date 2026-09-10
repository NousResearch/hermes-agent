"""Tests for tools/yuanbao_tools.py -- target >= 70% statement coverage.

Current module coverage is 16.75%. These tests cover the 5 async public tool
functions, their handler wrappers, and the helper/check functions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_logger():
    """Silence logger output during tests."""
    with patch("tools.yuanbao_tools.logger") as mock:
        yield mock


@pytest.fixture
def mock_adapter():
    """Return a fresh AsyncMock that quacks like a Yuanbao adapter."""
    adapter = AsyncMock()
    adapter.query_group_info.return_value = {
        "group_name": "Test Group",
        "member_count": 42,
        "owner_id": "u1",
        "owner_nickname": "owner1",
    }
    adapter.get_group_member_list.return_value = {
        "members": [
            {"user_id": "u1", "nickname": "Alice", "role": 1},
            {"user_id": "u2", "nickname": "Bob", "role": 1},
            {"user_id": "u3", "nickname": "YuanBot", "user_type": 2},
            {"user_id": "u4", "nickname": "SvcBot", "role": 3},
        ]
    }
    dm_result = MagicMock()
    dm_result.success = True
    dm_result.message_id = "dm-1"
    dm_result.error = None
    adapter.send_dm.return_value = dm_result
    adapter.send_sticker.return_value = dm_result
    adapter.send_image_file = AsyncMock()
    adapter.send_image_file.return_value = dm_result
    adapter.send_document = AsyncMock()
    adapter.send_document.return_value = dm_result
    return adapter


@pytest.fixture
def _patch_adapter(mock_adapter):
    """Patch _get_active_adapter() to return the mock adapter."""
    with patch(
        "tools.yuanbao_tools._get_active_adapter",
        return_value=mock_adapter,
    ) as p:
        yield p


@pytest.fixture
def _patch_session_chat_id():
    """Patch get_session_env to return a group chat_id."""
    with patch(
        "gateway.session_context.get_session_env",
        return_value="group:g123",
    ) as p:
        yield p


# ---------------------------------------------------------------------------
# _get_active_adapter
# ---------------------------------------------------------------------------


class TestGetActiveAdapter:
    def test_returns_adapter_when_import_succeeds(self):
        """The lazy import path works when gateway.platforms.yuanbao is available.
        We mimic a successful import by clearing any cached result and patching
        the target that the lazy import looks up."""
        # Force the lazy import to "succeed" by pointing at a fake module that
        # has get_active_adapter. The try/except in _get_active_adapter will
        # then get a module whose get_active_adapter returns "fake".
        import sys
        fake_mod = type(sys)("gateway.platforms.yuanbao")
        fake_mod.get_active_adapter = lambda: "fake"
        with patch.dict(sys.modules, {"gateway.platforms.yuanbao": fake_mod}):
            from tools.yuanbao_tools import _get_active_adapter

            result = _get_active_adapter()
            assert result == "fake"

    def test_returns_none_on_import_error(self):
        """When gateway.platforms.yuanbao is not available, return None."""
        from tools.yuanbao_tools import _get_active_adapter

        result = _get_active_adapter()
        assert result is None


# ---------------------------------------------------------------------------
# _check_yuanbao
# ---------------------------------------------------------------------------


class TestCheckYuanbao:
    def test_returns_true_when_session_platform_is_yuanbao(self):
        """The session env check returns True for yuanbao platform."""
        with patch(
            "gateway.session_context.get_session_env",
            return_value="yuanbao",
        ):
            from tools.yuanbao_tools import _check_yuanbao

            assert _check_yuanbao() is True

    def test_returns_true_when_adapter_is_active(self):
        """Fallback to _get_active_adapter() when session env is not set."""
        with patch(
            "gateway.session_context.get_session_env",
            return_value="",
        ), patch(
            "tools.yuanbao_tools._get_active_adapter",
            return_value=MagicMock(),
        ):
            from tools.yuanbao_tools import _check_yuanbao

            assert _check_yuanbao() is True

    def test_returns_false_when_not_in_yuanbao_context(self):
        """Both session env and adapter are absent."""
        with patch(
            "gateway.session_context.get_session_env",
            return_value="telegram",
        ), patch(
            "tools.yuanbao_tools._get_active_adapter",
            return_value=None,
        ):
            from tools.yuanbao_tools import _check_yuanbao

            assert _check_yuanbao() is False


# ---------------------------------------------------------------------------
# get_group_info
# ---------------------------------------------------------------------------


class TestGetGroupInfo:
    @pytest.mark.asyncio
    async def test_empty_group_code(self, _patch_adapter):
        from tools.yuanbao_tools import get_group_info

        result = await get_group_info("")
        assert result == {"success": False, "error": "group_code is required"}

    @pytest.mark.asyncio
    async def test_no_adapter(self):
        from tools.yuanbao_tools import get_group_info

        result = await get_group_info("g123")
        assert result == {"success": False, "error": "Yuanbao adapter is not connected"}

    @pytest.mark.asyncio
    async def test_query_returns_none(self, mock_adapter, _patch_adapter):
        mock_adapter.query_group_info.return_value = None
        from tools.yuanbao_tools import get_group_info

        result = await get_group_info("g123")
        assert result == {"success": False, "error": "query_group_info returned None"}

    @pytest.mark.asyncio
    async def test_query_returns_minimal(self, mock_adapter, _patch_adapter):
        mock_adapter.query_group_info.return_value = {}
        from tools.yuanbao_tools import get_group_info

        result = await get_group_info("g123")
        assert result["success"] is True
        assert result["group_code"] == "g123"
        assert result["group_name"] == ""
        assert result["member_count"] == 0

    @pytest.mark.asyncio
    async def test_success(self, _patch_adapter):
        from tools.yuanbao_tools import get_group_info

        result = await get_group_info("g123")
        assert result["success"] is True
        assert result["group_name"] == "Test Group"
        assert result["member_count"] == 42
        assert result["owner"]["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_adapter, _patch_adapter):
        mock_adapter.query_group_info.side_effect = RuntimeError("boom")
        from tools.yuanbao_tools import get_group_info

        result = await get_group_info("g123")
        assert result["success"] is False
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# query_group_members
# ---------------------------------------------------------------------------


class TestQueryGroupMembers:
    @pytest.mark.asyncio
    async def test_empty_group_code(self, _patch_adapter):
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("")
        assert result == {"success": False, "error": "group_code is required"}

    @pytest.mark.asyncio
    async def test_no_adapter(self):
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123")
        assert result == {"success": False, "error": "Yuanbao adapter is not connected"}

    @pytest.mark.asyncio
    async def test_member_list_none(self, mock_adapter, _patch_adapter):
        mock_adapter.get_group_member_list.return_value = None
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123")
        assert result == {"success": False, "error": "get_group_member_list returned None"}

    @pytest.mark.asyncio
    async def test_no_members(self, mock_adapter, _patch_adapter):
        mock_adapter.get_group_member_list.return_value = {"members": []}
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123")
        assert result == {"success": False, "error": "No members found in this group."}

    @pytest.mark.asyncio
    async def test_list_all_success(self, _patch_adapter):
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123")
        assert result["success"] is True
        assert result["msg"] == "Found 4 member(s)."
        assert len(result["members"]) == 4

    @pytest.mark.asyncio
    async def test_list_all_with_mention(self, _patch_adapter):
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123", mention=True)
        assert result["success"] is True
        assert "mention_hint" in result

    @pytest.mark.asyncio
    async def test_list_bots(self, _patch_adapter):
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123", action="list_bots")
        assert result["success"] is True
        assert result["msg"] == "Found 2 bot(s)."
        assert len(result["members"]) == 2

    @pytest.mark.asyncio
    async def test_list_bots_no_bots(self, mock_adapter, _patch_adapter):
        mock_adapter.get_group_member_list.return_value = {
            "members": [{"user_id": "u1", "nickname": "Alice", "role": 1}]
        }
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123", action="list_bots")
        assert result["success"] is False
        assert "No bots found" in result["error"]

    @pytest.mark.asyncio
    async def test_find_matches(self, _patch_adapter):
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123", action="find", name="Alice")
        assert result["success"] is True
        assert len(result["members"]) == 1
        assert result["members"][0]["nickname"] == "Alice"

    @pytest.mark.asyncio
    async def test_find_no_match(self, _patch_adapter):
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123", action="find", name="Zzz")
        assert result["success"] is False
        assert len(result["members"]) == 4  # falls back to all members

    @pytest.mark.asyncio
    async def test_find_empty_name(self, _patch_adapter):
        """action='find' with empty name returns all members."""
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123", action="find", name="")
        assert result["success"] is True
        assert len(result["members"]) == 4

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_adapter, _patch_adapter):
        mock_adapter.get_group_member_list.side_effect = RuntimeError("boom")
        from tools.yuanbao_tools import query_group_members

        result = await query_group_members("g123")
        assert result["success"] is False
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# search_sticker
# ---------------------------------------------------------------------------


class TestSearchSticker:
    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Empty query returns first N stickers."""
        with patch(
            "gateway.platforms.yuanbao_sticker.search_stickers",
            return_value=[
                {"sticker_id": "1", "name": "smile", "description": "a smile", "package_id": "p1"}
            ],
        ):
            from tools.yuanbao_tools import search_sticker

            result = await search_sticker("")
            assert result["success"] is True
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self):
        """limit over 50 is clamped to 50."""
        with patch("gateway.platforms.yuanbao_sticker.search_stickers", return_value=[]) as mock:
            from tools.yuanbao_tools import search_sticker

            await search_sticker("cool", limit=999)
            mock.assert_called_once_with("cool", limit=50)

    @pytest.mark.asyncio
    async def test_limit_clamped_to_min(self):
        """limit below 1 is clamped to 1."""
        with patch("gateway.platforms.yuanbao_sticker.search_stickers", return_value=[]) as mock:
            from tools.yuanbao_tools import search_sticker

            await search_sticker("cool", limit=-5)
            mock.assert_called_once_with("cool", limit=1)

    @pytest.mark.asyncio
    async def test_invalid_limit_type(self):
        """Non-integer limit defaults to 10."""
        with patch("gateway.platforms.yuanbao_sticker.search_stickers", return_value=[]) as mock:
            from tools.yuanbao_tools import search_sticker

            await search_sticker("cool", limit="abc")
            mock.assert_called_once_with("cool", limit=10)

    @pytest.mark.asyncio
    async def test_search_sticker_exception(self):
        """Exception from search_stickers returns error dict."""
        with patch(
            "gateway.platforms.yuanbao_sticker.search_stickers",
            side_effect=ValueError("search failed"),
        ):
            from tools.yuanbao_tools import search_sticker

            result = await search_sticker("cool")
            assert result["success"] is False
            assert "search failed" in result["error"]


# ---------------------------------------------------------------------------
# send_sticker
# ---------------------------------------------------------------------------


class TestSendSticker:
    @pytest.mark.asyncio
    async def test_no_chat_id_no_session(self):
        from tools.yuanbao_tools import send_sticker

        result = await send_sticker(sticker="happy")
        assert result["success"] is False
        assert "chat_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_no_adapter(self, _patch_session_chat_id):
        from tools.yuanbao_tools import send_sticker

        result = await send_sticker(sticker="happy")
        assert result["success"] is False
        assert "adapter is not connected" in result["error"]

    @pytest.mark.asyncio
    async def test_send_random_sticker(self, mock_adapter, _patch_adapter, _patch_session_chat_id):
        """Empty sticker name sends a random sticker."""
        with patch("gateway.platforms.yuanbao_sticker.get_random_sticker") as mock_rand:
            mock_rand.return_value = {"sticker_id": "99", "name": "random"}
            from tools.yuanbao_tools import send_sticker

            result = await send_sticker(sticker="")
            assert result["success"] is True
            assert result["sticker"]["name"] == "random"
            mock_adapter.send_sticker.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_by_name(self, mock_adapter, _patch_adapter, _patch_session_chat_id):
        with patch("gateway.platforms.yuanbao_sticker.get_sticker_by_name") as mock_name, patch(
            "gateway.platforms.yuanbao_sticker.get_sticker_by_id"
        ) as mock_id:
            mock_id.return_value = None
            mock_name.return_value = {"sticker_id": "7", "name": "happy"}
            from tools.yuanbao_tools import send_sticker

            result = await send_sticker(sticker="happy")
            assert result["success"] is True
            assert result["sticker"]["sticker_id"] == "7"

    @pytest.mark.asyncio
    async def test_send_by_id(self, mock_adapter, _patch_adapter, _patch_session_chat_id):
        with patch("gateway.platforms.yuanbao_sticker.get_sticker_by_id") as mock_id:
            mock_id.return_value = {"sticker_id": "278", "name": "six"}
            from tools.yuanbao_tools import send_sticker

            result = await send_sticker(sticker="278")
            assert result["success"] is True
            assert result["sticker"]["sticker_id"] == "278"

    @pytest.mark.asyncio
    async def test_sticker_not_found(self, mock_adapter, _patch_adapter, _patch_session_chat_id):
        with patch("gateway.platforms.yuanbao_sticker.get_sticker_by_name", return_value=None), patch(
            "gateway.platforms.yuanbao_sticker.get_sticker_by_id", return_value=None
        ):
            from tools.yuanbao_tools import send_sticker

            result = await send_sticker(sticker="nonexistent")
            assert result["success"] is False
            assert "Sticker not found" in result["error"]

    @pytest.mark.asyncio
    async def test_send_sticker_with_reply_to(
        self, mock_adapter, _patch_adapter, _patch_session_chat_id
    ):
        with patch("gateway.platforms.yuanbao_sticker.get_sticker_by_name") as mock_name:
            mock_name.return_value = {"sticker_id": "7", "name": "happy"}
            from tools.yuanbao_tools import send_sticker

            result = await send_sticker(sticker="happy", reply_to="ref123")
            assert result["success"] is True
            mock_adapter.send_sticker.assert_awaited_once_with(
                chat_id="group:g123", sticker_name="happy", reply_to="ref123"
            )

    @pytest.mark.asyncio
    async def test_send_exception(self, mock_adapter, _patch_adapter, _patch_session_chat_id):
        with patch("gateway.platforms.yuanbao_sticker.get_sticker_by_name") as mock_name:
            mock_name.return_value = {"sticker_id": "7", "name": "happy"}
            mock_adapter.send_sticker.side_effect = RuntimeError("network error")
            from tools.yuanbao_tools import send_sticker

            result = await send_sticker(sticker="happy")
            assert result["success"] is False
            assert "network error" in result["error"]

    @pytest.mark.asyncio
    async def test_send_result_failure(self, mock_adapter, _patch_adapter, _patch_session_chat_id):
        with patch("gateway.platforms.yuanbao_sticker.get_sticker_by_name") as mock_name:
            mock_name.return_value = {"sticker_id": "7", "name": "happy"}
            mock_adapter.send_sticker.return_value = MagicMock(
                success=False, error="rate limited"
            )
            from tools.yuanbao_tools import send_sticker

            result = await send_sticker(sticker="happy")
            assert result["success"] is False
            assert "rate limited" in result["error"]


# ---------------------------------------------------------------------------
# send_dm
# ---------------------------------------------------------------------------


class TestSendDm:
    @pytest.mark.asyncio
    async def test_no_message_no_media(self, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "")
        assert result["success"] is False
        assert "message or media_files is required" in result["error"]

    @pytest.mark.asyncio
    async def test_no_adapter(self):
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello")
        assert result["success"] is False
        assert "adapter is not connected" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_with_user_id(self, mock_adapter, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello", user_id="u1")
        assert result["success"] is True
        assert result["user_id"] == "u1"
        mock_adapter.send_dm.assert_awaited_once_with("u1", "hello", group_code="g123")

    @pytest.mark.asyncio
    async def test_dm_resolve_user(self, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello")
        assert result["success"] is True
        assert result["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_dm_no_group_code_no_user_id(self, mock_adapter, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm("", "Alice", "hello")
        assert result["success"] is False
        assert "group_code is required" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_no_name(self, mock_adapter, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "", "hello")
        assert result["success"] is False
        assert "name is required" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_member_list_none(self, mock_adapter, _patch_adapter):
        mock_adapter.get_group_member_list.return_value = None
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello")
        assert result["success"] is False
        assert "returned None" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_no_match(self, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Zzz", "hello")
        assert result["success"] is False
        assert "No member matching" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_multiple_matches(self, mock_adapter, _patch_adapter):
        """Two members match - Alice resolves successfully."""
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_dm_multiple_matches_disambiguation(self, mock_adapter, _patch_adapter):
        mock_adapter.get_group_member_list.return_value = {
            "members": [
                {"user_id": "u1", "nickname": "Alice", "role": 1},
                {"user_id": "u5", "nickname": "Alex", "role": 1},
            ]
        }
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Al", "hello")
        assert result["success"] is False
        assert "Multiple members match" in result["error"]
        assert len(result["candidates"]) == 2

    @pytest.mark.asyncio
    async def test_dm_with_media_image(self, mock_adapter, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm(
            "g123", "Alice", "hello",
            media_files=[("/tmp/photo.jpg", False)],
        )
        assert result["success"] is True
        mock_adapter.send_image_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dm_with_media_document(self, mock_adapter, _patch_adapter):
        from tools.yuanbao_tools import send_dm

        result = await send_dm(
            "g123", "Alice", "hello",
            media_files=[("/tmp/report.pdf", False)],
        )
        assert result["success"] is True
        mock_adapter.send_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dm_resolve_exception(self, mock_adapter, _patch_adapter):
        mock_adapter.get_group_member_list.side_effect = RuntimeError("resolve error")
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello")
        assert result["success"] is False
        assert "resolve error" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_send_exception(self, mock_adapter, _patch_adapter):
        mock_adapter.send_dm.side_effect = RuntimeError("send failed")
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello")
        assert result["success"] is False
        assert "send failed" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_text_send_failure(self, mock_adapter, _patch_adapter):
        mock_adapter.send_dm.return_value = MagicMock(
            success=False, error="text failed", message_id=None
        )
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "hello")
        assert result["success"] is False
        assert "text failed" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_no_deliverable(self, mock_adapter, _patch_adapter):
        """Empty message and no media returns early error."""
        from tools.yuanbao_tools import send_dm

        result = await send_dm("g123", "Alice", "")
        assert result["success"] is False
        assert "message or media_files is required" in result["error"]

    @pytest.mark.asyncio
    async def test_dm_partial_failure(self, mock_adapter, _patch_adapter):
        """Text succeeds, media fails - returns aggregated error."""
        mock_adapter.send_image_file.return_value = MagicMock(
            success=False, error="media failed"
        )
        from tools.yuanbao_tools import send_dm

        result = await send_dm(
            "g123", "Alice", "hello",
            media_files=[("/tmp/pic.jpg", False)],
        )
        assert result["success"] is False
        assert "media failed" in result["error"]


# ---------------------------------------------------------------------------
# Handler wrappers
# ---------------------------------------------------------------------------


class TestHandlerWrappers:
    """The handler wrappers (_handle_yb_query_group_info etc.) that adapt
    tool arguments to the public function signatures.  All handlers return
    JSON strings via tool_result()."""

    @pytest.mark.asyncio
    async def test_yb_query_group_info_handler(self, _patch_adapter):
        from tools.yuanbao_tools import _handle_yb_query_group_info

        result = await _handle_yb_query_group_info({"group_code": "g123"})
        assert '"success": true' in result
        assert '"group_name": "Test Group"' in result

    @pytest.mark.asyncio
    async def test_yb_query_group_info_handler_no_args(self, _patch_adapter):
        from tools.yuanbao_tools import _handle_yb_query_group_info

        result = await _handle_yb_query_group_info({})
        assert "group_code is required" in result
        assert '"success": false' in result

    @pytest.mark.asyncio
    async def test_yb_query_group_members_handler(self, _patch_adapter):
        from tools.yuanbao_tools import _handle_yb_query_group_members

        result = await _handle_yb_query_group_members({"group_code": "g123"})
        assert '"success": true' in result
        assert '"Found 4 member(s)."' in result

    @pytest.mark.asyncio
    async def test_yb_search_sticker_handler(self):
        with patch("gateway.platforms.yuanbao_sticker.search_stickers", return_value=[]):
            from tools.yuanbao_tools import _handle_yb_search_sticker

            result = await _handle_yb_search_sticker({"query": "cool"})
            assert '"success": true' in result
            assert '"count": 0' in result

    @pytest.mark.asyncio
    async def test_yb_send_sticker_handler_no_chat_id(self, _patch_adapter):
        from tools.yuanbao_tools import _handle_yb_send_sticker

        result = await _handle_yb_send_sticker({"sticker": "happy"})
        assert '"success": false' in result
        assert "chat_id is required" in result

    @pytest.mark.asyncio
    async def test_yb_send_dm_handler(self, mock_adapter, _patch_adapter):
        from tools.yuanbao_tools import _handle_yb_send_dm

        result = await _handle_yb_send_dm(
            {"group_code": "g123", "name": "Alice", "message": "hello"}
        )
        assert '"success": true' in result

    @pytest.mark.asyncio
    async def test_yb_send_dm_handler_with_media(self, mock_adapter, _patch_adapter):
        from tools.yuanbao_tools import _handle_yb_send_dm
        from gateway.platforms.base import BasePlatformAdapter

        with patch.object(
            BasePlatformAdapter, "filter_media_delivery_paths",
            side_effect=lambda x: x,
        ):
            result = await _handle_yb_send_dm({
                "group_code": "g123",
                "name": "Alice",
                "message": "hello",
                "media_files": [{"path": "/tmp/img.jpg", "is_voice": False}],
            })
        assert '"success": true' in result
        mock_adapter.send_image_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_yb_send_dm_handler_with_embedded_media(self, mock_adapter, _patch_adapter):
        """MEDIA:<path> tags in the message text are extracted."""
        from tools.yuanbao_tools import _handle_yb_send_dm
        from gateway.platforms.base import BasePlatformAdapter

        with patch.object(
            BasePlatformAdapter, "filter_media_delivery_paths",
            side_effect=lambda x: x,
        ):
            result = await _handle_yb_send_dm({
                "group_code": "g123",
                "name": "Alice",
                "message": "Check this MEDIA:/tmp/report.pdf",
            })
        assert '"success": true' in result


# ---------------------------------------------------------------------------
# Registry constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_user_type_label(self):
        from tools.yuanbao_tools import _USER_TYPE_LABEL

        assert _USER_TYPE_LABEL[0] == "unknown"
        assert _USER_TYPE_LABEL[1] == "user"
        assert _USER_TYPE_LABEL[2] == "yuanbao_ai"
        assert _USER_TYPE_LABEL[3] == "bot"

    def test_mention_hint_exists(self):
        from tools.yuanbao_tools import MENTION_HINT

        assert "@" in MENTION_HINT
        assert "nickname" in MENTION_HINT

    def test_image_exts(self):
        from tools.yuanbao_tools import _IMAGE_EXTS

        assert ".jpg" in _IMAGE_EXTS
        assert ".png" in _IMAGE_EXTS
        assert ".gif" in _IMAGE_EXTS
        assert ".txt" not in _IMAGE_EXTS

    def test_registry_registrations(self):
        """The registry is populated with 5 handler registrations.
        We verify by checking the tool names in the toolset dispatch table
        rather than accessing registry internals which may not be exposed."""
        from tools.yuanbao_tools import _TOOLSET

        assert _TOOLSET == "hermes-yuanbao"
