"""Tests for tools/vision_tools.py — URL validation, type hints, error logging."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Awaitable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import (
    _validate_image_url,
    _handle_vision_analyze,
    _determine_mime_type,
    _image_to_base64_data_url,
    vision_analyze_tool,
    check_vision_requirements,
    get_debug_session_info,
)


# ---------------------------------------------------------------------------
# _validate_image_url — urlparse-based validation
# ---------------------------------------------------------------------------


class TestValidateImageUrl:
    """Tests for URL validation, including urlparse-based netloc check."""

    def test_valid_https_url(self):
        assert _validate_image_url("https://example.com/image.jpg") is True

    def test_valid_http_url(self):
        assert _validate_image_url("http://cdn.example.org/photo.png") is True

    def test_valid_url_without_extension(self):
        """CDN endpoints that redirect to images should still pass."""
        assert _validate_image_url("https://cdn.example.com/abcdef123") is True

    def test_valid_url_with_query_params(self):
        assert _validate_image_url("https://img.example.com/pic?w=200&h=200") is True

    def test_valid_url_with_port(self):
        assert _validate_image_url("http://localhost:8080/image.png") is True

    def test_valid_url_with_path_only(self):
        assert _validate_image_url("https://example.com/") is True

    def test_rejects_empty_string(self):
        assert _validate_image_url("") is False

    def test_rejects_none(self):
        assert _validate_image_url(None) is False

    def test_rejects_non_string(self):
        assert _validate_image_url(12345) is False

    def test_rejects_ftp_scheme(self):
        assert _validate_image_url("ftp://files.example.com/image.jpg") is False

    def test_rejects_file_scheme(self):
        assert _validate_image_url("file:///etc/passwd") is False

    def test_rejects_no_scheme(self):
        assert _validate_image_url("example.com/image.jpg") is False

    def test_rejects_javascript_scheme(self):
        assert _validate_image_url("javascript:alert(1)") is False

    def test_rejects_http_without_netloc(self):
        """http:// alone has no network location — urlparse catches this."""
        assert _validate_image_url("http://") is False

    def test_rejects_https_without_netloc(self):
        assert _validate_image_url("https://") is False

    def test_rejects_http_colon_only(self):
        assert _validate_image_url("http:") is False

    def test_rejects_data_url(self):
        assert _validate_image_url("data:image/png;base64,iVBOR") is False

    def test_rejects_whitespace_only(self):
        assert _validate_image_url("   ") is False

    def test_rejects_boolean(self):
        assert _validate_image_url(True) is False

    def test_rejects_list(self):
        assert _validate_image_url(["https://example.com"]) is False


# ---------------------------------------------------------------------------
# _determine_mime_type
# ---------------------------------------------------------------------------


class TestDetermineMimeType:
    def test_jpg(self):
        assert _determine_mime_type(Path("photo.jpg")) == "image/jpeg"

    def test_jpeg(self):
        assert _determine_mime_type(Path("photo.jpeg")) == "image/jpeg"

    def test_png(self):
        assert _determine_mime_type(Path("screenshot.png")) == "image/png"

    def test_gif(self):
        assert _determine_mime_type(Path("anim.gif")) == "image/gif"

    def test_webp(self):
        assert _determine_mime_type(Path("modern.webp")) == "image/webp"

    def test_unknown_extension_defaults_to_jpeg(self):
        assert _determine_mime_type(Path("file.xyz")) == "image/jpeg"


# ---------------------------------------------------------------------------
# _image_to_base64_data_url
# ---------------------------------------------------------------------------


class TestImageToBase64DataUrl:
    def test_returns_data_url(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        result = _image_to_base64_data_url(img)
        assert result.startswith("data:image/png;base64,")

    def test_custom_mime_type(self, tmp_path):
        img = tmp_path / "test.bin"
        img.write_bytes(b"\x00" * 16)
        result = _image_to_base64_data_url(img, mime_type="image/webp")
        assert result.startswith("data:image/webp;base64,")

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _image_to_base64_data_url(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# _handle_vision_analyze — type signature & behavior
# ---------------------------------------------------------------------------


class TestHandleVisionAnalyze:
    """Verify _handle_vision_analyze returns an Awaitable and builds correct prompt."""

    def test_returns_awaitable(self):
        """The handler must return an Awaitable (coroutine) since it's registered as async."""
        with patch(
            "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
        ) as mock_tool:
            mock_tool.return_value = json.dumps({"result": "ok"})
            result = _handle_vision_analyze(
                {
                    "image_url": "https://example.com/img.png",
                    "question": "What is this?",
                }
            )
            # It should be an Awaitable (coroutine)
            assert isinstance(result, Awaitable)
            # Clean up the coroutine to avoid RuntimeWarning
            result.close()

    def test_prompt_contains_question(self):
        """The full prompt should incorporate the user's question."""
        with patch(
            "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
        ) as mock_tool:
            mock_tool.return_value = json.dumps({"result": "ok"})
            coro = _handle_vision_analyze(
                {
                    "image_url": "https://example.com/img.png",
                    "question": "Describe the cat",
                }
            )
            # Clean up coroutine
            coro.close()
            call_args = mock_tool.call_args
            full_prompt = call_args[0][1]  # second positional arg
            assert "Describe the cat" in full_prompt
            assert "Fully describe and explain" in full_prompt

    def test_uses_auxiliary_vision_model_env(self):
        """AUXILIARY_VISION_MODEL env var should override DEFAULT_VISION_MODEL."""
        with (
            patch(
                "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
            ) as mock_tool,
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "custom/model-v1"}),
        ):
            mock_tool.return_value = json.dumps({"result": "ok"})
            coro = _handle_vision_analyze(
                {"image_url": "https://example.com/img.png", "question": "test"}
            )
            coro.close()
            call_args = mock_tool.call_args
            model = call_args[0][2]  # third positional arg
            assert model == "custom/model-v1"

    def test_falls_back_to_default_model(self):
        """Without AUXILIARY_VISION_MODEL, model should be None (let call_llm resolve default)."""
        with (
            patch(
                "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
            ) as mock_tool,
            patch.dict(os.environ, {}, clear=False),
        ):
            # Ensure AUXILIARY_VISION_MODEL is not set
            os.environ.pop("AUXILIARY_VISION_MODEL", None)
            mock_tool.return_value = json.dumps({"result": "ok"})
            coro = _handle_vision_analyze(
                {"image_url": "https://example.com/img.png", "question": "test"}
            )
            coro.close()
            call_args = mock_tool.call_args
            model = call_args[0][2]
            # With no AUXILIARY_VISION_MODEL set, model should be None
            # (the centralized call_llm router picks the default)
            assert model is None

    def test_empty_args_graceful(self):
        """Missing keys should default to empty strings, not raise."""
        with patch(
            "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
        ) as mock_tool:
            mock_tool.return_value = json.dumps({"result": "ok"})
            result = _handle_vision_analyze({})
            assert isinstance(result, Awaitable)
            result.close()


# ---------------------------------------------------------------------------
# Error logging with exc_info — verify tracebacks are logged
# ---------------------------------------------------------------------------


class TestErrorLoggingExcInfo:
    """Verify that exc_info=True is used in error/warning log calls."""

    @pytest.mark.asyncio
    async def test_download_failure_logs_exc_info(self, tmp_path, caplog):
        """After max retries, the download error should include exc_info."""
        from tools.vision_tools import _download_image

        with patch("tools.vision_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=ConnectionError("network down"))
            mock_client_cls.return_value = mock_client

            dest = tmp_path / "image.jpg"
            with (
                caplog.at_level(logging.ERROR, logger="tools.vision_tools"),
                pytest.raises(ConnectionError),
            ):
                await _download_image(
                    "https://example.com/img.jpg", dest, max_retries=1
                )

            # Should have logged with exc_info (traceback present)
            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert len(error_records) >= 1
            assert error_records[0].exc_info is not None

    @pytest.mark.asyncio
    async def test_analysis_error_logs_exc_info(self, caplog):
        """When vision_analyze_tool encounters an error, it should log with exc_info."""
        with (
            patch("tools.vision_tools._validate_image_url", return_value=True),
            patch(
                "tools.vision_tools._download_image",
                new_callable=AsyncMock,
                side_effect=Exception("download boom"),
            ),
            caplog.at_level(logging.ERROR, logger="tools.vision_tools"),
        ):
            result = await vision_analyze_tool(
                "https://example.com/img.jpg", "describe this", "test/model"
            )
            result_data = json.loads(result)
            # Error response uses "success": False, not an "error" key
            assert result_data["success"] is False

            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert any(r.exc_info and r.exc_info[0] is not None for r in error_records)

    @pytest.mark.asyncio
    async def test_cleanup_error_logs_exc_info(self, tmp_path, caplog):
        """Temp file cleanup failure should log warning with exc_info."""
        # Create a real temp file that will be "downloaded"
        temp_dir = tmp_path / "temp_vision_images"
        temp_dir.mkdir()

        async def fake_download(url, dest, max_retries=3):
            """Simulate download by writing file to the expected destination."""
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\xff\xd8\xff" + b"\x00" * 16)
            return dest

        with (
            patch("tools.vision_tools._validate_image_url", return_value=True),
            patch("tools.vision_tools._download_image", side_effect=fake_download),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/jpeg;base64,abc",
            ),
            caplog.at_level(logging.WARNING, logger="tools.vision_tools"),
        ):
            # Mock the async_call_llm function to return a mock response
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "A test image description"
            mock_response.choices = [mock_choice]

            with (
                patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response),
            ):
                # Make unlink fail to trigger cleanup warning
                original_unlink = Path.unlink

                def failing_unlink(self, *args, **kwargs):
                    raise PermissionError("no permission")

                with patch.object(Path, "unlink", failing_unlink):
                    result = await vision_analyze_tool(
                        "https://example.com/tempimg.jpg", "describe", "test/model"
                    )

            warning_records = [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING
                and "temporary file" in r.getMessage().lower()
            ]
            assert len(warning_records) >= 1
            assert warning_records[0].exc_info is not None


# ---------------------------------------------------------------------------
# check_vision_requirements & get_debug_session_info
# ---------------------------------------------------------------------------


class TestVisionRequirements:
    def test_check_requirements_returns_bool(self):
        result = check_vision_requirements()
        assert isinstance(result, bool)

    def test_debug_session_info_returns_dict(self):
        info = get_debug_session_info()
        assert isinstance(info, dict)
        # DebugSession.get_session_info() returns these keys
        assert "enabled" in info
        assert "session_id" in info
        assert "total_calls" in info


# ---------------------------------------------------------------------------
# Integration: registry entry
# ---------------------------------------------------------------------------


class TestVisionRegistration:
    def test_vision_analyze_registered(self):
        from tools.registry import registry

        entry = registry._tools.get("vision_analyze")
        assert entry is not None
        assert entry.toolset == "vision"
        assert entry.is_async is True

    def test_schema_has_required_fields(self):
        from tools.registry import registry

        entry = registry._tools.get("vision_analyze")
        schema = entry.schema
        assert schema["name"] == "vision_analyze"
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        assert "image_url" in props
        assert "question" in props

    def test_handler_is_callable(self):
        from tools.registry import registry

        entry = registry._tools.get("vision_analyze")
        assert callable(entry.handler)
