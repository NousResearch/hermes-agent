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
    _resize_image_for_vision,
    _is_image_size_error,
    _MAX_BASE64_BYTES,
    _RESIZE_TARGET_BYTES,
    vision_analyze_tool,
    check_vision_requirements,
)


# ---------------------------------------------------------------------------
# _validate_image_url — urlparse-based validation
# ---------------------------------------------------------------------------


class TestValidateImageUrl:
    """Tests for URL validation, including urlparse-based netloc check."""

    def test_valid_https_url(self):
        with patch("tools.url_safety.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert _validate_image_url("https://example.com/image.jpg") is True

    def test_valid_http_url(self):
        with patch("tools.url_safety.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert _validate_image_url("http://cdn.example.org/photo.png") is True

    def test_valid_url_without_extension(self):
        """CDN endpoints that redirect to images should still pass."""
        with patch("tools.url_safety.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert _validate_image_url("https://cdn.example.com/abcdef123") is True

    def test_valid_url_with_query_params(self):
        with patch("tools.url_safety.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert _validate_image_url("https://img.example.com/pic?w=200&h=200") is True

    def test_localhost_url_blocked_by_ssrf(self):
        """localhost URLs are now blocked by SSRF protection."""
        assert _validate_image_url("http://localhost:8080/image.png") is False

    def test_valid_url_with_port(self):
        with patch("tools.url_safety.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
            assert _validate_image_url("http://example.com:8080/image.png") is True

    def test_valid_url_with_path_only(self):
        with patch("tools.url_safety.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]):
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


class TestVisionSafetyGuards:
    @pytest.mark.asyncio
    async def test_local_non_image_file_rejected_before_llm_call(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP-SECRET=1\n", encoding="utf-8")

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = json.loads(await vision_analyze_tool(str(secret), "extract text"))

        assert result["success"] is False
        assert "Only real image files are supported" in result["error"]
        mock_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_blocked_remote_url_short_circuits_before_download(self):
        blocked = {
            "host": "blocked.test",
            "rule": "blocked.test",
            "source": "config",
            "message": "Blocked by website policy",
        }

        with (
            patch("tools.vision_tools.check_website_access", return_value=blocked),
            patch("tools.vision_tools._validate_image_url", return_value=True),
            patch("tools.vision_tools._download_image", new_callable=AsyncMock) as mock_download,
        ):
            result = json.loads(await vision_analyze_tool("https://blocked.test/cat.png", "describe"))

        assert result["success"] is False
        assert "Blocked by website policy" in result["error"]
        mock_download.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_download_blocks_redirected_final_url(self, tmp_path):
        from tools.vision_tools import _download_image

        def fake_check(url):
            if url == "https://allowed.test/cat.png":
                return None
            if url == "https://blocked.test/final.png":
                return {
                    "host": "blocked.test",
                    "rule": "blocked.test",
                    "source": "config",
                    "message": "Blocked by website policy",
                }
            raise AssertionError(f"unexpected URL checked: {url}")

        class FakeResponse:
            url = "https://blocked.test/final.png"
            headers = {"content-length": "24"}
            content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

            def raise_for_status(self):
                return None

        with (
            patch("tools.vision_tools.check_website_access", side_effect=fake_check),
            patch("tools.vision_tools.httpx.AsyncClient") as mock_client_cls,
            pytest.raises(PermissionError, match="Blocked by website policy"),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=FakeResponse())
            mock_client_cls.return_value = mock_client

            await _download_image("https://allowed.test/cat.png", tmp_path / "cat.png", max_retries=1)

        assert not (tmp_path / "cat.png").exists()


# ---------------------------------------------------------------------------
# check_vision_requirements
# ---------------------------------------------------------------------------


class TestVisionRequirements:
    def test_check_requirements_returns_bool(self):
        result = check_vision_requirements()
        assert isinstance(result, bool)

    def test_check_requirements_accepts_codex_auth(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            '{"active_provider":"openai-codex","providers":{"openai-codex":{"tokens":{"access_token":"codex-access-token","refresh_token":"codex-refresh-token"}}}}'
        )
        # config.yaml must reference the codex provider so vision auto-detect
        # falls back to the active provider via _read_main_provider().
        (tmp_path / "config.yaml").write_text(
            'model:\n  default: gpt-4o\n  provider: openai-codex\n'
        )
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        assert check_vision_requirements() is True


# ---------------------------------------------------------------------------
# Integration: registry entry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tilde expansion in local file paths
# ---------------------------------------------------------------------------


class TestTildeExpansion:
    """Verify that ~/path style paths are expanded correctly."""

    @pytest.mark.asyncio
    async def test_tilde_path_expanded_to_local_file(self, tmp_path, monkeypatch):
        """vision_analyze_tool should expand ~ in file paths."""
        # Create a fake image file under a fake home directory
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        img = fake_home / "test_image.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        monkeypatch.setenv("HOME", str(fake_home))

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "A test image"
        mock_response.choices = [mock_choice]

        with (
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/png;base64,abc",
            ),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            result = await vision_analyze_tool(
                "~/test_image.png", "describe this", "test/model"
            )
            data = json.loads(result)
            assert data["success"] is True
            assert data["analysis"] == "A test image"

    @pytest.mark.asyncio
    async def test_tilde_path_nonexistent_file_gives_error(self, tmp_path, monkeypatch):
        """A tilde path that doesn't resolve to a real file should fail gracefully."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        result = await vision_analyze_tool(
            "~/nonexistent.png", "describe this", "test/model"
        )
        data = json.loads(result)
        assert data["success"] is False


# ---------------------------------------------------------------------------
# file:// URI support
# ---------------------------------------------------------------------------


class TestFileUriSupport:
    """Verify that file:// URIs resolve as local file paths."""

    @pytest.mark.asyncio
    async def test_file_uri_resolved_as_local_path(self, tmp_path):
        """file:///absolute/path should be treated as a local file."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "A test image"
        mock_response.choices = [mock_choice]

        with (
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/png;base64,abc",
            ),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            result = await vision_analyze_tool(
                f"file://{img}", "describe this", "test/model"
            )
            data = json.loads(result)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_file_uri_nonexistent_gives_error(self, tmp_path):
        """file:// pointing to a missing file should fail gracefully."""
        result = await vision_analyze_tool(
            f"file://{tmp_path}/nonexistent.png", "describe this", "test/model"
        )
        data = json.loads(result)
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Base64 size pre-flight check
# ---------------------------------------------------------------------------


class TestBase64SizeLimit:
    """Verify that oversized images are rejected before hitting the API."""

    @pytest.mark.asyncio
    async def test_oversized_image_rejected_before_api_call(self, tmp_path):
        """Images exceeding the 20 MB hard limit should fail with a clear error."""
        img = tmp_path / "huge.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (4 * 1024 * 1024))

        # Patch the hard limit to a small value so the test runs fast.
        with patch("tools.vision_tools._MAX_BASE64_BYTES", 1000), \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = json.loads(await vision_analyze_tool(str(img), "describe this"))

        assert result["success"] is False
        assert "too large" in result["error"].lower()
        mock_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_small_image_not_rejected(self, tmp_path):
        """Images well under the limit should pass the size check."""
        img = tmp_path / "small.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Small image"
        mock_response.choices = [mock_choice]

        with (
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            result = json.loads(await vision_analyze_tool(str(img), "describe this", "test/model"))

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Error classification for 400 responses
# ---------------------------------------------------------------------------


class TestErrorClassification:
    """Verify that API 400 errors produce actionable guidance."""

    @pytest.mark.asyncio
    async def test_invalid_request_error_gives_image_guidance(self, tmp_path):
        """An invalid_request_error from the API should mention image size/format."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        api_error = Exception(
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': 'Invalid request data'}}"
        )

        with (
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/png;base64,abc",
            ),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                side_effect=api_error,
            ),
        ):
            result = json.loads(await vision_analyze_tool(str(img), "describe", "test/model"))

        assert result["success"] is False
        assert "rejected the image" in result["analysis"].lower()
        assert "smaller" in result["analysis"].lower()


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


# ---------------------------------------------------------------------------
# _resize_image_for_vision — auto-resize oversized images
# ---------------------------------------------------------------------------


class TestResizeImageForVision:
    """Tests for the auto-resize function."""

    def test_small_image_returned_as_is(self, tmp_path):
        """Images under the limit should be returned unchanged."""
        # Create a small 10x10 red PNG
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        path = tmp_path / "small.png"
        img.save(path, "PNG")

        result = _resize_image_for_vision(path, mime_type="image/png")
        assert result.startswith("data:image/png;base64,")
        assert len(result) < _MAX_BASE64_BYTES

    def test_large_image_is_resized(self, tmp_path):
        """Images over the default target should be auto-resized to fit."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        # Create a large image that will exceed 5 MB in base64
        # A 4000x4000 uncompressed PNG will be large
        img = Image.new("RGB", (4000, 4000), (128, 200, 50))
        path = tmp_path / "large.png"
        img.save(path, "PNG")

        result = _resize_image_for_vision(path, mime_type="image/png")
        assert result.startswith("data:image/png;base64,")
        # Default target is _RESIZE_TARGET_BYTES (5 MB), not _MAX_BASE64_BYTES (20 MB)
        assert len(result) <= _RESIZE_TARGET_BYTES

    def test_custom_max_bytes(self, tmp_path):
        """The max_base64_bytes parameter should be respected."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (200, 200), (0, 128, 255))
        path = tmp_path / "medium.png"
        img.save(path, "PNG")

        # Set a very low limit to force resizing
        result = _resize_image_for_vision(path, max_base64_bytes=500)
        # Should still return a valid data URL
        assert result.startswith("data:image/")

    def test_jpeg_output_for_non_png(self, tmp_path):
        """Non-PNG images should be resized as JPEG."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (2000, 2000), (255, 128, 0))
        path = tmp_path / "photo.jpg"
        img.save(path, "JPEG", quality=95)

        result = _resize_image_for_vision(path, mime_type="image/jpeg",
                                           max_base64_bytes=50_000)
        assert result.startswith("data:image/jpeg;base64,")

    def test_constants_sane(self):
        """Hard limit should be larger than resize target."""
        assert _MAX_BASE64_BYTES == 20 * 1024 * 1024
        assert _RESIZE_TARGET_BYTES == 5 * 1024 * 1024
        assert _MAX_BASE64_BYTES > _RESIZE_TARGET_BYTES

    def test_extreme_aspect_ratio_preserved(self, tmp_path):
        """Extreme aspect ratios should be preserved during resize."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        # Very wide panorama: 8000x200
        img = Image.new("RGB", (8000, 200), (100, 150, 200))
        path = tmp_path / "panorama.png"
        img.save(path, "PNG")

        result = _resize_image_for_vision(path, mime_type="image/png",
                                           max_base64_bytes=50_000)
        assert result.startswith("data:image/")
        # Decode and check aspect ratio is roughly preserved
        import base64
        header, b64data = result.split(",", 1)
        raw = base64.b64decode(b64data)
        from io import BytesIO
        resized = Image.open(BytesIO(raw))
        original_ratio = 8000 / 200  # 40:1
        resized_ratio = resized.width / resized.height if resized.height > 0 else 0
        # Allow some tolerance (floor clamping), but ratio should stay above 10:1
        # With independent halving, ratio would collapse to ~1:1. Proportional
        # scaling should keep it well above 10.
        assert resized_ratio > 10, (
            f"Aspect ratio collapsed: {resized.width}x{resized.height} "
            f"(ratio {resized_ratio:.1f}, expected >10)"
        )

    def test_tall_narrow_image_preserved(self, tmp_path):
        """Tall narrow images should also preserve aspect ratio."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        # Very tall: 200x6000
        img = Image.new("RGB", (200, 6000), (200, 100, 50))
        path = tmp_path / "tall.png"
        img.save(path, "PNG")

        result = _resize_image_for_vision(path, mime_type="image/png",
                                           max_base64_bytes=50_000)
        assert result.startswith("data:image/")
        import base64
        from io import BytesIO
        header, b64data = result.split(",", 1)
        raw = base64.b64decode(b64data)
        resized = Image.open(BytesIO(raw))
        original_ratio = 6000 / 200  # 30:1 (h/w)
        resized_ratio = resized.height / resized.width if resized.width > 0 else 0
        assert resized_ratio > 5, (
            f"Aspect ratio collapsed: {resized.width}x{resized.height} "
            f"(h/w ratio {resized_ratio:.1f}, expected >5)"
        )

    def test_no_pillow_returns_original(self, tmp_path):
        """Without Pillow, oversized images should be returned as-is."""
        # Create a dummy file
        path = tmp_path / "test.png"
        # Write enough bytes to exceed a tiny limit
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000)

        with patch("tools.vision_tools._image_to_base64_data_url") as mock_b64:
            # Simulate a large base64 result
            mock_b64.return_value = "data:image/png;base64," + "A" * 200
            with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
                result = _resize_image_for_vision(path, max_base64_bytes=100)
                # Should return the original (oversized) data url
                assert len(result) > 100


# ---------------------------------------------------------------------------
# _is_image_size_error — detect size-related API errors
# ---------------------------------------------------------------------------


class TestIsImageSizeError:
    """Tests for the size-error detection helper."""

    def test_too_large_message(self):
        assert _is_image_size_error(Exception("Request payload too large"))

    def test_413_status(self):
        assert _is_image_size_error(Exception("HTTP 413 Payload Too Large"))

    def test_invalid_request(self):
        assert _is_image_size_error(Exception("invalid_request_error: image too big"))

    def test_exceeds_limit(self):
        assert _is_image_size_error(Exception("Image exceeds maximum size"))

    def test_unrelated_error(self):
        assert not _is_image_size_error(Exception("Connection refused"))

    def test_auth_error(self):
        assert not _is_image_size_error(Exception("401 Unauthorized"))

    def test_empty_message(self):
        assert not _is_image_size_error(Exception(""))
