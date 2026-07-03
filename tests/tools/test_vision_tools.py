"""Tests for tools/vision_tools.py — URL validation, type hints, error logging."""

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
    _image_exceeds_dimension,
    _EMBED_MAX_DIMENSION,
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

    @pytest.mark.asyncio
    async def test_prompt_contains_question(self):
        """The full prompt should incorporate the user's question."""
        with (
            patch(
                "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
            ) as mock_tool,
            patch(
                "tools.vision_tools._should_use_native_vision_fast_path",
                return_value=False,
            ),
        ):
            mock_tool.return_value = json.dumps({"result": "ok"})
            await _handle_vision_analyze(
                {
                    "image_url": "https://example.com/img.png",
                    "question": "Describe the cat",
                }
            )
            call_args = mock_tool.call_args
            full_prompt = call_args[0][1]  # second positional arg
            assert "Describe the cat" in full_prompt
            assert "Fully describe and explain" in full_prompt

    @pytest.mark.asyncio
    async def test_uses_auxiliary_vision_model_env(self):
        """AUXILIARY_VISION_MODEL env var should override DEFAULT_VISION_MODEL."""
        with (
            patch(
                "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
            ) as mock_tool,
            patch(
                "tools.vision_tools._should_use_native_vision_fast_path",
                return_value=False,
            ),
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "custom/model-v1"}),
        ):
            mock_tool.return_value = json.dumps({"result": "ok"})
            await _handle_vision_analyze(
                {"image_url": "https://example.com/img.png", "question": "test"}
            )
            call_args = mock_tool.call_args
            model = call_args[0][2]  # third positional arg
            assert model == "custom/model-v1"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_model(self):
        """Without AUXILIARY_VISION_MODEL, model should be None (let call_llm resolve default)."""
        with (
            patch(
                "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
            ) as mock_tool,
            patch(
                "tools.vision_tools._should_use_native_vision_fast_path",
                return_value=False,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            # Ensure AUXILIARY_VISION_MODEL is not set
            os.environ.pop("AUXILIARY_VISION_MODEL", None)
            mock_tool.return_value = json.dumps({"result": "ok"})
            await _handle_vision_analyze(
                {"image_url": "https://example.com/img.png", "question": "test"}
            )
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
            patch("tools.vision_tools._validate_image_url_async", new_callable=AsyncMock, return_value=True),
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
            patch("tools.vision_tools._validate_image_url_async", new_callable=AsyncMock, return_value=True),
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


class TestVisionConfig:
    @pytest.mark.asyncio
    async def test_vision_uses_configured_temperature_and_timeout(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Configured image analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("hermes_cli.config.load_config", return_value={
                "auxiliary": {"vision": {"temperature": 1, "timeout": 77}}
            }),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/png;base64,abc",
            ),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_llm,
        ):
            result = json.loads(await vision_analyze_tool(str(img), "describe this", "test/model"))

        assert result["success"] is True
        assert mock_llm.await_args.kwargs["temperature"] == 1.0
        assert mock_llm.await_args.kwargs["timeout"] == 77.0

    @pytest.mark.asyncio
    async def test_vision_defaults_temperature_when_config_omits_it(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Default image analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("hermes_cli.config.load_config", return_value={"auxiliary": {"vision": {}}}),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/png;base64,abc",
            ),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_llm,
        ):
            result = json.loads(await vision_analyze_tool(str(img), "describe this", "test/model"))

        assert result["success"] is True
        assert mock_llm.await_args.kwargs["temperature"] == 0.1
        assert mock_llm.await_args.kwargs["timeout"] == 120.0


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
            patch("tools.vision_tools._validate_image_url_async", new_callable=AsyncMock, return_value=True),
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

        # Windows expanduser() prefers USERPROFILE over HOME; POSIX uses HOME.
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("USERPROFILE", str(fake_home))

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
        monkeypatch.setenv("USERPROFILE", str(fake_home))

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
# _image_exceeds_dimension — proactive embed-time pixel-cap detector
# ---------------------------------------------------------------------------


class TestImageExceedsDimension:
    """The proactive embed path checks pixel dimensions, not just bytes.

    A tall full-page screenshot can be well under the byte budget yet far
    over Anthropic's 8000px per-side cap (e.g. 1200x12000 at 0.06 MB). The
    byte-only embed guard let it slip into immutable history un-resized,
    bricking the session on a non-retryable 400. This helper flags it so the
    embed-time resize fires on dimensions too.
    """

    def test_tall_small_byte_image_flagged(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        # 1200x12000 solid color: trips the pixel cap, tiny in bytes.
        img = Image.new("RGB", (1200, 12000), (40, 40, 40))
        path = tmp_path / "tall.png"
        img.save(path, "PNG")
        assert _image_exceeds_dimension(path, _EMBED_MAX_DIMENSION) is True

    def test_small_image_not_flagged(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (800, 600), (10, 200, 10))
        path = tmp_path / "small.png"
        img.save(path, "PNG")
        assert _image_exceeds_dimension(path, _EMBED_MAX_DIMENSION) is False

    def test_exactly_at_cap_not_flagged(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (_EMBED_MAX_DIMENSION, 100), (1, 2, 3))
        path = tmp_path / "edge.png"
        img.save(path, "PNG")
        # max == cap is fine; only strictly greater forces a resize.
        assert _image_exceeds_dimension(path, _EMBED_MAX_DIMENSION) is False

    def test_missing_pillow_returns_false(self, tmp_path):
        # Without Pillow we can't inspect dimensions — return False so the
        # byte-based checks still apply and a missing soft dep never breaks
        # the embed path.
        path = tmp_path / "x.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            assert _image_exceeds_dimension(path, _EMBED_MAX_DIMENSION) is False

    def test_corrupt_file_returns_false(self, tmp_path):
        try:
            import PIL  # noqa: F401
        except ImportError:
            pytest.skip("Pillow not installed")
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"not an image at all")
        assert _image_exceeds_dimension(path, _EMBED_MAX_DIMENSION) is False


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


class TestDownloadRetryClassification:
    """Error-class-aware retry: 4xx fail-fast, 429/5xx/transient retried (issue #32296)."""

    @staticmethod
    def _status_error(status_code):
        import httpx

        request = httpx.Request("GET", "https://example.com/img.jpg")
        response = httpx.Response(status_code, request=request)
        return httpx.HTTPStatusError(
            f"{status_code}", request=request, response=response
        )

    def _make_client_raising_status(self, status_code):
        """AsyncClient whose response.raise_for_status() raises HTTPStatusError."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=self._status_error(status_code)
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        return mock_client

    def test_is_retryable_classification(self):
        from tools.vision_tools import _is_retryable_download_error

        # Non-retryable client errors
        for code in (400, 403, 404, 410):
            assert _is_retryable_download_error(self._status_error(code)) is False
        # Retryable: rate limit + server errors
        for code in (429, 500, 502, 503):
            assert _is_retryable_download_error(self._status_error(code)) is True
        # Policy/SSRF/size errors are terminal
        assert _is_retryable_download_error(PermissionError("blocked")) is False
        assert _is_retryable_download_error(ValueError("too large")) is False
        # Unclassified (network blip) is retryable
        assert _is_retryable_download_error(ConnectionError("reset")) is True

    @pytest.mark.asyncio
    async def test_404_fails_fast_without_retry(self, tmp_path):
        """A 404 must raise on the first attempt — no backoff sleep, no extra GETs."""
        import httpx
        from tools.vision_tools import _download_image

        mock_client = self._make_client_raising_status(404)
        with (
            patch("tools.vision_tools.httpx.AsyncClient", return_value=mock_client),
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(httpx.HTTPStatusError),
        ):
            await _download_image(
                "https://example.com/missing.jpg", tmp_path / "x.jpg", max_retries=3
            )
        # Exactly one attempt, zero backoff sleeps.
        assert mock_client.get.await_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_503_retries_then_raises(self, tmp_path):
        """A 5xx is retried up to max_retries, sleeping between attempts."""
        import httpx
        from tools.vision_tools import _download_image

        mock_client = self._make_client_raising_status(503)
        with (
            patch("tools.vision_tools.httpx.AsyncClient", return_value=mock_client),
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(httpx.HTTPStatusError),
        ):
            await _download_image(
                "https://example.com/flaky.jpg", tmp_path / "y.jpg", max_retries=3
            )
        # All three attempts used, two backoff sleeps between them.
        assert mock_client.get.await_count == 3
        assert mock_sleep.await_count == 2


# ---------------------------------------------------------------------------
# CPU-burst concurrency cap — a single turn (or several concurrent sessions in
# one process) can launch dozens of vision_analyze calls at once. Only the
# CPU-bound encode/resize is bounded (to host cores), so a video-frame storm
# can't saturate every core and starve the dashboard event loop — while the
# network-bound LLM calls stay fully concurrent for legitimate multi-image work.
# ---------------------------------------------------------------------------


class TestVisionCpuBurstCap:
    """The bounded CPU executor caps concurrent encode/resize, not LLM calls."""

    def test_resolver_defaults_to_host_cpus_no_ceiling(self):
        from tools import vision_tools as vt

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("tools.vision_tools._detect_host_cpus", return_value=64),
            patch("hermes_cli.config.load_config", side_effect=Exception),
        ):
            os.environ.pop("HERMES_VISION_MAX_CONCURRENCY", None)
            # No fixed ceiling: a 64-core host gets 64 encode workers. The cap
            # tracks the actual resource (cores), not a magic number.
            assert vt._resolve_vision_cpu_workers() == 64

    def test_resolver_respects_low_host_cpu_count(self):
        from tools import vision_tools as vt

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("tools.vision_tools._detect_host_cpus", return_value=2),
            patch("hermes_cli.config.load_config", side_effect=Exception),
        ):
            os.environ.pop("HERMES_VISION_MAX_CONCURRENCY", None)
            assert vt._resolve_vision_cpu_workers() == 2

    def test_resolver_env_override(self):
        from tools import vision_tools as vt

        with patch.dict(os.environ, {"HERMES_VISION_MAX_CONCURRENCY": "16"}):
            # Explicit override is honored verbatim — including ABOVE core count,
            # so operators can raise it for heavy multi-image workloads.
            assert vt._resolve_vision_cpu_workers() == 16

    def test_resolver_rejects_sub_one_override(self):
        from tools import vision_tools as vt

        with (
            patch.dict(os.environ, {"HERMES_VISION_MAX_CONCURRENCY": "0"}),
            patch("tools.vision_tools._detect_host_cpus", return_value=2),
            patch("hermes_cli.config.load_config", side_effect=Exception),
        ):
            # 0 is ignored (cap can never be disabled) → falls back to host cores.
            assert vt._resolve_vision_cpu_workers() == 2

    def test_cpu_executor_is_dedicated_and_sized_to_workers(self):
        """The encode executor must be dedicated, not the shared default pool."""
        import importlib
        from concurrent.futures import ThreadPoolExecutor

        vt = importlib.import_module("tools.vision_tools")
        assert isinstance(vt._vision_cpu_executor, ThreadPoolExecutor)
        assert vt._vision_cpu_executor._max_workers == vt._VISION_CPU_WORKERS

    @pytest.mark.asyncio
    async def test_encode_runs_on_dedicated_cpu_executor(self):
        """Encode/resize must execute on a ``vision-encode`` thread, off the loop.

        Regression guard: the CPU burst is what saturated cores and starved the
        loop. It must run on the bounded vision executor, not the caller's loop
        thread nor the shared default pool.
        """
        import importlib
        import threading

        vt = importlib.import_module("tools.vision_tools")

        seen_threads = []

        def fake_encode(path, mime_type=None):
            seen_threads.append(threading.current_thread().name)
            return "data:image/jpeg;base64,AAAA"

        result = await vt._run_encode_on_cpu_executor(fake_encode, "p", mime_type="image/jpeg")
        assert result == "data:image/jpeg;base64,AAAA"
        assert len(seen_threads) == 1
        assert seen_threads[0].startswith("vision-encode"), seen_threads

    @pytest.mark.asyncio
    async def test_encode_bursts_bounded_but_llm_stays_concurrent(self):
        """Encode concurrency is clamped to the cap; the LLM call is not.

        Drives many native-path calls whose encode step is the only thing on
        the CPU executor. With the executor sized to CAP, no more than CAP
        encodes ever run at once — even though all N calls are in flight
        simultaneously (proving the analyses themselves are NOT serialized).
        """
        import asyncio
        import importlib
        from concurrent.futures import ThreadPoolExecutor

        vt = importlib.import_module("tools.vision_tools")

        CAP = 3
        N = 12
        enc_inflight = 0
        enc_peak = 0
        calls_inflight = 0
        calls_peak = 0
        import threading as _t
        enc_lock = _t.Lock()

        def slow_encode(path, mime_type=None):
            nonlocal enc_inflight, enc_peak
            with enc_lock:
                enc_inflight += 1
                enc_peak = max(enc_peak, enc_inflight)
            try:
                _t.Event().wait(0.04)  # simulate CPU burst
            finally:
                with enc_lock:
                    enc_inflight -= 1
            return "data:image/jpeg;base64,AAAA"

        async def fake_native(image_url, question):
            nonlocal calls_inflight, calls_peak
            calls_inflight += 1
            calls_peak = max(calls_peak, calls_inflight)
            try:
                # The encode is the capped CPU step.
                await vt._run_encode_on_cpu_executor(slow_encode, "p", mime_type="image/jpeg")
                # The "LLM call" is NOT capped — overlaps freely.
                await asyncio.sleep(0.02)
            finally:
                calls_inflight -= 1
            return json.dumps({"ok": True})

        with (
            patch.object(vt, "_vision_cpu_executor",
                         ThreadPoolExecutor(max_workers=CAP, thread_name_prefix="vision-encode")),
            patch.object(vt, "_should_use_native_vision_fast_path", return_value=True),
            patch.object(vt, "_vision_analyze_native", side_effect=fake_native),
        ):
            await asyncio.gather(*[
                vt._handle_vision_analyze(
                    {"image_url": f"https://example.com/frame_{i}.png",
                     "question": "what is this"}
                )
                for i in range(N)
            ])

        assert enc_peak <= CAP, f"encode peak {enc_peak} exceeded cap {CAP}"
        assert enc_peak == CAP, f"expected to saturate encode cap {CAP}, got {enc_peak}"
        # The analyses themselves were NOT serialized to the cap — all N ran
        # concurrently, which is the whole point (multi-image workflows keep
        # their concurrency; only the CPU burst is bounded).
        assert calls_peak > CAP, (
            f"analyses were serialized to the cap (peak={calls_peak}); only the "
            "encode burst should be bounded, not the whole call"
        )
