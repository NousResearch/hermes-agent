"""Tests for tools/vision_tools.py — URL validation, type hints, error logging."""

import base64
import json
import logging
import os
from contextlib import ExitStack
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


_RESOLVES = [(2, 1, 6, "", ("93.184.216.34", 0))]


# ---------------------------------------------------------------------------
# _validate_image_url — urlparse-based validation
# ---------------------------------------------------------------------------


class TestValidateImageUrl:
    """Tests for URL validation, including urlparse-based netloc check."""

    def test_accepts_valid_http_and_https_urls(self):
        with patch("tools.url_safety.socket.getaddrinfo", return_value=_RESOLVES):
            assert _validate_image_url("https://example.com/image.jpg") is True
            assert _validate_image_url("http://cdn.example.org/photo.png") is True
            # CDN endpoints that redirect to images should still pass.
            assert _validate_image_url("https://cdn.example.com/abcdef123") is True
            assert _validate_image_url("https://img.example.com/pic?w=200&h=200") is True
            assert _validate_image_url("http://example.com:8080/image.png") is True
            assert _validate_image_url("https://example.com/") is True

    def test_localhost_url_blocked_by_ssrf(self):
        """localhost URLs are blocked by SSRF protection."""
        assert _validate_image_url("http://localhost:8080/image.png") is False


    def test_rejects_malformed_and_non_string_inputs(self):
        # http:// alone has no network location — urlparse catches this.
        assert _validate_image_url("http://") is False
        assert _validate_image_url("https://") is False
        assert _validate_image_url("http:") is False
        assert _validate_image_url("") is False
        assert _validate_image_url("   ") is False
        assert _validate_image_url(None) is False
        assert _validate_image_url(12345) is False
        assert _validate_image_url(True) is False
        assert _validate_image_url(["https://example.com"]) is False


# ---------------------------------------------------------------------------
# _determine_mime_type
# ---------------------------------------------------------------------------


class TestDetermineMimeType:
    def test_extension_maps_to_mime_type(self):
        assert _determine_mime_type(Path("photo.jpg")) == "image/jpeg"
        assert _determine_mime_type(Path("photo.jpeg")) == "image/jpeg"
        assert _determine_mime_type(Path("screenshot.png")) == "image/png"
        assert _determine_mime_type(Path("anim.gif")) == "image/gif"
        assert _determine_mime_type(Path("modern.webp")) == "image/webp"
        # Unknown extensions fall back to JPEG.
        assert _determine_mime_type(Path("file.xyz")) == "image/jpeg"


# ---------------------------------------------------------------------------
# _image_to_base64_data_url
# ---------------------------------------------------------------------------


class TestImageToBase64DataUrl:
    def test_returns_data_url_with_detected_or_given_mime(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        assert _image_to_base64_data_url(img).startswith("data:image/png;base64,")

        other = tmp_path / "test.bin"
        other.write_bytes(b"\x00" * 16)
        result = _image_to_base64_data_url(other, mime_type="image/webp")
        assert result.startswith("data:image/webp;base64,")

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _image_to_base64_data_url(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# _handle_vision_analyze — type signature & behavior
# ---------------------------------------------------------------------------


class TestHandleVisionAnalyze:
    """Verify _handle_vision_analyze returns an Awaitable and builds correct prompt."""

    def test_returns_awaitable_even_for_empty_args(self):
        """The handler is registered as async, and missing keys must not raise."""
        with patch(
            "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
        ) as mock_tool:
            mock_tool.return_value = json.dumps({"result": "ok"})
            for args in ({"image_url": "https://example.com/img.png",
                          "question": "What is this?"}, {}):
                result = _handle_vision_analyze(args)
                assert isinstance(result, Awaitable)
                # Clean up the coroutine to avoid RuntimeWarning
                result.close()


    @pytest.mark.asyncio
    async def test_model_resolution_config_then_env_then_default(self):
        """config.yaml auxiliary.vision.model wins; then AUXILIARY_VISION_MODEL;
        then None, letting the centralized call_llm router pick the default."""

        async def resolve(config=None, env_model=None):
            with ExitStack() as st:
                mock_tool = st.enter_context(
                    patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock)
                )
                st.enter_context(patch(
                    "tools.vision_tools._should_use_native_vision_fast_path",
                    return_value=False,
                ))
                st.enter_context(patch.dict(os.environ, {}, clear=False))
                if config is not None:
                    st.enter_context(patch(
                        "hermes_cli.config.load_config", return_value=config,
                    ))
                if env_model is None:
                    os.environ.pop("AUXILIARY_VISION_MODEL", None)
                else:
                    os.environ["AUXILIARY_VISION_MODEL"] = env_model
                mock_tool.return_value = json.dumps({"result": "ok"})
                await _handle_vision_analyze(
                    {"image_url": "https://example.com/img.png", "question": "test"}
                )
                return mock_tool.call_args[0][2]  # third positional arg

        assert await resolve(env_model="custom/model-v1") == "custom/model-v1"
        assert await resolve() is None
        assert await resolve(
            config={"auxiliary": {"vision": {"model": "qwen3.7-plus"}}},
            env_model="env-model",
        ) == "qwen3.7-plus"
        assert await resolve(
            config={"auxiliary": {"vision": {}}}, env_model="fallback-model",
        ) == "fallback-model"


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
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream.side_effect = ConnectionError("network down")
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
    async def test_cleanup_error_logs_exc_info(self, tmp_path, caplog):
        """Temp file cleanup failure should log warning with exc_info."""
        # A data: URL resolves to bytes without any network, materializes to a
        # vision temp file, then the analysis runs — exercising the temp-cleanup
        # path in the finally block. A tiny valid JPEG (magic bytes) passes the
        # resolver's magic-byte sniff.
        jpeg_b64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 32).decode("ascii")
        data_url = f"data:image/jpeg;base64,{jpeg_b64}"

        with (
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/jpeg;base64,abc",
            ),
            caplog.at_level(logging.WARNING, logger="tools.vision_tools"),
        ):
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "A test image description"
            mock_response.choices = [mock_choice]

            with (
                patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response),
            ):
                # Make unlink fail to trigger the cleanup warning.
                def failing_unlink(self, *args, **kwargs):
                    raise PermissionError("no permission")

                with patch.object(Path, "unlink", failing_unlink):
                    await vision_analyze_tool(data_url, "describe", "test/model")

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
    async def test_temperature_and_timeout_come_from_config_with_defaults(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        async def call_with(config):
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Configured image analysis"
            mock_response.choices = [mock_choice]

            with (
                patch("hermes_cli.config.load_config", return_value=config),
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
                result = json.loads(
                    await vision_analyze_tool(str(img), "describe this", "test/model")
                )
            assert result["success"] is True
            return mock_llm.await_args.kwargs

        kwargs = await call_with(
            {"auxiliary": {"vision": {"temperature": 1, "timeout": 77}}}
        )
        assert kwargs["temperature"] == 1.0
        assert kwargs["timeout"] == 77.0
        # No hardcoded output cap — the aux client omits max_tokens so the
        # provider uses its full output budget (max-tokens-knob policy).
        assert "max_tokens" not in kwargs

        # Omitted values fall back to the built-in defaults.
        kwargs = await call_with({"auxiliary": {"vision": {}}})
        assert kwargs["temperature"] == 0.1
        assert kwargs["timeout"] == 120.0
        assert "max_tokens" not in kwargs

        # Even an explicit auxiliary.vision.max_tokens config entry must NOT
        # be forwarded: user-facing max_tokens knobs are policy-prohibited.
        kwargs = await call_with({"auxiliary": {"vision": {"max_tokens": 8000}}})
        assert "max_tokens" not in kwargs
        assert kwargs["temperature"] == 0.1
        assert kwargs["timeout"] == 120.0


class TestVisionSafetyGuards:
    @pytest.mark.asyncio
    async def test_local_non_image_file_rejected_before_llm_call(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP-SECRET=1\n", encoding="utf-8")

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = json.loads(await vision_analyze_tool(str(secret), "extract text"))

        assert result["success"] is False
        # The unified resolver's magic-byte sniff rejects non-images.
        assert "not a recognized image" in result["error"]
        mock_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_local_env_file_blocked_via_read_guard(self, tmp_path):
        """A .env file must be blocked even if given an image-like path.

        Mirrors the video_analyze_tool regression: the local-file branch
        must route through agent.file_safety.raise_if_read_blocked before
        vision_analyze_tool ever opens the file, not rely solely on the
        magic-byte mime check as an accidental side effect.
        """
        secret = tmp_path / ".env"
        secret.write_text("OPENAI_API_KEY=sk-super-secret\n", encoding="utf-8")

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = json.loads(await vision_analyze_tool(str(secret), "extract text"))

        assert result["success"] is False
        assert "secret-bearing environment file" in result["error"]
        mock_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_native_fast_path_local_env_file_blocked_via_read_guard(self, tmp_path):
        """Same read guard must apply to the native vision fast path."""
        from tools.vision_tools import _vision_analyze_native

        secret = tmp_path / ".env"
        secret.write_text("OPENAI_API_KEY=sk-super-secret\n", encoding="utf-8")

        result = json.loads(await _vision_analyze_native(str(secret), "extract text"))

        assert result["success"] is False
        assert "secret-bearing environment file" in result["error"]

    @pytest.mark.asyncio
    async def test_blocked_remote_url_short_circuits_before_download(self):
        blocked = {
            "host": "blocked.test",
            "rule": "blocked.test",
            "source": "config",
            "message": "Blocked by website policy",
        }

        with (
            patch("tools.website_policy.check_website_access", return_value=blocked),
            patch("tools.url_safety.is_safe_url", return_value=True),
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

        class _FakeStreamResponse:
            url = "https://blocked.test/final.png"
            headers = {"content-length": "24"}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

        class _FakeAsyncStream:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with (
            patch("tools.vision_tools.check_website_access", side_effect=fake_check),
            patch("tools.vision_tools.httpx.AsyncClient") as mock_client_cls,
            pytest.raises(PermissionError, match="Blocked by website policy"),
        ):
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream.return_value = _FakeAsyncStream(_FakeStreamResponse())
            mock_client_cls.return_value = mock_client

            await _download_image(
                "https://allowed.test/cat.png", tmp_path / "cat.png", max_retries=1
            )

        assert not (tmp_path / "cat.png").exists()

    @pytest.mark.asyncio
    async def test_download_enforces_size_cap_while_streaming(self, tmp_path):
        """Streaming download rejects oversize payloads chunk-by-chunk and cleans up."""
        from tools.vision_tools import _download_image

        class _FakeStreamResponse:
            url = "https://example.com/big.png"
            headers = {}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield b"12345"
                yield b"678901"

        class _FakeAsyncStream:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with (
            patch("tools.vision_tools._VISION_MAX_DOWNLOAD_BYTES", 10),
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.vision_tools.httpx.AsyncClient") as mock_client_cls,
            pytest.raises(ValueError, match="Image too large"),
        ):
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream.return_value = _FakeAsyncStream(_FakeStreamResponse())
            mock_client_cls.return_value = mock_client

            await _download_image(
                "https://example.com/big.png", tmp_path / "big.png", max_retries=1
            )

        assert not (tmp_path / "big.png").exists()
        assert not list(tmp_path.glob(".big.png.*.tmp"))

    @pytest.mark.asyncio
    async def test_download_ignores_malformed_content_length(self, tmp_path):
        """Malformed Content-Length is ignored; streaming size cap still works."""
        from tools.vision_tools import _download_image

        body = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        dest = tmp_path / "cat.png"

        class _FakeStreamResponse:
            url = "https://example.com/cat.png"
            headers = {"content-length": "not-a-number"}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield body[:4]
                yield body[4:]

        class _FakeAsyncStream:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.vision_tools.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream.return_value = _FakeAsyncStream(_FakeStreamResponse())
            mock_client_cls.return_value = mock_client

            await _download_image("https://example.com/cat.png", dest, max_retries=1)

        assert dest.read_bytes() == body


# ---------------------------------------------------------------------------
# check_vision_requirements
# ---------------------------------------------------------------------------


class TestVisionRequirements:
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
# Local path forms: tilde expansion and file:// URIs
# ---------------------------------------------------------------------------


class TestLocalPathForms:
    """Verify ~/path and file:// URIs resolve as local files."""

    @pytest.mark.asyncio
    async def test_tilde_path_expanded_to_local_file(self, tmp_path, monkeypatch):
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
            data = json.loads(await vision_analyze_tool(
                "~/test_image.png", "describe this", "test/model"
            ))
            assert data["success"] is True
            assert data["analysis"] == "A test image"

        # A tilde path that doesn't resolve to a real file fails gracefully.
        data = json.loads(await vision_analyze_tool(
            "~/nonexistent.png", "describe this", "test/model"
        ))
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_file_uri_resolved_as_local_path(self, tmp_path):
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
            data = json.loads(await vision_analyze_tool(
                f"file://{img}", "describe this", "test/model"
            ))
            assert data["success"] is True

        # file:// pointing to a missing file fails gracefully.
        data = json.loads(await vision_analyze_tool(
            f"file://{tmp_path}/nonexistent.png", "describe this", "test/model"
        ))
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Base64 size pre-flight check
# ---------------------------------------------------------------------------


class TestBase64SizeLimit:
    """Verify that oversized images are rejected before hitting the API."""

    @pytest.mark.asyncio
    async def test_oversized_rejected_before_api_call_small_passes(self, tmp_path):
        img = tmp_path / "huge.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (4 * 1024 * 1024))

        # Patch the hard limit to a small value so the test runs fast.
        with patch("tools.vision_tools._MAX_BASE64_BYTES", 1000), \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = json.loads(await vision_analyze_tool(str(img), "describe this"))

        assert result["success"] is False
        assert "too large" in result["error"].lower()
        mock_llm.assert_not_awaited()

        # An image well under the limit passes the size check.
        small = tmp_path / "small.png"
        small.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Small image"
        mock_response.choices = [mock_choice]

        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = json.loads(
                await vision_analyze_tool(str(small), "describe this", "test/model")
            )

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
    def test_vision_analyze_registered_with_schema(self):
        from tools.registry import registry

        entry = registry._tools.get("vision_analyze")
        assert entry is not None
        assert entry.toolset == "vision"
        assert entry.is_async is True
        assert callable(entry.handler)

        props = entry.schema.get("parameters", {}).get("properties", {})
        assert "image_url" in props
        assert "question" in props


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

    def test_force_jpeg_keeps_resolution_of_dense_png(self, tmp_path):
        """A dense PNG over the byte cap shrinks via JPEG quality, not halving.

        PNG has no quality ladder — before force_jpeg, the only shrink lever
        for a text-dense screenshot was halving dimensions (1568px → ~784px),
        destroying text legibility (#92699 follow-up). With force_jpeg the
        quality ladder absorbs the byte pressure and resolution survives.
        """
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        import base64 as _b64
        import random
        from io import BytesIO

        # Photo-like PNG at exactly the long-edge cap: a gradient with
        # low-amplitude noise defeats PNG's lossless filters (no exact
        # repeats → >256KB) but JPEG's DCT quantization absorbs it easily —
        # the realistic screenshot shape, unlike pure noise which no codec
        # can fit in-budget at full resolution.
        rng = random.Random(42)
        img = Image.new("RGB", (1568, 900))
        img.putdata([
            (
                min(255, (x * 255) // 1568 + rng.randrange(12)),
                min(255, (y * 255) // 900 + rng.randrange(12)),
                min(255, ((x + y) * 255) // 2468 + rng.randrange(12)),
            )
            for y in range(900)
            for x in range(1568)
        ])
        path = tmp_path / "dense.png"
        img.save(path, "PNG")

        budget = 256 * 1024
        result = _resize_image_for_vision(
            path, mime_type="image/png",
            max_base64_bytes=budget, max_dimension=1568,
            force_jpeg=True,
        )
        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) <= budget
        with Image.open(BytesIO(_b64.b64decode(result.partition(",")[2]))) as out:
            # Resolution preserved (the PNG path would have halved to 784px).
            assert max(out.size) == 1568

    def test_force_jpeg_leaves_small_png_untouched(self, tmp_path):
        """Under-cap images skip the resize entirely — still PNG."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (10, 10), (0, 255, 0))
        path = tmp_path / "tiny.png"
        img.save(path, "PNG")

        result = _resize_image_for_vision(
            path, mime_type="image/png",
            max_base64_bytes=256 * 1024, max_dimension=1568,
            force_jpeg=True,
        )
        assert result.startswith("data:image/png;base64,")

    def test_force_jpeg_handles_alpha_and_exotic_modes(self, tmp_path):
        """RGBA/LA/P PNGs must convert cleanly — JPEG can't encode alpha.

        force_jpeg newly routes PNG inputs to the JPEG encoder, so modes
        JPEG can't save (LA grayscale+alpha especially) must be normalized
        to RGB instead of crashing img.save().
        """
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        import random

        rng = random.Random(7)
        for mode, size in (("RGBA", (600, 400)), ("LA", (600, 400)), ("P", (600, 400))):
            img = Image.new(mode, size)
            # Noise so the PNG exceeds a tiny budget and the resize fires.
            if mode == "P":
                img.putpalette([i % 256 for i in range(768)])
                img.putdata([rng.randrange(256) for _ in range(size[0] * size[1])])
            else:
                bands = len(img.getbands())
                img.putdata([
                    tuple(rng.randrange(256) for _ in range(bands))
                    for _ in range(size[0] * size[1])
                ])
            path = tmp_path / f"img_{mode}.png"
            img.save(path, "PNG")

            result = _resize_image_for_vision(
                path, mime_type="image/png",
                max_base64_bytes=16 * 1024, max_dimension=1568,
                force_jpeg=True,
            )
            assert result.startswith("data:image/jpeg;base64,"), mode


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


    def test_undetectable_dimensions_return_false(self, tmp_path):
        # Without Pillow — or with bytes Pillow can't parse — we can't inspect
        # dimensions, so return False: the byte-based checks still apply and a
        # missing soft dep never breaks the embed path.
        path = tmp_path / "x.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            assert _image_exceeds_dimension(path, _EMBED_MAX_DIMENSION) is False

        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image at all")
        assert _image_exceeds_dimension(corrupt, _EMBED_MAX_DIMENSION) is False


# ---------------------------------------------------------------------------
# _is_image_size_error — detect size-related API errors
# ---------------------------------------------------------------------------


class TestIsImageSizeError:
    """Tests for the size-error detection helper."""

    def test_only_size_related_errors_match(self):
        assert _is_image_size_error(Exception("Request payload too large"))
        assert _is_image_size_error(Exception("HTTP 413 Payload Too Large"))
        assert _is_image_size_error(Exception("invalid_request_error: image too big"))
        assert _is_image_size_error(Exception("Image exceeds maximum size"))

        assert not _is_image_size_error(Exception("Connection refused"))
        assert not _is_image_size_error(Exception("401 Unauthorized"))
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
        """AsyncClient whose stream response.raise_for_status() raises HTTPStatusError."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=self._status_error(status_code)
        )
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_stream)
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
        assert mock_client.stream.call_count == 3
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

    def test_worker_count_tracks_host_cpus_with_env_override(self):
        from tools import vision_tools as vt

        def resolve(host_cpus, env_value=None):
            with (
                patch.dict(os.environ, {}, clear=False),
                patch("tools.vision_tools._detect_host_cpus", return_value=host_cpus),
                patch("hermes_cli.config.load_config", side_effect=Exception),
            ):
                if env_value is None:
                    os.environ.pop("HERMES_VISION_MAX_CONCURRENCY", None)
                else:
                    os.environ["HERMES_VISION_MAX_CONCURRENCY"] = env_value
                return vt._resolve_vision_cpu_workers()

        # No fixed ceiling: a 64-core host gets 64 encode workers. The cap
        # tracks the actual resource (cores), not a magic number.
        assert resolve(64) == 64
        assert resolve(2) == 2
        # Explicit override is honored verbatim — including ABOVE core count,
        # so operators can raise it for heavy multi-image workloads.
        assert resolve(2, "16") == 16
        # 0 is ignored (cap can never be disabled) → falls back to host cores.
        assert resolve(2, "0") == 2

    @pytest.mark.asyncio
    async def test_encode_runs_on_dedicated_cpu_executor(self):
        """Encode/resize must execute on a ``vision-encode`` thread, off the loop.

        Regression guard: the CPU burst is what saturated cores and starved the
        loop. It must run on the bounded vision executor, not the caller's loop
        thread nor the shared default pool.
        """
        import importlib
        import threading
        from concurrent.futures import ThreadPoolExecutor

        vt = importlib.import_module("tools.vision_tools")

        # The executor is dedicated, not the shared default pool.
        assert isinstance(vt._vision_cpu_executor, ThreadPoolExecutor)
        assert vt._vision_cpu_executor._max_workers == vt._VISION_CPU_WORKERS

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

        async def fake_native(image_url, question, task_id=None, **_kw):
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
