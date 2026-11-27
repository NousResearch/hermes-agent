"""Additional coverage tests for tools/vision_tools.py (issue #36601)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from PIL import Image
import pytest

from tools.vision_tools import (
    VISION_ANALYZE_SCHEMA,
    VIDEO_ANALYZE_SCHEMA,
    _ANTHROPIC_SUPPORTED_MEDIA_TYPES,
    _EMBED_MAX_DIMENSION,
    _EMBED_TARGET_BYTES,
    _MAX_BASE64_BYTES,
    _MAX_VIDEO_BASE64_BYTES,
    _RESIZE_TARGET_BYTES,
    _VIDEO_MIME_TYPES,
    _VIDEO_SIZE_WARN_BYTES,
    _VISION_MAX_VALIDATED_AGGREGATE_PIXELS,
    _VISION_MAX_VALIDATED_FRAME_COUNT,
    _build_native_vision_tool_result,
    _build_scale_note,
    _crop_image_region,
    _detect_host_cpus,
    _detect_image_mime_type_from_bytes,
    _detect_video_mime_type,
    _determine_mime_type,
    _download_image,
    _download_video,
    _handle_video_analyze,
    _handle_vision_analyze,
    _image_exceeds_dimension,
    _image_to_base64_data_url,
    _image_url_shape_ok,
    _is_image_size_error,
    _is_path_like_video_source,
    _is_retryable_download_error,
    _load_auxiliary_client,
    _materialize_video_from_terminal_backend,
    _normalize_to_supported_image,
    _rasterize_svg_to_png,
    _resize_image_for_vision,
    _resolve_download_timeout,
    _resolve_vision_cpu_workers,
    _run_encode_on_cpu_executor,
    _should_use_native_vision_fast_path,
    _stream_download_to_file,
    _supports_media_in_tool_results,
    _terminal_backend_is_local,
    _validate_image_url,
    _validate_image_url_async,
    _validate_raster_image_decodable,
    _video_to_base64_data_url,
    _vision_analyze_native,
    _vision_concurrency_slot,
    check_vision_requirements,
    video_analyze_tool,
    vision_analyze_tool,
)


# Structurally valid 1x1 RGB PNG constant
VALID_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _create_image_bytes(fmt: str, size: tuple[int, int] = (2, 2), mode: str = "RGB", color: str = "red") -> bytes:
    img = Image.new(mode, size, color=color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _create_animated_gif_bytes(num_frames: int = 2, size: tuple[int, int] = (4, 4)) -> bytes:
    frames = [Image.new("RGB", size, color=c) for c in ("red", "blue", "green")[:num_frames]]
    buf = BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. _detect_image_mime_type_from_bytes
# ---------------------------------------------------------------------------


class TestDetectImageMimeTypeFromBytes:
    """Tests for magic-byte MIME sniff on raw bytes."""

    def test_valid_png_detected(self):
        assert _detect_image_mime_type_from_bytes(VALID_PNG) == "image/png"

    def test_corrupt_png_rejected(self):
        corrupt = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        assert _detect_image_mime_type_from_bytes(corrupt) is None

    def test_png_fallback_when_pil_import_fails(self):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            result = _detect_image_mime_type_from_bytes(VALID_PNG)
            assert result == "image/png"

    def test_jpeg_sniff(self):
        jpeg_bytes = _create_image_bytes("JPEG")
        assert _detect_image_mime_type_from_bytes(jpeg_bytes) == "image/jpeg"

    def test_gif_sniff(self):
        gif89_bytes = b"GIF89a" + b"\x00" * 10
        gif87_bytes = b"GIF87a" + b"\x00" * 10
        assert _detect_image_mime_type_from_bytes(gif89_bytes) == "image/gif"
        assert _detect_image_mime_type_from_bytes(gif87_bytes) == "image/gif"

    def test_bmp_sniff(self):
        bmp_bytes = _create_image_bytes("BMP")
        assert _detect_image_mime_type_from_bytes(bmp_bytes) == "image/bmp"

    def test_webp_sniff(self):
        webp_bytes = _create_image_bytes("WEBP")
        assert _detect_image_mime_type_from_bytes(webp_bytes) == "image/webp"

    def test_unrecognized_bytes_returns_none(self):
        assert _detect_image_mime_type_from_bytes(b"not-an-image-data-header") is None
        assert _detect_image_mime_type_from_bytes(b"") is None


# ---------------------------------------------------------------------------
# 2. _rasterize_svg_to_png
# ---------------------------------------------------------------------------


class TestRasterizeSvgToPng:
    """Tests for SVG to PNG rasterization across cairosvg, svglib, and CLI tools."""

    def test_cairosvg_strategy_success(self, tmp_path):
        svg_file = tmp_path / "test.svg"
        svg_file.write_text("<svg></svg>")
        out_png = tmp_path / "out.png"

        mock_cairosvg = MagicMock()
        def fake_svg2png(url=None, write_to=None):
            Path(write_to).write_bytes(VALID_PNG)

        mock_cairosvg.svg2png = fake_svg2png
        with patch.dict("sys.modules", {"cairosvg": mock_cairosvg}):
            assert _rasterize_svg_to_png(svg_file, out_png) is True
            assert out_png.exists()

    def test_svglib_reportlab_strategy_success(self, tmp_path):
        svg_file = tmp_path / "test.svg"
        svg_file.write_text("<svg></svg>")
        out_png = tmp_path / "out.png"

        mock_svglib = MagicMock()
        mock_svglib.svglib = MagicMock()
        mock_svglib.svglib.svg2rlg = MagicMock(return_value=MagicMock())

        mock_reportlab = MagicMock()
        mock_reportlab.graphics = MagicMock()
        mock_renderPM = MagicMock()
        def fake_draw_to_file(drawing, path, fmt="PNG"):
            Path(path).write_bytes(VALID_PNG)
        mock_renderPM.drawToFile = fake_draw_to_file
        mock_reportlab.graphics.renderPM = mock_renderPM

        with patch.dict("sys.modules", {
            "cairosvg": None,
            "svglib": mock_svglib,
            "svglib.svglib": mock_svglib.svglib,
            "reportlab": mock_reportlab,
            "reportlab.graphics": mock_reportlab.graphics,
            "reportlab.graphics.renderPM": mock_renderPM,
        }):
            assert _rasterize_svg_to_png(svg_file, out_png) is True
            assert out_png.exists()

    def test_system_rasterizer_rsvg_convert_success(self, tmp_path):
        svg_file = tmp_path / "test.svg"
        svg_file.write_text("<svg></svg>")
        out_png = tmp_path / "out.png"

        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "rsvg-convert":
                out_png.write_bytes(VALID_PNG)
                return MagicMock(returncode=0)
            raise FileNotFoundError()

        with (
            patch.dict("sys.modules", {"cairosvg": None, "svglib": None, "reportlab": None}),
            patch("shutil.which", side_effect=lambda binary: binary if binary == "rsvg-convert" else None),
            patch("subprocess.run", side_effect=fake_run),
        ):
            assert _rasterize_svg_to_png(svg_file, out_png) is True
            assert out_png.exists()

    def test_system_rasterizer_inkscape_success(self, tmp_path):
        svg_file = tmp_path / "test.svg"
        svg_file.write_text("<svg></svg>")
        out_png = tmp_path / "out.png"

        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "inkscape":
                out_png.write_bytes(VALID_PNG)
                return MagicMock(returncode=0)
            raise FileNotFoundError()

        with (
            patch.dict("sys.modules", {"cairosvg": None, "svglib": None, "reportlab": None}),
            patch("shutil.which", side_effect=lambda binary: binary if binary == "inkscape" else None),
            patch("subprocess.run", side_effect=fake_run),
        ):
            assert _rasterize_svg_to_png(svg_file, out_png) is True
            assert out_png.exists()

    def test_all_rasterizers_fail_returns_false(self, tmp_path):
        svg_file = tmp_path / "test.svg"
        svg_file.write_text("<svg></svg>")
        out_png = tmp_path / "out.png"

        with (
            patch.dict("sys.modules", {"cairosvg": None, "svglib": None, "reportlab": None}),
            patch("shutil.which", return_value=None),
        ):
            assert _rasterize_svg_to_png(svg_file, out_png) is False
            assert not out_png.exists()

    def test_system_rasterizer_rsvg_convert_exception_continues(self, tmp_path):
        svg_file = tmp_path / "test.svg"
        svg_file.write_text("<svg></svg>")
        out_png = tmp_path / "out.png"

        with (
            patch.dict("sys.modules", {"cairosvg": None, "svglib": None, "reportlab": None}),
            patch("shutil.which", return_value="rsvg-convert"),
            patch("subprocess.run", side_effect=Exception("binary crashed")),
        ):
            assert _rasterize_svg_to_png(svg_file, out_png) is False


# ---------------------------------------------------------------------------
# 3. _normalize_to_supported_image
# ---------------------------------------------------------------------------


class TestNormalizeToSupportedImage:
    """Tests for image normalization to Anthropic-compatible raster types."""

    def test_already_supported_mime_is_passthrough(self, tmp_path):
        jpeg_path = tmp_path / "image.jpg"
        jpeg_path.write_bytes(_create_image_bytes("JPEG"))
        path, mime, err = _normalize_to_supported_image(jpeg_path, "image/jpeg")
        assert path == jpeg_path
        assert mime == "image/jpeg"
        assert err is None

    def test_svg_rasterization_success(self, tmp_path):
        svg_path = tmp_path / "image.svg"
        svg_path.write_text("<svg></svg>")

        with patch("tools.vision_tools._rasterize_svg_to_png", return_value=True):
            path, mime, err = _normalize_to_supported_image(svg_path, "image/svg+xml")
            assert path is not None
            assert mime == "image/png"
            assert err is None

    def test_svg_rasterization_failure_returns_actionable_error(self, tmp_path):
        svg_path = tmp_path / "image.svg"
        svg_path.write_text("<svg></svg>")

        with patch("tools.vision_tools._rasterize_svg_to_png", return_value=False):
            path, mime, err = _normalize_to_supported_image(svg_path, "image/svg+xml")
            assert path is None
            assert mime is None
            assert err is not None
            assert "SVG" in err
            assert "rasterizer" in err

    def test_bmp_converted_to_png(self, tmp_path):
        bmp_path = tmp_path / "image.bmp"
        bmp_path.write_bytes(_create_image_bytes("BMP", size=(10, 10), mode="RGB"))
        path, mime, err = _normalize_to_supported_image(bmp_path, "image/bmp")
        assert path is not None
        assert path != bmp_path
        assert mime == "image/png"
        assert err is None
        assert path.exists()
        assert bmp_path.exists()
        path.unlink(missing_ok=True)

    def test_non_rgb_image_converted_to_rgba_png(self, tmp_path):
        bmp_path = tmp_path / "palette.bmp"
        bmp_path.write_bytes(_create_image_bytes("BMP", size=(5, 5), mode="P"))
        path, mime, err = _normalize_to_supported_image(bmp_path, "image/bmp")
        assert path is not None
        assert mime == "image/png"
        assert err is None
        path.unlink(missing_ok=True)

    def test_unsupported_corrupt_format_returns_error(self, tmp_path):
        corrupt_bmp = tmp_path / "bad.bmp"
        corrupt_bmp.write_bytes(b"BMnot-a-real-bmp")
        path, mime, err = _normalize_to_supported_image(corrupt_bmp, "image/bmp")
        assert path is None
        assert mime is None
        assert err is not None
        assert "not supported" in err or "could not be converted" in err


# ---------------------------------------------------------------------------
# 4. _validate_raster_image_decodable
# ---------------------------------------------------------------------------


class TestValidateRasterImageDecodable:
    """Tests for frame and pixel decodability validation."""

    def test_valid_png_passes_validation(self, tmp_path):
        img_path = tmp_path / "valid.png"
        img_path.write_bytes(VALID_PNG)
        assert _validate_raster_image_decodable(img_path) is None

    def test_corrupt_raster_image_returns_error_string(self, tmp_path):
        bad_png = tmp_path / "truncated.png"
        bad_png.write_bytes(VALID_PNG[: len(VALID_PNG) - 10])
        err = _validate_raster_image_decodable(bad_png)
        assert err is not None
        assert "could not be fully decoded" in err

    def test_animation_frame_count_exceeded(self, tmp_path, monkeypatch):
        gif_path = tmp_path / "anim.gif"
        gif_path.write_bytes(_create_animated_gif_bytes(num_frames=3))
        monkeypatch.setattr("tools.vision_tools._VISION_MAX_VALIDATED_FRAME_COUNT", 1)
        err = _validate_raster_image_decodable(gif_path)
        assert err is not None
        assert "frame 2 exceeds the maximum 1 validated frames" in err

    def test_animation_aggregate_pixels_exceeded(self, tmp_path, monkeypatch):
        gif_path = tmp_path / "anim_large.gif"
        gif_path.write_bytes(_create_animated_gif_bytes(num_frames=2, size=(10, 10)))
        monkeypatch.setattr("tools.vision_tools._VISION_MAX_VALIDATED_AGGREGATE_PIXELS", 50)
        err = _validate_raster_image_decodable(gif_path)
        assert err is not None
        assert "exceeding the maximum 50" in err

    def test_pillow_missing_skips_validation(self, tmp_path):
        img_path = tmp_path / "valid.png"
        img_path.write_bytes(VALID_PNG)
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None, "PIL.ImageSequence": None}):
            assert _validate_raster_image_decodable(img_path) is None


# ---------------------------------------------------------------------------
# 5. _is_retryable_download_error
# ---------------------------------------------------------------------------


class TestIsRetryableDownloadError:
    """Tests for classification of retryable vs non-retryable download errors."""

    def test_permission_and_value_errors_are_non_retryable(self):
        assert _is_retryable_download_error(PermissionError("blocked")) is False
        assert _is_retryable_download_error(ValueError("too large")) is False

    def test_http_4xx_statuses_are_non_retryable_except_429(self):
        req = httpx.Request("GET", "https://example.com/img.png")
        for status in (400, 403, 404, 410, 422):
            resp = httpx.Response(status, request=req)
            err = httpx.HTTPStatusError("client error", request=req, response=resp)
            assert _is_retryable_download_error(err) is False

        resp_429 = httpx.Response(429, request=req)
        err_429 = httpx.HTTPStatusError("rate limited", request=req, response=resp_429)
        assert _is_retryable_download_error(err_429) is True

    def test_http_5xx_statuses_are_retryable(self):
        req = httpx.Request("GET", "https://example.com/img.png")
        for status in (500, 502, 503, 504):
            resp = httpx.Response(status, request=req)
            err = httpx.HTTPStatusError("server error", request=req, response=resp)
            assert _is_retryable_download_error(err) is True

    def test_network_and_transport_errors_are_retryable(self):
        assert _is_retryable_download_error(httpx.ConnectError("down")) is True
        assert _is_retryable_download_error(httpx.ReadTimeout("timeout")) is True
        assert _is_retryable_download_error(TimeoutError("timeout")) is True
        assert _is_retryable_download_error(RuntimeError("generic glitch")) is True


# ---------------------------------------------------------------------------
# 6. _stream_download_to_file
# ---------------------------------------------------------------------------


class TestStreamDownloadToFile:
    """Tests for streaming download with byte-budget and SSRF guards."""

    @pytest.mark.asyncio
    async def test_happy_path_writes_destination_file(self, tmp_path):
        dest = tmp_path / "downloaded.png"

        class AsyncIterBytes:
            def __init__(self, chunks):
                self.chunks = list(chunks)
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self.chunks:
                    raise StopAsyncIteration
                return self.chunks.pop(0)

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.headers = {"content-length": "100"}
        fake_resp.url = "https://example.com/download.png"
        fake_resp.aiter_bytes = MagicMock(return_value=AsyncIterBytes([b"header", b"", b"data"]))

        fake_client = MagicMock()
        @contextlib.asynccontextmanager
        async def fake_stream(method, url, headers=None):
            yield fake_resp

        fake_client.stream = fake_stream

        with patch("tools.vision_tools.check_website_access", return_value=None):
            res = await _stream_download_to_file(
                fake_client, "https://example.com/download.png", dest, max_bytes=1000, headers={}
            )
            assert res == dest
            assert dest.read_bytes() == b"headerdata"

    @pytest.mark.asyncio
    async def test_content_length_early_rejection(self, tmp_path):
        dest = tmp_path / "toolarge.png"
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.headers = {"content-length": "50000"}
        fake_resp.url = "https://example.com/large.png"

        fake_client = MagicMock()
        @contextlib.asynccontextmanager
        async def fake_stream(method, url, headers=None):
            yield fake_resp

        fake_client.stream = fake_stream

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            pytest.raises(ValueError, match="too large"),
        ):
            await _stream_download_to_file(
                fake_client, "https://example.com/large.png", dest, max_bytes=1000, headers={}
            )

    @pytest.mark.asyncio
    async def test_invalid_content_length_continues_streaming(self, tmp_path):
        dest = tmp_path / "valid.png"

        class AsyncIterBytes:
            def __init__(self, chunks):
                self.chunks = list(chunks)
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self.chunks:
                    raise StopAsyncIteration
                return self.chunks.pop(0)

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.headers = {"content-length": "not-an-int"}
        fake_resp.url = "https://example.com/img.png"
        fake_resp.aiter_bytes = MagicMock(return_value=AsyncIterBytes([b"chunk1", b"chunk2"]))

        fake_client = MagicMock()
        @contextlib.asynccontextmanager
        async def fake_stream(method, url, headers=None):
            yield fake_resp

        fake_client.stream = fake_stream

        with patch("tools.vision_tools.check_website_access", return_value=None):
            res = await _stream_download_to_file(
                fake_client, "https://example.com/img.png", dest, max_bytes=1000, headers={}
            )
            assert res.read_bytes() == b"chunk1chunk2"

    @pytest.mark.asyncio
    async def test_policy_blocked_raises_permission_error(self, tmp_path):
        dest = tmp_path / "blocked.png"
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.headers = {}
        fake_resp.url = "https://blocked.example.com/img.png"

        fake_client = MagicMock()
        @contextlib.asynccontextmanager
        async def fake_stream(method, url, headers=None):
            yield fake_resp

        fake_client.stream = fake_stream

        with (
            patch("tools.vision_tools.check_website_access", return_value={"message": "Domain blocked by policy"}),
            pytest.raises(PermissionError, match="Domain blocked by policy"),
        ):
            await _stream_download_to_file(
                fake_client, "https://blocked.example.com/img.png", dest, max_bytes=1000, headers={}
            )

    @pytest.mark.asyncio
    async def test_streaming_exceeding_max_bytes_removes_temp_file(self, tmp_path):
        dest = tmp_path / "overflow.png"

        class AsyncIterBytes:
            def __init__(self, chunks):
                self.chunks = list(chunks)
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self.chunks:
                    raise StopAsyncIteration
                return self.chunks.pop(0)

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.headers = {}
        fake_resp.url = "https://example.com/overflow.png"
        fake_resp.aiter_bytes = MagicMock(return_value=AsyncIterBytes([b"x" * 60, b"y" * 60]))

        fake_client = MagicMock()
        @contextlib.asynccontextmanager
        async def fake_stream(method, url, headers=None):
            yield fake_resp

        fake_client.stream = fake_stream

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            pytest.raises(ValueError, match="too large"),
        ):
            await _stream_download_to_file(
                fake_client, "https://example.com/overflow.png", dest, max_bytes=100, headers={}
            )

        assert not dest.exists()
        assert len(list(tmp_path.glob("*.tmp"))) == 0

    @pytest.mark.asyncio
    async def test_temp_file_cleanup_oserror_handled(self, tmp_path):
        dest = tmp_path / "oserror.png"

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.headers = {}
        fake_resp.url = "https://example.com/img.png"
        fake_resp.aiter_bytes = MagicMock(side_effect=RuntimeError("stream broken"))

        fake_client = MagicMock()
        @contextlib.asynccontextmanager
        async def fake_stream(method, url, headers=None):
            yield fake_resp

        fake_client.stream = fake_stream

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch.object(Path, "unlink", side_effect=OSError("disk failure")),
            pytest.raises(RuntimeError, match="stream broken"),
        ):
            await _stream_download_to_file(
                fake_client, "https://example.com/img.png", dest, max_bytes=100, headers={}
            )


# ---------------------------------------------------------------------------
# 7. _download_image
# ---------------------------------------------------------------------------


class TestDownloadImage:
    """Tests for image downloading with retries, SSRF check, and backoff."""

    @pytest.mark.asyncio
    async def test_check_website_access_blocked_raises_immediately(self, tmp_path):
        dest = tmp_path / "blocked.png"
        with (
            patch("tools.vision_tools.check_website_access", return_value={"message": "Site blocked"}),
            pytest.raises(PermissionError, match="Site blocked"),
        ):
            await _download_image("https://blocked.com/pic.png", dest, max_retries=3)

    @pytest.mark.asyncio
    async def test_ssrf_redirect_guard_blocks_private_target(self, tmp_path):
        dest = tmp_path / "redirect_blocked.png"
        mock_response = MagicMock()

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.url_safety.redirect_target_from_response", return_value="http://169.254.169.254/secret"),
            patch("tools.url_safety.async_is_safe_url", new=AsyncMock(return_value=False)),
        ):
            mock_client = MagicMock()
            @contextlib.asynccontextmanager
            async def fake_client_cm(*args, **kwargs):
                hooks = kwargs.get("event_hooks", {}).get("response", [])
                for h in hooks:
                    await h(mock_response)
                yield mock_client

            with (
                patch("tools.url_safety.create_ssrf_safe_async_client", side_effect=fake_client_cm),
                pytest.raises(ValueError, match="Blocked redirect to private/internal address"),
            ):
                await _download_image("https://example.com/redirect", dest, max_retries=1)

    @pytest.mark.asyncio
    async def test_retry_on_transient_error_then_succeeds(self, tmp_path):
        dest = tmp_path / "img.png"
        attempts = 0

        @contextlib.asynccontextmanager
        async def fake_client_cm(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("Connection reset")
            dest.write_bytes(VALID_PNG)
            yield MagicMock()

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.url_safety.create_ssrf_safe_async_client", side_effect=fake_client_cm),
            patch("tools.vision_tools._stream_download_to_file", new=AsyncMock(return_value=dest)),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            res = await _download_image("https://example.com/img.png", dest, max_retries=3)
            assert res == dest
            assert attempts == 2
            assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self, tmp_path):
        dest = tmp_path / "img.png"
        req = httpx.Request("GET", "https://example.com/img.png")
        err_404 = httpx.HTTPStatusError("Not Found", request=req, response=httpx.Response(404, request=req))

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.url_safety.create_ssrf_safe_async_client", side_effect=err_404),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await _download_image("https://example.com/img.png", dest, max_retries=3)

    @pytest.mark.asyncio
    async def test_max_retries_zero_or_negative_raises_runtime_error(self, tmp_path):
        dest = tmp_path / "img.png"
        with pytest.raises(RuntimeError, match="exited retry loop without attempting"):
            await _download_image("https://example.com/img.png", dest, max_retries=0)


# ---------------------------------------------------------------------------
# 7.5. _crop_image_region
# ---------------------------------------------------------------------------


class TestCropImageRegion:
    """Tests for image cropping and region bounds clamping."""

    def test_pillow_missing_import_error(self, tmp_path):
        img_path = tmp_path / "img.png"
        img_path.write_bytes(VALID_PNG)
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            path, mime, err = _crop_image_region(img_path, [0, 0, 1, 1])
            assert path is None
            assert "requires Pillow" in err

    def test_non_standard_image_mode_converts_to_rgb(self, tmp_path):
        img_path = tmp_path / "cmyk.jpg"
        img_path.write_bytes(_create_image_bytes("JPEG", size=(20, 20), mode="CMYK"))
        path, mime, err = _crop_image_region(img_path, [0, 0, 10, 10])
        assert path is not None
        assert mime == "image/png"
        assert err is None
        path.unlink(missing_ok=True)

    def test_open_exception_returns_error(self, tmp_path):
        img_path = tmp_path / "img.png"
        img_path.write_bytes(VALID_PNG)
        with patch("PIL.Image.open", side_effect=Exception("corrupt file")):
            path, mime, err = _crop_image_region(img_path, [0, 0, 1, 1])
            assert path is None
            assert "Failed to crop region" in err


# ---------------------------------------------------------------------------
# 8. _resize_image_for_vision full ladder & helpers
# ---------------------------------------------------------------------------


class TestResizeImageForVision:
    """Tests for progressive downscaling and format adjustments."""

    def test_pillow_open_fails_when_resize_needed(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(_create_image_bytes("PNG", size=(100, 100)))

        with patch("PIL.Image.open", side_effect=Exception("cannot open")):
            data_url = _resize_image_for_vision(img_path, max_base64_bytes=10)
            assert data_url.startswith("data:image/png;base64,")

    def test_dimensions_exceed_max_dimension_triggers_downscale(self, tmp_path):
        img_path = tmp_path / "wide.png"
        img_path.write_bytes(_create_image_bytes("PNG", size=(3000, 2000)))

        scale_out = {}
        data_url = _resize_image_for_vision(
            img_path,
            mime_type="image/png",
            max_dimension=1568,
            scale_out=scale_out,
        )
        assert data_url.startswith("data:image/png;base64,")
        assert "orig_width" in scale_out
        assert scale_out["orig_width"] == 3000
        assert scale_out["new_width"] <= 1568

    def test_force_jpeg_reencodes_rgba_to_rgb_jpeg(self, tmp_path):
        img_path = tmp_path / "alpha.png"
        img_path.write_bytes(_create_image_bytes("PNG", size=(2000, 2000), mode="RGBA", color="blue"))

        scale_out = {}
        data_url = _resize_image_for_vision(
            img_path,
            mime_type="image/png",
            max_base64_bytes=1000,
            force_jpeg=True,
            scale_out=scale_out,
        )
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_pillow_quick_open_failure_handled(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(VALID_PNG)

        with patch("PIL.Image.open", side_effect=Exception("quick open boom")):
            data_url = _resize_image_for_vision(img_path, max_dimension=100)
            assert data_url.startswith("data:image/png;base64,")

    def test_tiny_image_floor_reaches_64(self, tmp_path):
        img_path = tmp_path / "tall.png"
        img_path.write_bytes(_create_image_bytes("PNG", size=(1000, 10)))

        scale_out = {}
        data_url = _resize_image_for_vision(
            img_path,
            max_base64_bytes=10,
            scale_out=scale_out,
        )
        assert data_url.startswith("data:")
        assert "new_height" in scale_out
        assert scale_out["new_height"] >= 64

    def test_small_image_under_limits_returns_direct(self, tmp_path):
        img_path = tmp_path / "tiny.png"
        img_path.write_bytes(_create_image_bytes("PNG", size=(10, 10)))

        data_url = _resize_image_for_vision(img_path, max_base64_bytes=1024 * 1024, max_dimension=100)
        assert data_url.startswith("data:image/png;base64,")

    def test_pillow_missing_import_error_falls_back(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(VALID_PNG)

        with (
            patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}),
            patch("tools.lazy_deps.ensure", side_effect=Exception("no pip")),
        ):
            data_url = _resize_image_for_vision(img_path, max_base64_bytes=1)
            assert data_url.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# 9. _supports_media_in_tool_results
# ---------------------------------------------------------------------------


class TestSupportsMediaInToolResults:
    """Tests for the provider + model capability table."""

    def test_empty_and_non_string_returns_false(self):
        assert _supports_media_in_tool_results(None, "gpt-4") is False
        assert _supports_media_in_tool_results("", "gpt-4") is False
        assert _supports_media_in_tool_results("   ", "gpt-4") is False
        assert _supports_media_in_tool_results(123, "gpt-4") is False  # type: ignore

    def test_aggregators_return_true(self):
        for provider in ("openrouter", "nous", "vertex", "bedrock", "anthropic-vertex", "google-vertex"):
            assert _supports_media_in_tool_results(provider, "any-model") is True

    def test_anthropic_variants_return_true(self):
        for provider in ("anthropic", "claude", "anthropic-direct"):
            assert _supports_media_in_tool_results(provider, "claude-3-opus") is True

    def test_openai_variants_return_true(self):
        for provider in ("openai", "openai-chat", "openai-codex", "azure-openai"):
            assert _supports_media_in_tool_results(provider, "gpt-4o") is True

    def test_gemini_models_support_table(self):
        assert _supports_media_in_tool_results("google", "gemini-3-pro") is True
        assert _supports_media_in_tool_results("gemini", "gemini-pro-3-preview") is True
        assert _supports_media_in_tool_results("google-gemini", "gemini-flash-3") is True
        assert _supports_media_in_tool_results("google-vertex-gemini", "gemini-2.5-pro") is False
        assert _supports_media_in_tool_results("google", None) is False  # type: ignore
        assert _supports_media_in_tool_results("google", 123) is False  # type: ignore

    def test_provider_profile_supports_vision_lookup(self):
        profile_mock = SimpleNamespace(supports_vision=True)
        with patch("providers.get_provider_profile", return_value=profile_mock):
            assert _supports_media_in_tool_results("custom-vision-provider", "model-v1") is True

        profile_no_vision = SimpleNamespace(supports_vision=False)
        with patch("providers.get_provider_profile", return_value=profile_no_vision):
            assert _supports_media_in_tool_results("custom-text-provider", "model-v1") is False

        with patch("providers.get_provider_profile", side_effect=Exception("not found")):
            assert _supports_media_in_tool_results("broken-provider", "model-v1") is False

    def test_unknown_provider_returns_false(self):
        assert _supports_media_in_tool_results("totally-unheard-of-provider-xyz", "model-v1") is False


# ---------------------------------------------------------------------------
# 10. _should_use_native_vision_fast_path
# ---------------------------------------------------------------------------


class TestShouldUseNativeVisionFastPath:
    """Tests for native vision fast path routing logic."""

    def test_native_mode_with_supported_media_returns_true(self):
        with (
            patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"),
            patch("agent.auxiliary_client._read_main_model", return_value="claude-3-5-sonnet"),
            patch("agent.image_routing.decide_image_input_mode", return_value="native"),
            patch("agent.image_routing._lookup_supports_vision", return_value=False),
            patch("hermes_cli.config.load_config", return_value={}),
        ):
            assert _should_use_native_vision_fast_path() is True

    def test_native_mode_with_override_returns_true(self):
        with (
            patch("agent.auxiliary_client._read_main_provider", return_value="custom-local"),
            patch("agent.auxiliary_client._read_main_model", return_value="llama-vision"),
            patch("agent.image_routing.decide_image_input_mode", return_value="native"),
            patch("agent.image_routing._lookup_supports_vision", return_value=True),
            patch("hermes_cli.config.load_config", return_value={}),
        ):
            assert _should_use_native_vision_fast_path() is True

    def test_legacy_mode_returns_false(self):
        with (
            patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"),
            patch("agent.auxiliary_client._read_main_model", return_value="claude-3"),
            patch("agent.image_routing.decide_image_input_mode", return_value="legacy"),
            patch("hermes_cli.config.load_config", return_value={}),
        ):
            assert _should_use_native_vision_fast_path() is False

    def test_import_or_resolution_exception_returns_false(self):
        with patch("agent.auxiliary_client._read_main_provider", side_effect=Exception("config corrupted")):
            assert _should_use_native_vision_fast_path() is False


# ---------------------------------------------------------------------------
# 11. _build_native_vision_tool_result & scale note
# ---------------------------------------------------------------------------


class TestBuildNativeVisionToolResult:
    """Tests for envelope structure and scale disclosure note creation."""

    def test_full_envelope_structure(self):
        long_url = "https://example.com/images/" + ("a" * 300)
        res = _build_native_vision_tool_result(
            image_url=long_url,
            question="What is this?",
            image_data_url="data:image/png;base64,ABC",
            image_size_bytes=2048,
            scale_note="Image downscaled 2x",
        )
        assert res["_multimodal"] is True
        assert len(res["content"]) == 2
        assert res["content"][0]["type"] == "text"
        assert "What is this?" in res["content"][0]["text"]
        assert "Note: Image downscaled 2x" in res["content"][0]["text"]
        assert res["content"][1]["image_url"]["url"] == "data:image/png;base64,ABC"
        assert "2.0 KB" in res["text_summary"]
        assert len(res["meta"]["image_url"]) == 200
        assert res["meta"]["size_bytes"] == 2048
        assert res["meta"]["native_vision"] is True

    def test_empty_question_omits_question_block(self):
        res = _build_native_vision_tool_result(
            image_url="https://example.com/pic.png",
            question="",
            image_data_url="data:image/png;base64,ABC",
            image_size_bytes=1000,
        )
        assert "Question:" not in res["content"][0]["text"]

    def test_build_scale_note_both_uniform_and_non_uniform(self):
        uniform = _build_scale_note(
            {"orig_width": 2000, "orig_height": 1000, "new_width": 1000, "new_height": 500},
            None,
        )
        assert "multiply any coordinates you report by 2.00" in uniform

        non_uniform = _build_scale_note(
            {"orig_width": 3000, "orig_height": 1000, "new_width": 1000, "new_height": 500},
            None,
        )
        assert "multiply any x coordinates you report by 3.00 and any y coordinates by 2.00" in non_uniform

        crop_only = _build_scale_note(
            None,
            {"x": 10, "y": 20, "width": 100, "height": 100},
        )
        assert "Analysis was performed on a cropped region" in crop_only
        assert "offset (10, 20)" in crop_only

        assert _build_scale_note(None, None) is None


# ---------------------------------------------------------------------------
# 12. _vision_analyze_native
# ---------------------------------------------------------------------------


class TestVisionAnalyzeNative:
    """Tests for native vision fast-path pipeline."""

    @pytest.mark.asyncio
    async def test_empty_image_url_returns_error(self):
        for empty in ("", "   ", None):
            res = await _vision_analyze_native(empty, "question")  # type: ignore
            data = json.loads(res) if isinstance(res, str) else res
            assert data["success"] is False
            assert "image_url is required" in data["error"]

    @pytest.mark.asyncio
    async def test_interrupted_returns_error(self):
        with patch("tools.interrupt.is_interrupted", return_value=True):
            res = await _vision_analyze_native("https://example.com/a.png", "question")
            data = json.loads(res) if isinstance(res, str) else res
            assert data["success"] is False
            assert "Interrupted" in data["error"]

    @pytest.mark.asyncio
    async def test_resolution_error_returns_error(self):
        from tools.image_source import ImageResolutionError

        with patch("tools.image_source.resolve_image_source", side_effect=ImageResolutionError("file not found")):
            res = await _vision_analyze_native("missing.png", "question")
            data = json.loads(res) if isinstance(res, str) else res
            assert data["success"] is False
            assert "file not found" in data["error"]

    @pytest.mark.asyncio
    async def test_normalization_failure_returns_error(self):
        fake_resolved = SimpleNamespace(data=b"<svg></svg>", mime="image/svg+xml")
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._normalize_to_supported_image", return_value=(None, None, "SVG conversion failed")),
        ):
            res = await _vision_analyze_native("test.svg", "question")
            data = json.loads(res) if isinstance(res, str) else res
            assert data["success"] is False
            assert "SVG conversion failed" in data["error"]

    @pytest.mark.asyncio
    async def test_region_crop_success_and_failure(self, tmp_path):
        img_bytes = _create_image_bytes("PNG", size=(100, 100))
        fake_resolved = SimpleNamespace(data=img_bytes, mime="image/png")

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
        ):
            res = await _vision_analyze_native("test.png", "question", region=[10, 10, 50, 50])
            assert isinstance(res, dict)
            assert res.get("_multimodal") is True
            assert "Analysis was performed on a cropped region" in res["content"][0]["text"]

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
        ):
            res_fail = await _vision_analyze_native("test.png", "question", region=[50, 50, 50, 50])
            data = json.loads(res_fail) if isinstance(res_fail, str) else res_fail
            assert data["success"] is False
            assert "crops to zero area" in data["error"]

    @pytest.mark.asyncio
    async def test_oversize_embed_cap_triggers_downscale(self, monkeypatch):
        img_bytes = _create_image_bytes("PNG", size=(100, 100))
        fake_resolved = SimpleNamespace(data=img_bytes, mime="image/png")

        monkeypatch.setattr("tools.vision_tools._EMBED_TARGET_BYTES", 50)
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
        ):
            res = await _vision_analyze_native("test.png", "question")
            assert isinstance(res, dict)
            assert res.get("_multimodal") is True

    @pytest.mark.asyncio
    async def test_exceeding_max_base64_bytes_returns_error(self, monkeypatch):
        img_bytes = _create_image_bytes("PNG", size=(100, 100))
        fake_resolved = SimpleNamespace(data=img_bytes, mime="image/png")

        monkeypatch.setattr("tools.vision_tools._EMBED_TARGET_BYTES", 50)
        monkeypatch.setattr("tools.vision_tools._MAX_BASE64_BYTES", 10)
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
        ):
            res = await _vision_analyze_native("test.png", "question")
            data = json.loads(res) if isinstance(res, str) else res
            assert data["success"] is False
            assert "Image too large for vision API" in data["error"]

    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_tool_error(self):
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", side_effect=Exception("catastrophic failure")),
        ):
            res = await _vision_analyze_native("test.png", "question")
            data = json.loads(res) if isinstance(res, str) else res
            assert data["success"] is False
            assert "Native vision failed" in data["error"]

    @pytest.mark.asyncio
    async def test_bmp_normalization_swaps_path(self, tmp_path):
        bmp_bytes = _create_image_bytes("BMP", size=(10, 10))
        fake_resolved = SimpleNamespace(data=bmp_bytes, mime="image/bmp")

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
        ):
            res = await _vision_analyze_native("test.bmp", "question")
            assert isinstance(res, dict)
            assert res.get("_multimodal") is True

    @pytest.mark.asyncio
    async def test_region_crop_and_finally_unlink_exceptions_handled(self, tmp_path):
        img_bytes = _create_image_bytes("PNG", size=(20, 20))
        fake_resolved = SimpleNamespace(data=img_bytes, mime="image/png")

        def failing_unlink(self, *args, **kwargs):
            raise PermissionError("cannot delete")

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch.object(Path, "unlink", failing_unlink),
        ):
            res = await _vision_analyze_native("test.png", "question", region=[0, 0, 10, 10])
            assert isinstance(res, dict)
            assert res.get("_multimodal") is True


# ---------------------------------------------------------------------------
# 13. vision_analyze_tool legacy path
# ---------------------------------------------------------------------------


class TestVisionAnalyzeToolLegacy:
    """Tests for the legacy auxiliary LLM vision tool."""

    @pytest.mark.asyncio
    async def test_non_string_user_prompt_handled(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        mock_resp = {"choices": [{"message": {"content": "an image of something"}}]}

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="an image of something"),
        ):
            raw = await vision_analyze_tool("https://example.com/img.png", None)  # type: ignore
            res = json.loads(raw)
            assert res["success"] is True
            assert res["analysis"] == "an image of something"

    @pytest.mark.asyncio
    async def test_interrupted_returns_tool_error(self):
        with patch("tools.interrupt.is_interrupted", return_value=True):
            raw = await vision_analyze_tool("https://example.com/img.png", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Interrupted" in res["error"]

    @pytest.mark.asyncio
    async def test_resolution_error_handled(self):
        from tools.image_source import ImageResolutionError

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", side_effect=ImageResolutionError("not found")),
        ):
            raw = await vision_analyze_tool("https://example.com/missing.png", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "not found" in res["error"]

    @pytest.mark.asyncio
    async def test_normalization_failure_handled(self):
        fake_resolved = SimpleNamespace(data=b"<svg></svg>", mime="image/svg+xml")
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._normalize_to_supported_image", return_value=(None, None, "SVG conversion error")),
        ):
            raw = await vision_analyze_tool("test.svg", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "SVG conversion error" in res["error"]

    @pytest.mark.asyncio
    async def test_oversize_base64_exceeds_max_raises(self, monkeypatch):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        monkeypatch.setattr("tools.vision_tools._MAX_BASE64_BYTES", 10)

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
        ):
            raw = await vision_analyze_tool("test.png", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Image too large for vision API" in res["error"]

    @pytest.mark.asyncio
    async def test_api_size_error_triggers_resize_and_retry(self, monkeypatch):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        mock_resp = {"choices": [{"message": {"content": "analyzed after resize"}}]}

        monkeypatch.setattr("tools.vision_tools._RESIZE_TARGET_BYTES", 10)

        call_count = 0
        async def fake_call_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("413 Request Entity Too Large: image_url too big")
            return mock_resp

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=fake_call_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="analyzed after resize"),
        ):
            raw = await vision_analyze_tool("test.png", "describe")
            res = json.loads(raw)
            assert res["success"] is True
            assert res["analysis"] == "analyzed after resize"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_empty_analysis_retries_once(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        call_count = 0

        async def fake_call_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"attempt": call_count}

        def fake_extract(resp):
            if resp.get("attempt") == 1:
                return ""
            return "recovered analysis"

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=fake_call_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", side_effect=fake_extract),
        ):
            raw = await vision_analyze_tool("test.png", "describe")
            res = json.loads(raw)
            assert res["success"] is True
            assert res["analysis"] == "recovered analysis"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_error_classification_billing(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("402 payment required - insufficient credits")),
        ):
            raw = await vision_analyze_tool("test.png", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Insufficient credits" in res["analysis"]

    @pytest.mark.asyncio
    async def test_error_classification_unsupported_vision(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("model does not support image input")),
        ):
            raw = await vision_analyze_tool("test.png", "describe", model="text-only-model")
            res = json.loads(raw)
            assert res["success"] is False
            assert "does not support vision" in res["analysis"]

    @pytest.mark.asyncio
    async def test_error_classification_invalid_request(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("invalid_request: image_url corrupt")),
        ):
            raw = await vision_analyze_tool("test.png", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "rejected the image" in res["analysis"]

    @pytest.mark.asyncio
    async def test_error_classification_generic(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("server crashed")),
        ):
            raw = await vision_analyze_tool("test.png", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "There was a problem" in res["analysis"]

    @pytest.mark.asyncio
    async def test_bmp_normalization_swaps_path(self):
        bmp_bytes = _create_image_bytes("BMP", size=(10, 10))
        fake_resolved = SimpleNamespace(data=bmp_bytes, mime="image/bmp")
        mock_resp = {"choices": [{"message": {"content": "ok"}}]}

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="ok"),
        ):
            raw = await vision_analyze_tool("test.bmp", "describe")
            res = json.loads(raw)
            assert res["success"] is True

    @pytest.mark.asyncio
    async def test_region_crop_failure_raises(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
        ):
            raw = await vision_analyze_tool("test.png", "describe", region=[50, 50, 50, 50])
            res = json.loads(raw)
            assert res["success"] is False
            assert "crops to zero area" in res["error"]

    @pytest.mark.asyncio
    async def test_config_reading_exception_handled(self):
        fake_resolved = SimpleNamespace(data=VALID_PNG, mime="image/png")
        mock_resp = {"choices": [{"message": {"content": "ok"}}]}

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)),
            patch("hermes_cli.config.load_config", side_effect=Exception("corrupt yaml")),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="ok"),
        ):
            raw = await vision_analyze_tool("test.png", "describe")
            res = json.loads(raw)
            assert res["success"] is True


# ---------------------------------------------------------------------------
# 14. check_vision_requirements
# ---------------------------------------------------------------------------


class TestCheckVisionRequirements:
    """Tests for vision requirement gating and fallback chain."""

    def test_explicit_provider_client_resolves_returns_true(self):
        mock_client = MagicMock()
        with (
            patch("agent.auxiliary_client.resolve_vision_provider_client", return_value=("anthropic", mock_client, "claude-3")),
            patch("agent.auxiliary_client.aux_probe_mode"),
        ):
            assert check_vision_requirements() is True

    def test_explicit_provider_fails_auto_fallback_succeeds(self):
        mock_client = MagicMock()
        def fake_resolve(provider=None):
            if provider == "auto":
                return ("openrouter", mock_client, "gemini-flash")
            return (None, None, None)

        with (
            patch("agent.auxiliary_client.resolve_vision_provider_client", side_effect=fake_resolve),
            patch("agent.auxiliary_client.aux_probe_mode"),
        ):
            assert check_vision_requirements() is True

    def test_both_explicit_and_auto_fail_returns_false(self):
        with (
            patch("agent.auxiliary_client.resolve_vision_provider_client", return_value=(None, None, None)),
            patch("agent.auxiliary_client.aux_probe_mode"),
        ):
            assert check_vision_requirements() is False

    def test_import_error_returns_false(self):
        with patch.dict("sys.modules", {"agent.auxiliary_client": None}):
            assert check_vision_requirements() is False

    def test_exception_in_probe_returns_false(self):
        with patch("agent.auxiliary_client.resolve_vision_provider_client", side_effect=Exception("config failure")):
            assert check_vision_requirements() is False


# ---------------------------------------------------------------------------
# 15. Video helpers & video_analyze_tool
# ---------------------------------------------------------------------------


class TestVideoHelpers:
    """Tests for video MIME sniff, terminal backend resolution, and downloads."""

    def test_detect_video_mime_type(self):
        assert _detect_video_mime_type(Path("clip.mp4")) == "video/mp4"
        assert _detect_video_mime_type(Path("clip.webm")) == "video/webm"
        assert _detect_video_mime_type(Path("clip.mov")) == "video/mov"
        assert _detect_video_mime_type(Path("clip.avi")) == "video/mp4"
        assert _detect_video_mime_type(Path("clip.mkv")) == "video/mp4"
        assert _detect_video_mime_type(Path("clip.mpeg")) == "video/mpeg"
        assert _detect_video_mime_type(Path("clip.mpg")) == "video/mpeg"
        assert _detect_video_mime_type(Path("clip.xyz")) is None

    def test_video_to_base64_data_url(self, tmp_path):
        vid_file = tmp_path / "test.mp4"
        vid_file.write_bytes(b"fake-video-bytes")

        url = _video_to_base64_data_url(vid_file)
        assert url.startswith("data:video/mp4;base64,")

        custom_url = _video_to_base64_data_url(vid_file, mime_type="video/webm")
        assert custom_url.startswith("data:video/webm;base64,")

    def test_terminal_backend_is_local(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert _terminal_backend_is_local() is True
        monkeypatch.setenv("TERMINAL_ENV", "")
        assert _terminal_backend_is_local() is True
        monkeypatch.setenv("TERMINAL_ENV", "container")
        assert _terminal_backend_is_local() is False

    def test_is_path_like_video_source(self):
        assert _is_path_like_video_source("/path/to/vid.mp4") is True
        assert _is_path_like_video_source("relative/vid.mp4") is True
        assert _is_path_like_video_source("file:///path/vid.mp4") is True
        assert _is_path_like_video_source("https://example.com/vid.mp4") is False
        assert _is_path_like_video_source("http://example.com/vid.mp4") is False
        assert _is_path_like_video_source("data:video/mp4;base64,ABC") is False
        assert _is_path_like_video_source("") is False
        assert _is_path_like_video_source(None) is False  # type: ignore

    @pytest.mark.asyncio
    async def test_materialize_video_from_terminal_backend_success(self):
        fake_resolved = SimpleNamespace(data=b"fake-video-payload")
        with patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=fake_resolved)):
            path = await _materialize_video_from_terminal_backend("file:///tmp/sample.mp4", task_id="task-1")
            try:
                assert path.exists()
                assert path.suffix == ".mp4"
                assert path.read_bytes() == b"fake-video-payload"
            finally:
                path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_materialize_video_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported video format"):
            await _materialize_video_from_terminal_backend("clip.xyz", task_id=None)

    @pytest.mark.asyncio
    async def test_materialize_video_resolution_error_raises(self):
        from tools.image_source import ImageResolutionError

        with (
            patch("tools.image_source.resolve_image_source", side_effect=ImageResolutionError("not in container")),
            pytest.raises(ValueError, match="Could not read video from terminal backend"),
        ):
            await _materialize_video_from_terminal_backend("clip.mp4", task_id="task-1")

    @pytest.mark.asyncio
    async def test_download_video_policy_blocked(self, tmp_path):
        dest = tmp_path / "vid.mp4"
        with (
            patch("tools.vision_tools.check_website_access", return_value={"message": "Video URL blocked"}),
            pytest.raises(PermissionError, match="Video URL blocked"),
        ):
            await _download_video("https://blocked.com/v.mp4", dest, max_retries=1)

    @pytest.mark.asyncio
    async def test_download_video_ssrf_redirect_blocked(self, tmp_path):
        dest = tmp_path / "redirect_vid.mp4"
        mock_resp = MagicMock()

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.url_safety.redirect_target_from_response", return_value="http://10.0.0.1/video.mp4"),
            patch("tools.url_safety.async_is_safe_url", new=AsyncMock(return_value=False)),
        ):
            mock_client = MagicMock()
            @contextlib.asynccontextmanager
            async def fake_client_cm(*args, **kwargs):
                hooks = kwargs.get("event_hooks", {}).get("response", [])
                for h in hooks:
                    await h(mock_resp)
                yield mock_client

            with (
                patch("tools.url_safety.create_ssrf_safe_async_client", side_effect=fake_client_cm),
                pytest.raises(ValueError, match="Blocked redirect to private/internal address"),
            ):
                await _download_video("https://example.com/v.mp4", dest, max_retries=1)

    @pytest.mark.asyncio
    async def test_download_video_success_and_retry(self, tmp_path):
        dest = tmp_path / "v.mp4"
        attempts = 0

        @contextlib.asynccontextmanager
        async def fake_client_cm(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("glitch")
            dest.write_bytes(b"videodata")
            yield MagicMock()

        with (
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.url_safety.create_ssrf_safe_async_client", side_effect=fake_client_cm),
            patch("tools.vision_tools._stream_download_to_file", new=AsyncMock(return_value=dest)),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            res = await _download_video("https://example.com/v.mp4", dest, max_retries=2)
            assert res == dest
            assert attempts == 2
            assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_download_video_zero_retries_raises_runtime_error(self, tmp_path):
        dest = tmp_path / "v.mp4"
        with pytest.raises(RuntimeError, match="exited retry loop without attempting"):
            await _download_video("https://example.com/v.mp4", dest, max_retries=0)


# ---------------------------------------------------------------------------
# 16. video_analyze_tool & _handle_video_analyze
# ---------------------------------------------------------------------------


class TestVideoAnalyzeTool:
    """Tests for the video analysis tool and handler."""

    @pytest.mark.asyncio
    async def test_interrupted_returns_error(self):
        with patch("tools.interrupt.is_interrupted", return_value=True):
            raw = await video_analyze_tool("https://example.com/v.mp4", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Interrupted" in res["error"]

    @pytest.mark.asyncio
    async def test_local_video_read_blocked(self, tmp_path):
        vid_path = tmp_path / "blocked.mp4"
        vid_path.write_bytes(b"videodata")

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked", side_effect=PermissionError("File read forbidden")),
        ):
            raw = await video_analyze_tool(str(vid_path), "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "File read forbidden" in res["error"]

    @pytest.mark.asyncio
    async def test_remote_video_url_download_and_analyze(self, tmp_path):
        mock_resp = {"choices": [{"message": {"content": "Video shows a flying bird."}}]}

        async def fake_download(url, dest, **kwargs):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"valid-video-bytes")
            return dest

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)),
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.vision_tools._download_video", side_effect=fake_download),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="Video shows a flying bird."),
        ):
            raw = await video_analyze_tool("https://example.com/bird.mp4", "describe", model="google/gemini-2.5-flash")
            res = json.loads(raw)
            assert res["success"] is True
            assert res["analysis"] == "Video shows a flying bird."

    @pytest.mark.asyncio
    async def test_remote_backend_path_like_video(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "container")
        mat_path = tmp_path / "container_video.mp4"
        mat_path.write_bytes(b"container-video-bytes")
        mock_resp = {"choices": [{"message": {"content": "Container video analysis."}}]}

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.vision_tools._materialize_video_from_terminal_backend", new=AsyncMock(return_value=mat_path)),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="Container video analysis."),
        ):
            raw = await video_analyze_tool("/sandbox/app/video.mp4", "describe")
            res = json.loads(raw)
            assert res["success"] is True
            assert res["analysis"] == "Container video analysis."

    @pytest.mark.asyncio
    async def test_invalid_video_source_raises_error(self):
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=False)),
        ):
            raw = await video_analyze_tool("not_a_real_file_or_url.mp4", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Invalid video source" in res["error"]

    @pytest.mark.asyncio
    async def test_unsupported_video_format_raises_error(self, tmp_path):
        bad_vid = tmp_path / "file.unsupported"
        bad_vid.write_bytes(b"some-data")

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
        ):
            raw = await video_analyze_tool(str(bad_vid), "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Unsupported video format" in res["error"]

    @pytest.mark.asyncio
    async def test_oversize_video_base64_raises_error(self, tmp_path, monkeypatch):
        vid_file = tmp_path / "large.mp4"
        vid_file.write_bytes(b"x" * 100)
        monkeypatch.setattr("tools.vision_tools._MAX_VIDEO_BASE64_BYTES", 10)

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Video too large for API" in res["error"]

    @pytest.mark.asyncio
    async def test_video_warning_size_logged(self, tmp_path, monkeypatch, caplog):
        vid_file = tmp_path / "warning_size.mp4"
        vid_file.write_bytes(b"x" * 50)
        monkeypatch.setattr("tools.vision_tools._VIDEO_SIZE_WARN_BYTES", 10)
        mock_resp = {"choices": [{"message": {"content": "ok"}}]}

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="ok"),
            caplog.at_level(logging.WARNING, logger="tools.vision_tools"),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert res["success"] is True
            assert any("may be slow or rejected" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_video_empty_response_retry(self, tmp_path):
        vid_file = tmp_path / "clip.mp4"
        vid_file.write_bytes(b"vid")
        call_count = 0

        async def fake_call_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"attempt": call_count}

        def fake_extract(resp):
            return "final video description" if resp.get("attempt") == 2 else ""

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=fake_call_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", side_effect=fake_extract),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert res["success"] is True
            assert res["analysis"] == "final video description"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_video_error_classifications(self, tmp_path):
        vid_file = tmp_path / "clip.mp4"
        vid_file.write_bytes(b"vid")

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("402 payment required")),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert "Insufficient credits" in res["analysis"]

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("does not support video input")),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert "does not support video analysis" in res["analysis"]

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("413 content_too_large")),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert "video is too large" in res["analysis"]

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", side_effect=Exception("unknown glitch")),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert "There was a problem" in res["analysis"]

    @pytest.mark.asyncio
    async def test_user_prompt_non_string_and_file_uri(self, tmp_path):
        vid_file = tmp_path / "file_uri.mp4"
        vid_file.write_bytes(b"vid")
        mock_resp = {"choices": [{"message": {"content": "ok"}}]}

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="ok"),
        ):
            raw = await video_analyze_tool(f"file://{vid_file}", None)  # type: ignore
            res = json.loads(raw)
            assert res["success"] is True

    @pytest.mark.asyncio
    async def test_remote_video_url_blocked_by_website_policy(self):
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)),
            patch("tools.vision_tools.check_website_access", return_value={"message": "Forbidden URL"}),
        ):
            raw = await video_analyze_tool("https://example.com/blocked.mp4", "describe")
            res = json.loads(raw)
            assert res["success"] is False
            assert "Forbidden URL" in res["error"]

    @pytest.mark.asyncio
    async def test_video_config_temperature_and_timeout(self, tmp_path):
        vid_file = tmp_path / "cfg.mp4"
        vid_file.write_bytes(b"vid")
        mock_resp = {"choices": [{"message": {"content": "ok"}}]}

        cfg = {"auxiliary": {"vision": {"timeout": 200.0, "temperature": 0.5}}}
        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("agent.file_safety.raise_if_read_blocked"),
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="ok"),
        ):
            raw = await video_analyze_tool(str(vid_file), "describe")
            res = json.loads(raw)
            assert res["success"] is True

    @pytest.mark.asyncio
    async def test_video_cleanup_failure_logged(self, tmp_path, caplog):
        mock_resp = {"choices": [{"message": {"content": "ok"}}]}

        async def fake_download(url, dest, **kwargs):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"vid")
            return dest

        def failing_unlink(self, *args, **kwargs):
            raise PermissionError("cannot unlink temp video")

        with (
            patch("tools.interrupt.is_interrupted", return_value=False),
            patch("tools.vision_tools._validate_image_url_async", new=AsyncMock(return_value=True)),
            patch("tools.vision_tools.check_website_access", return_value=None),
            patch("tools.vision_tools._download_video", side_effect=fake_download),
            patch("tools.vision_tools._load_auxiliary_client"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(return_value=mock_resp)),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="ok"),
            patch.object(Path, "unlink", failing_unlink),
            caplog.at_level(logging.WARNING, logger="tools.vision_tools"),
        ):
            raw = await video_analyze_tool("https://example.com/vid.mp4", "describe")
            res = json.loads(raw)
            assert res["success"] is True
            assert any("Could not delete temporary file" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_handle_video_analyze_model_resolution(self):
        with patch("tools.vision_tools.video_analyze_tool", new=AsyncMock(return_value=json.dumps({"success": True}))) as mock_tool:
            await _handle_video_analyze({"video_url": "https://example.com/v.mp4", "question": "describe this"})
            assert mock_tool.call_count == 1
            call_prompt = mock_tool.call_args[0][1]
            assert "describe this" in call_prompt

        cfg_video = {"auxiliary": {"video": {"model": "custom-video-model"}}}
        with (
            patch("hermes_cli.config.load_config", return_value=cfg_video),
            patch("tools.vision_tools.video_analyze_tool", new=AsyncMock(return_value="{}")) as mock_tool,
        ):
            await _handle_video_analyze({"video_url": "https://example.com/v.mp4", "question": "q"})
            assert mock_tool.call_args[0][2] == "custom-video-model"

        with (
            patch("hermes_cli.config.load_config", side_effect=Exception("broken")),
            patch.dict(os.environ, {"AUXILIARY_VIDEO_MODEL": "env-video-model"}),
            patch("tools.vision_tools.video_analyze_tool", new=AsyncMock(return_value="{}")) as mock_tool,
        ):
            await _handle_video_analyze({"video_url": "https://example.com/v.mp4", "question": "q"})
            assert mock_tool.call_args[0][2] == "env-video-model"


# ---------------------------------------------------------------------------
# 17. Misc small helpers & Schemas
# ---------------------------------------------------------------------------


class TestMiscAndConfiguration:
    """Tests for utility helpers, worker concurrency, timeouts, and schemas."""

    def test_resolve_download_timeout_env_and_config(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_DOWNLOAD_TIMEOUT", "45.0")
        assert _resolve_download_timeout() == 45.0

        monkeypatch.setenv("HERMES_VISION_DOWNLOAD_TIMEOUT", "invalid")
        with patch("hermes_cli.config.load_config", return_value={"auxiliary": {"vision": {"download_timeout": 55.0}}}):
            assert _resolve_download_timeout() == 55.0

        monkeypatch.delenv("HERMES_VISION_DOWNLOAD_TIMEOUT", raising=False)
        with patch("hermes_cli.config.load_config", side_effect=Exception("missing")):
            assert _resolve_download_timeout() == 30.0

    def test_detect_host_cpus(self):
        with patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True):
            assert _detect_host_cpus() == 4

        with (
            patch.object(os, "sched_getaffinity", side_effect=AttributeError, create=True),
            patch("os.cpu_count", return_value=8),
        ):
            assert _detect_host_cpus() == 8

        with (
            patch.object(os, "sched_getaffinity", side_effect=AttributeError, create=True),
            patch("os.cpu_count", return_value=None),
        ):
            assert _detect_host_cpus() == 1

    def test_resolve_vision_cpu_workers(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_MAX_CONCURRENCY", "4")
        assert _resolve_vision_cpu_workers() == 4

        monkeypatch.setenv("HERMES_VISION_MAX_CONCURRENCY", "0")
        with patch("hermes_cli.config.load_config", return_value={"auxiliary": {"vision": {"max_concurrency": 6}}}):
            assert _resolve_vision_cpu_workers() == 6

        monkeypatch.delenv("HERMES_VISION_MAX_CONCURRENCY", raising=False)
        with (
            patch("hermes_cli.config.load_config", side_effect=Exception("missing")),
            patch("tools.vision_tools._detect_host_cpus", return_value=2),
        ):
            assert _resolve_vision_cpu_workers() == 2

    def test_resolve_vision_cpu_workers_invalid_string(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_MAX_CONCURRENCY", "not-a-number")
        with patch("tools.vision_tools._detect_host_cpus", return_value=3):
            assert _resolve_vision_cpu_workers() == 3

    @pytest.mark.asyncio
    async def test_handle_vision_analyze_config_exception(self):
        with (
            patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False),
            patch("hermes_cli.config.load_config", side_effect=Exception("config boom")),
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "env-vision-model"}),
            patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value="{}")) as mock_tool,
        ):
            await _handle_vision_analyze({"image_url": "https://example.com/a.png", "question": "q"})
            assert mock_tool.call_args[0][2] == "env-vision-model"

    @pytest.mark.asyncio
    async def test_run_encode_on_cpu_executor(self):
        res = await _run_encode_on_cpu_executor(lambda x, y: x * y, 6, 7)
        assert res == 42

    def test_image_url_shape_ok(self):
        assert _image_url_shape_ok("http://example.com/img.png") is True
        assert _image_url_shape_ok("https://example.com/img.jpg") is True
        assert _image_url_shape_ok("http://") is False
        assert _image_url_shape_ok("https://") is False
        assert _image_url_shape_ok("ftp://example.com/img.png") is False
        assert _image_url_shape_ok("") is False
        assert _image_url_shape_ok(None) is False  # type: ignore

    @pytest.mark.asyncio
    async def test_validate_image_url_async(self):
        assert await _validate_image_url_async("not-a-url") is False
        with patch("tools.url_safety.async_is_safe_url", new=AsyncMock(return_value=True)):
            assert await _validate_image_url_async("https://example.com/pic.png") is True

    def test_is_image_size_error(self):
        assert _is_image_size_error(Exception("413 Request Entity Too Large")) is True
        assert _is_image_size_error(Exception("image_url too big")) is True
        assert _is_image_size_error(Exception("exceeds maximum allowed payload")) is True
        assert _is_image_size_error(Exception("database connection failed")) is False

    @pytest.mark.asyncio
    async def test_vision_concurrency_slot(self):
        entered = False
        async with _vision_concurrency_slot():
            entered = True
        assert entered is True

    def test_load_auxiliary_client_populates_globals(self):
        import tools.vision_tools as vt
        orig_call = vt.async_call_llm
        orig_extract = vt.extract_content_or_reasoning
        try:
            vt.async_call_llm = None
            vt.extract_content_or_reasoning = None
            _load_auxiliary_client()
            assert vt.async_call_llm is not None
            assert vt.extract_content_or_reasoning is not None
        finally:
            vt.async_call_llm = orig_call
            vt.extract_content_or_reasoning = orig_extract

    def test_schemas_validity(self):
        assert VISION_ANALYZE_SCHEMA["name"] == "vision_analyze"
        assert "image_url" in VISION_ANALYZE_SCHEMA["parameters"]["properties"]
        assert "question" in VISION_ANALYZE_SCHEMA["parameters"]["properties"]

        assert VIDEO_ANALYZE_SCHEMA["name"] == "video_analyze"
        assert "video_url" in VIDEO_ANALYZE_SCHEMA["parameters"]["properties"]
        assert "question" in VIDEO_ANALYZE_SCHEMA["parameters"]["properties"]
