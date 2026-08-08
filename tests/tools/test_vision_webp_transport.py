#!/usr/bin/env python3
"""WebP transport-compatibility tests — tools/ollama_vision_client.py.

Behavioral tests for the two-layer hash contract:
- canonical normalized_image_sha256 (calibration identity, WebP hash)
- transport_image_sha256 / transport_mime_type / transport_transcoded
  (exact bytes sent to the endpoint)

No network requests. Source files are never modified.
"""
import asyncio
import base64
import hashlib
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.ollama_vision_client import (  # noqa: E402
    _ensure_ollama_transport_compatible_image,
    prepare_image,
)
from tools.vision_tools import _detect_image_mime_type_from_bytes  # noqa: E402


def _make_webp_bytes(size=(8, 8), rgba=False):
    """Build deterministic in-memory WebP bytes (suffix irrelevant)."""
    from PIL import Image

    im = Image.new("RGBA" if rgba else "RGB", size, (200, 30, 30, 128) if rgba else (200, 30, 30))
    buf = io.BytesIO()
    im.save(buf, format="WEBP")
    return buf.getvalue()


def _make_png_bytes(size=(8, 8)):
    from PIL import Image

    im = Image.new("RGB", size, (10, 200, 10))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(size=(8, 8)):
    from PIL import Image

    im = Image.new("RGB", size, (10, 10, 200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


class TestMimeDetectionFromBytes:
    def test_webp_detected_by_bytes_despite_jpg_suffix(self, tmp_path):
        webp = _make_webp_bytes()
        p = tmp_path / "image.jpg"  # misleading suffix
        p.write_bytes(webp)
        assert _detect_image_mime_type_from_bytes(p.read_bytes()) == "image/webp"


class TestTransportConversion:
    def test_webp_converted_to_png_bytes(self, tmp_path):
        webp = _make_webp_bytes()
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        transport_bytes, mime, transcoded = _ensure_ollama_transport_compatible_image(p, "image/webp")
        assert transcoded is True
        assert mime == "image/png"
        # PNG magic bytes
        assert transport_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_data_url_prefix_png(self, tmp_path):
        from tools.vision_tools import _image_to_base64_data_url

        webp = _make_webp_bytes()
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        transport_bytes, mime, _ = _ensure_ollama_transport_compatible_image(p, "image/webp")
        tmp2 = tmp_path / "transport.png"
        tmp2.write_bytes(transport_bytes)
        url = _image_to_base64_data_url(tmp2, mime_type=mime)
        assert url.startswith("data:image/png;base64,")

    def test_png_transport_bytes_decode(self, tmp_path):
        from PIL import Image

        webp = _make_webp_bytes((16, 9))
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        transport_bytes, _, _ = _ensure_ollama_transport_compatible_image(p, "image/webp")
        im = Image.open(io.BytesIO(transport_bytes))
        assert im.format == "PNG"
        assert im.size == (16, 9)

    def test_dimensions_unchanged(self, tmp_path):
        webp = _make_webp_bytes((32, 24))
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        transport_bytes, _, _ = _ensure_ollama_transport_compatible_image(p, "image/webp")
        from PIL import Image

        im = Image.open(io.BytesIO(transport_bytes))
        assert im.size == (32, 24)

    def test_transparent_webp_safe(self, tmp_path):
        from PIL import Image

        webp = _make_webp_bytes((8, 8), rgba=True)
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        transport_bytes, mime, transcoded = _ensure_ollama_transport_compatible_image(p, "image/webp")
        assert transcoded is True
        im = Image.open(io.BytesIO(transport_bytes))
        assert im.mode in ("RGBA", "RGB")
        assert im.size == (8, 8)

    def test_conversion_deterministic(self, tmp_path):
        webp = _make_webp_bytes((10, 10))
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        b1, _, _ = _ensure_ollama_transport_compatible_image(p, "image/webp")
        b2, _, _ = _ensure_ollama_transport_compatible_image(p, "image/webp")
        assert b1 == b2

    def test_source_untouched(self, tmp_path):
        webp = _make_webp_bytes((8, 8))
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        before = hashlib.sha256(webp).hexdigest()
        import os

        before_mtime = os.stat(p).st_mtime_ns
        _ensure_ollama_transport_compatible_image(p, "image/webp")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == before
        assert os.stat(p).st_mtime_ns == before_mtime

    def test_png_not_transcoded(self, tmp_path):
        png = _make_png_bytes()
        p = tmp_path / "img.png"
        p.write_bytes(png)
        transport_bytes, mime, transcoded = _ensure_ollama_transport_compatible_image(p, "image/png")
        assert transcoded is False
        assert transport_bytes == png
        assert mime == "image/png"

    def test_jpeg_not_transcoded(self, tmp_path):
        jpg = _make_jpeg_bytes()
        p = tmp_path / "img.jpg"
        p.write_bytes(jpg)
        transport_bytes, mime, transcoded = _ensure_ollama_transport_compatible_image(p, "image/jpeg")
        assert transcoded is False
        assert transport_bytes == jpg
        assert mime == "image/jpeg"

    def test_mime_from_bytes_not_suffix(self, tmp_path):
        """A WebP file with .jpg suffix must still be transcoded."""
        webp = _make_webp_bytes()
        p = tmp_path / "image.jpg"
        p.write_bytes(webp)
        _, mime, transcoded = _ensure_ollama_transport_compatible_image(p, None)
        assert transcoded is True
        assert mime == "image/png"

    def test_temp_artifacts_cleaned(self, tmp_path):
        """prepare_image must not leave stray files behind beyond its temp dir."""
        import os

        webp = _make_webp_bytes((8, 8))
        p = tmp_path / "img.webp"
        p.write_bytes(webp)
        before = set(os.listdir(tmp_path))
        _ensure_ollama_transport_compatible_image(p, "image/webp")
        after = set(os.listdir(tmp_path))
        assert after == before


class TestPrepareImageTwoLayerHash:
    @pytest.mark.asyncio
    async def test_webp_canonical_vs_transport_hash(self, tmp_path):
        """prepare_image keeps canonical WebP hash and emits transport PNG
        hash + metadata, for a WebP file with .jpg suffix."""
        webp = _make_webp_bytes((12, 12))
        src = tmp_path / "source.jpg"
        src.write_bytes(webp)

        async def fake_resolve(image_source, task_id=None):
            return webp

        import tools.ollama_vision_client as mod

        with (
            patch.object(mod, "_resolve_image_bytes_async", fake_resolve),
            patch("tools.vision_tools._normalize_to_supported_image", _passthrough_norm),
        ):
            (
                data_url,
                w,
                h,
                mime,
                normalized_sha,
                transport_meta,
            ) = await prepare_image(str(src))

        expected_canonical = hashlib.sha256(webp).hexdigest()
        assert normalized_sha == expected_canonical  # canonical = WebP hash
        assert transport_meta["transport_transcoded"] is True
        assert transport_meta["transport_mime_type"] == "image/png"
        assert transport_meta["transport_image_sha256"] != expected_canonical
        # Transport hash must match the actual PNG bytes in the data URL:
        payload = data_url.split(",", 1)[1]
        decoded = base64.b64decode(payload)
        assert hashlib.sha256(decoded).hexdigest() == transport_meta["transport_image_sha256"]
        assert w == 12 and h == 12

    @pytest.mark.asyncio
    async def test_png_canonical_equals_transport(self, tmp_path):
        png = _make_png_bytes((10, 10))
        src = tmp_path / "img.png"
        src.write_bytes(png)

        async def fake_resolve(image_source, task_id=None):
            return png

        import tools.ollama_vision_client as mod

        with (
            patch.object(mod, "_resolve_image_bytes_async", fake_resolve),
            patch("tools.vision_tools._normalize_to_supported_image", _passthrough_norm),
        ):
            (
                _url,
                _w,
                _h,
                _mime,
                normalized_sha,
                transport_meta,
            ) = await prepare_image(str(src))

        expected = hashlib.sha256(png).hexdigest()
        assert normalized_sha == expected
        assert transport_meta["transport_image_sha256"] == expected  # identical
        assert transport_meta["transport_transcoded"] is False
        assert transport_meta["transport_mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_trace_and_result_contain_transport_meta_no_bytes(self):
        """End-to-end orchestrator result carries both hashes but no bytes."""
        from tools.vision_orchestrator import analyze_image
        from tools.vision_policy import (
            VisionRequest,
            VisionTask,
            VisionMode,
            VisionCriticality,
            ExecutionStatus,
        )
        from unittest.mock import AsyncMock

        with (
            patch(
                "tools.vision_orchestrator.prepare_image",
                new_callable=AsyncMock,
                return_value=(
                    "data:image/png;base64,TRANSPORT",
                    600,
                    600,
                    "image/webp",
                    "c" * 64,  # canonical (webp) hash
                    {
                        "transport_image_sha256": "d" * 64,
                        "transport_mime_type": "image/png",
                        "transport_transcoded": True,
                    },
                ),
            ),
            patch(
                "tools.vision_orchestrator.invoke_vision_model",
                new_callable=AsyncMock,
                return_value={
                    "execution_status": ExecutionStatus.SUCCESS.value,
                    "raw_text": '{"observation": "ok"}',
                    "error": None,
                },
            ),
        ):
            req = VisionRequest(
                request_id="t-webp",
                image_source="opaque://x",
                task=VisionTask.SCENE_DESCRIBE,
                mode=VisionMode.AUTO,
                criticality=VisionCriticality.NORMAL,
                question="describe",
            )
            result = await analyze_image(req, enabled=True)

        assert result["normalized_image_sha256"] == "c" * 64
        assert result["transport_image_sha256"] == "d" * 64
        assert result["transport_mime_type"] == "image/png"
        assert result["transport_transcoded"] is True
        assert result["logical_model_calls"] == 1
        # Trace carries hashes, never bytes:
        trace = result["trace"][0]
        assert trace["transport_image_sha256"] == "d" * 64
        blob = str(result)
        assert "TRANSPORT" not in blob  # no base64 payload
        assert "data:image" not in blob


def _passthrough_norm(path, mime):
    """Test double for _normalize_to_supported_image: keep bytes as-is."""
    return path, mime, None
