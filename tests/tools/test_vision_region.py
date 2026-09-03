"""Tests for the optional region crop parameter on vision_analyze.

``region: [x1, y1, x2, y2]`` (pixel coords in the ORIGINAL image space) crops
the image BEFORE the downscale pipeline so the cropped area gets the full
resolution budget — a "zoom" for detail work after a full shot.

Ported from: QwenLM/qwen-code zoom-image.ts (Apache-2.0).
"""

from __future__ import annotations

import asyncio
import base64
import io

import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

pytestmark = pytest.mark.skipif(Image is None, reason="Pillow not installed")


def _make_png(path, width=100, height=50):
    img = Image.new("RGB", (width, height), (200, 30, 30))
    img.save(path, format="PNG")
    return path


def _decoded_size(data_url: str):
    """Return (w, h) of the image inside a base64 data URL."""
    b64 = data_url.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(b64))) as img:
        return img.size


# ─── _crop_image_region helper ───────────────────────────────────────────────


class TestCropImageRegion:
    def test_crop_applied(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        src = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, mime, err = _crop_image_region(src, [10, 10, 60, 40])
        assert err is None
        assert cropped_path is not None and cropped_path.exists()
        with Image.open(cropped_path) as img:
            assert img.size == (50, 30)

    def test_out_of_bounds_clamped_to_image(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        src = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, mime, err = _crop_image_region(src, [-10, -10, 200, 200])
        assert err is None
        with Image.open(cropped_path) as img:
            assert img.size == (100, 50)

    def test_zero_area_rejected_with_actual_dims_in_error(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        src = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, mime, err = _crop_image_region(src, [200, 200, 300, 300])
        assert cropped_path is None
        assert err is not None
        # Error must name the actual image dimensions so the model can retry.
        assert "100" in err and "50" in err

    def test_inverted_coords_rejected_with_dims(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        src = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, mime, err = _crop_image_region(src, [60, 40, 10, 10])
        assert cropped_path is None
        assert "100" in err and "50" in err

    def test_malformed_region_rejected(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        src = _make_png(tmp_path / "src.png", 100, 50)
        for bad in ([1, 2, 3], "10,10,60,40", [1, 2, 3, "x"], None):
            cropped_path, mime, err = _crop_image_region(src, bad)
            assert cropped_path is None
            assert err is not None


# ─── native fast path with region ────────────────────────────────────────────


class TestNativePathRegion:
    def test_region_crops_before_embed(self, tmp_path):
        from tools.vision_tools import _vision_analyze_native

        src = _make_png(tmp_path / "img.png", 100, 50)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(src), "zoom", region=[10, 10, 60, 40])
        )
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            p["image_url"]["url"]
            for p in result["content"]
            if p.get("type") == "image_url"
        )
        assert _decoded_size(url) == (50, 30)

    def test_no_region_behavior_unchanged(self, tmp_path):
        from tools.vision_tools import _vision_analyze_native

        src = _make_png(tmp_path / "img.png", 100, 50)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(src), "full shot")
        )
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            p["image_url"]["url"]
            for p in result["content"]
            if p.get("type") == "image_url"
        )
        assert _decoded_size(url) == (100, 50)

    def test_zero_area_region_returns_error_with_dims(self, tmp_path):
        import json

        from tools.vision_tools import _vision_analyze_native

        src = _make_png(tmp_path / "img.png", 100, 50)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(src), "zoom", region=[500, 500, 600, 600])
        )
        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload.get("success") is False
        msg = json.dumps(payload)
        assert "100" in msg and "50" in msg

    def test_crop_applied_before_downscale_gets_full_budget(self, tmp_path):
        """The crop happens BEFORE _resize_image_for_vision, so a small region
        of a huge image survives at native resolution instead of being
        downscaled with the rest."""
        from tools.vision_tools import _EMBED_MAX_DIMENSION, _vision_analyze_native

        # Taller than the embed long-edge cap — full shot would be downscaled.
        big = tmp_path / "big.png"
        Image.new("RGB", (200, _EMBED_MAX_DIMENSION + 500), (0, 100, 0)).save(
            big, format="PNG"
        )
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(big), "zoom", region=[0, 0, 200, 300])
        )
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            p["image_url"]["url"]
            for p in result["content"]
            if p.get("type") == "image_url"
        )
        # Region kept at native resolution — no downscale applied to the crop.
        assert _decoded_size(url) == (200, 300)


# ─── schema + handler wiring ─────────────────────────────────────────────────


class TestSchemaAndHandler:
    def test_schema_declares_optional_region(self):
        from tools.vision_tools import VISION_ANALYZE_SCHEMA

        props = VISION_ANALYZE_SCHEMA["parameters"]["properties"]
        assert "region" in props
        assert props["region"]["type"] == "array"
        assert "region" not in VISION_ANALYZE_SCHEMA["parameters"]["required"]
        # Description must document original-image pixel space.
        assert "original" in props["region"]["description"].lower()

    def test_handler_passes_region_to_native_path(self, tmp_path, monkeypatch):
        from tools import vision_tools
        from tools.vision_tools import _handle_vision_analyze

        src = _make_png(tmp_path / "img.png", 100, 50)
        seen = {}

        async def _fake_native(image_url, question, task_id=None, region=None):
            seen["region"] = region
            return {"_multimodal": True, "content": []}

        monkeypatch.setattr(vision_tools, "_vision_analyze_native", _fake_native)
        monkeypatch.setattr(
            vision_tools, "_should_use_native_vision_fast_path", lambda: True
        )
        asyncio.get_event_loop().run_until_complete(
            _handle_vision_analyze(
                {"image_url": str(src), "question": "q", "region": [1, 2, 30, 40]}
            )
        )
        assert seen["region"] == [1, 2, 30, 40]
