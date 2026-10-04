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

    def test_crop_uses_display_orientation_for_exif_rotated_jpeg(self, tmp_path):
        from PIL import ImageOps

        from tools.vision_tools import _crop_image_region

        src = tmp_path / "oriented.jpg"
        image = Image.new("RGB", (80, 40))
        for x in range(image.width):
            for y in range(image.height):
                image.putpixel((x, y), (x * 3, y * 6, (x + y) * 2))
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        image.save(src, format="JPEG", quality=95, exif=exif)

        region = [0, 0, 40, 60]
        with Image.open(src) as encoded:
            expected = ImageOps.exif_transpose(encoded).crop(region).convert("RGB")

        cropped_path, mime, err = _crop_image_region(src, region)

        assert err is None
        assert mime == "image/png"
        with Image.open(cropped_path) as actual:
            assert actual.size == (40, 60)
            assert list(actual.convert("RGB").getdata()) == list(expected.getdata())

        # Pass 1 (the full image the model reads coordinates off) must show the pixels the crop
        # indexes, whether or not the provider honours EXIF — so decode it without the tag. The
        # file is small enough that no resize runs.
        from agent.image_routing import _file_to_data_url
        from tools.vision_tools import _vision_analyze_native

        native = asyncio.run(_vision_analyze_native(str(src), "full shot"))
        full_views = {
            "attachment": _file_to_data_url(src),
            "vision_analyze": next(p["image_url"]["url"] for p in native["content"] if p.get("type") == "image_url"),
        }
        for label, url in full_views.items():
            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as shown:
                seen = shown.convert("RGB").crop(region)
            diffs = [abs(a - b) for pa, pb in zip(seen.getdata(), expected.getdata()) for a, b in zip(pa, pb)]
            assert seen.size == expected.size and max(diffs) < 16, label  # JPEG re-encode tolerance

        # Without the tag nothing is re-encoded: the attachment carries the file's own bytes.
        untagged = tmp_path / "plain.jpg"
        image.save(untagged, format="JPEG", quality=95)
        assert base64.b64decode(_file_to_data_url(untagged).split(",", 1)[1]) == untagged.read_bytes()

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
