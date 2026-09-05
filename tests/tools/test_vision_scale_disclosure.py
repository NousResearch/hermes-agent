"""Downscale coordinate-scale disclosure tests.

When an image is downscaled (or region-cropped) before reaching a vision
model, the model's reported coordinates are in the *shrunk* (or crop-local)
coordinate space. These tests verify that both vision paths now disclose the
scale factor / crop offset so coordinates can be mapped back deterministically:

* ``tools.computer_use.tool._shrink_capture_for_vision`` returns
  ``(bytes, scale_note)``;
* ``tools.vision_tools.vision_analyze_tool`` emits a ``scale_note`` field and
  prefixes the analysis text when its downscale/region paths fire.

The scale math is verified deterministically with Pillow — no LLM needed.
"""

import io
import json
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tools.computer_use.tool import _shrink_capture_for_vision  # noqa: E402
from tools.vision_tools import _build_scale_note, vision_analyze_tool  # noqa: E402


ORIG_W, ORIG_H = 3024, 1964
SQUARE_X, SQUARE_Y, SQUARE_SIZE = 2400, 1500, 10


def _make_marker_png_bytes() -> bytes:
    """Synthetic 3024x1964 black PNG with a red 10px square at (2400, 1500)."""
    img = Image.new("RGB", (ORIG_W, ORIG_H), (0, 0, 0))
    for x in range(SQUARE_X, SQUARE_X + SQUARE_SIZE):
        for y in range(SQUARE_Y, SQUARE_Y + SQUARE_SIZE):
            img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_noise_png(path, width: int, height: int) -> None:
    """Random-noise PNG: incompressible, so file size ~ raw pixel bytes."""
    img = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    img.save(path, format="PNG")


def _red_square_center(img: Image.Image) -> tuple[float, float]:
    """Bounding-box center of reddish pixels (antialiasing-tolerant)."""
    rgb = img.convert("RGB")
    xs, ys = [], []
    px = rgb.load()
    for x in range(rgb.width):
        for y in range(rgb.height):
            r, g, b = px[x, y]
            if r > 100 and g < 100 and b < 100:
                xs.append(x)
                ys.append(y)
    assert xs, "red marker square not found in image"
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


class TestShrinkCaptureForVision:
    def test_downscale_note_recovers_original_position(self):
        raw = _make_marker_png_bytes()
        shrunk_bytes, note = _shrink_capture_for_vision(raw, ".png")

        assert note is not None
        assert "downscaled" in note
        assert f"{ORIG_W}x{ORIG_H}" in note
        # Stated rounded factor: 3024/1456 = 2.0769... -> 2.08
        assert "2.08" in note

        m = re.search(r"downscaled from (\d+)x(\d+) to (\d+)x(\d+)", note)
        assert m, f"note missing dimensions: {note}"
        ow, oh, nw, nh = (int(v) for v in m.groups())
        assert (ow, oh) == (ORIG_W, ORIG_H)

        shrunk = Image.open(io.BytesIO(shrunk_bytes))
        assert shrunk.size == (nw, nh)
        assert max(shrunk.size) <= 1456

        # Recompute the marker position in the shrunk image via PIL and map
        # it back with the factors stated in the note.
        cx, cy = _red_square_center(shrunk)
        fx, fy = ow / nw, oh / nh
        recovered_x, recovered_y = cx * fx, cy * fy
        orig_cx = SQUARE_X + (SQUARE_SIZE - 1) / 2.0
        orig_cy = SQUARE_Y + (SQUARE_SIZE - 1) / 2.0
        assert abs(recovered_x - orig_cx) <= 2.0, (recovered_x, orig_cx)
        assert abs(recovered_y - orig_cy) <= 2.0, (recovered_y, orig_cy)

    def test_no_note_when_under_cap(self):
        img = Image.new("RGB", (800, 600), (10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        out, note = _shrink_capture_for_vision(raw, ".png")
        assert note is None
        assert out == raw  # returned unchanged

    def test_no_note_on_undecodable_bytes(self):
        raw = b"not an image at all"
        out, note = _shrink_capture_for_vision(raw, ".png")
        assert out == raw
        assert note is None


class TestBuildScaleNote:
    def test_none_when_nothing_happened(self):
        assert _build_scale_note(None, None) is None
        assert _build_scale_note({}, {}) is None

    def test_scale_factor_math(self):
        note = _build_scale_note(
            {"orig_width": 3024, "orig_height": 1964,
             "new_width": 1512, "new_height": 982},
            None,
        )
        assert note is not None
        assert "3024x1964" in note and "1512x982" in note
        assert "2.00" in note

    def test_crop_offset_only(self):
        note = _build_scale_note(None, {"x": 300, "y": 200,
                                        "width": 500, "height": 400})
        assert note is not None
        assert "(300, 200)" in note
        assert "crop" in note.lower()


def _mock_llm_response(text: str = "described"):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response.choices = [mock_choice]
    return mock_response


class TestVisionAnalyzeScaleDisclosure:
    @pytest.mark.asyncio
    async def test_downscale_path_emits_scale_note(self, tmp_path):
        # Noise is incompressible: 3024x1964 RGB noise -> ~17 MB PNG, base64
        # ~23 MB. With the hard cap patched to 8 MB, the pre-flight resize
        # fires and must disclose the downscale.
        img_path = tmp_path / "big_noise.png"
        _make_noise_png(img_path, ORIG_W, ORIG_H)

        with (
            patch("tools.vision_tools._MAX_BASE64_BYTES", 8 * 1024 * 1024),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=_mock_llm_response(),
            ),
        ):
            result = json.loads(
                await vision_analyze_tool(str(img_path), "describe", "test/model")
            )

        assert result["success"] is True
        assert "scale_note" in result
        note = result["scale_note"]
        assert f"downscaled from {ORIG_W}x{ORIG_H}" in note

        # Deterministic scale math: the factors in the note must equal
        # orig/new from the stated dimensions (2-decimal rounding).
        m = re.search(r"downscaled from (\d+)x(\d+) to (\d+)x(\d+)", note)
        assert m
        ow, oh, nw, nh = (int(v) for v in m.groups())
        assert (ow, oh) == (ORIG_W, ORIG_H)
        assert nw < ORIG_W and nh < ORIG_H
        fx = ow / nw
        assert f"{fx:.2f}" in note

        # Non-schema-aware consumers still see the note: analysis is prefixed.
        assert result["analysis"].startswith(f"[{note}]")

    @pytest.mark.asyncio
    async def test_small_image_has_no_scale_note(self, tmp_path):
        img_path = tmp_path / "small.png"
        Image.new("RGB", (320, 200), (5, 5, 5)).save(img_path, format="PNG")

        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response(),
        ):
            result = json.loads(
                await vision_analyze_tool(str(img_path), "describe", "test/model")
            )

        assert result["success"] is True
        assert "scale_note" not in result
        assert not result["analysis"].startswith("[")

    @pytest.mark.asyncio
    async def test_region_plus_downscale_discloses_offset_and_factor(self, tmp_path):
        # Crop a 2400x1800 noise region (still ~17 MB base64) so BOTH the
        # crop-offset and the downscale disclosures must appear.
        img_path = tmp_path / "big_noise_region.png"
        _make_noise_png(img_path, ORIG_W, ORIG_H)
        region = [300, 100, 2700, 1900]  # 2400x1800 at offset (300, 100)

        with (
            patch("tools.vision_tools._MAX_BASE64_BYTES", 8 * 1024 * 1024),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=_mock_llm_response(),
            ),
        ):
            result = json.loads(
                await vision_analyze_tool(
                    str(img_path), "describe", "test/model", region=region,
                )
            )

        assert result["success"] is True
        note = result["scale_note"]
        # Downscale factor disclosed, computed from the crop dimensions.
        m = re.search(r"downscaled from (\d+)x(\d+) to (\d+)x(\d+)", note)
        assert m
        ow, oh, nw, nh = (int(v) for v in m.groups())
        assert (ow, oh) == (2400, 1800)
        assert f"{ow / nw:.2f}" in note
        # Crop offset disclosed: coordinates are relative to the crop origin.
        assert "(300, 100)" in note
        assert "relative" in note
        assert result["analysis"].startswith(f"[{note}]")

    @pytest.mark.asyncio
    async def test_region_only_discloses_offset(self, tmp_path):
        img_path = tmp_path / "small_region.png"
        Image.new("RGB", (800, 600), (0, 0, 0)).save(img_path, format="PNG")

        with patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=_mock_llm_response(),
        ):
            result = json.loads(
                await vision_analyze_tool(
                    str(img_path), "describe", "test/model",
                    region=[100, 50, 400, 300],
                )
            )

        assert result["success"] is True
        note = result["scale_note"]
        assert "(100, 50)" in note
        assert "downscaled" not in note  # crop fits: no scale clause
