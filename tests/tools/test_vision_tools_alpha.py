"""Tests for alpha-channel compositing in vision_tools (#64548).

These tests verify that transparent PNG images are properly composited
onto a white background before being sent to vision providers.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.vision_tools import (
    _composite_alpha_to_background,
    _composite_png_alpha,
    _normalize_to_supported_image,
)

# All tests here require Pillow — skip if not importable.
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("PIL", reason="Pillow not installed"),
    reason="Pillow required for alpha compositing tests",
)


# ---------------------------------------------------------------------------
# _composite_alpha_to_background  (unit tests)
# ---------------------------------------------------------------------------


def _rgba_image(size=(10, 10), color=(255, 0, 0, 128), check_import=True):
    """Create a small RGBA image. Injects a couple fully-transparent pixels."""
    from PIL import Image as PIL

    img = PIL.new("RGBA", size, color)
    # Set a few pixels to fully transparent with garbage RGB.
    pixels = img.load()
    if pixels:
        pixels[0, 0] = (42, 128, 200, 0)   # garbage RGB, fully transparent
        pixels[9, 9] = (7, 99, 255, 0)      # garbage RGB, fully transparent
    return img


def test_composite_alpha_to_background_flattens_rgba():
    """RGBA image with partial transparency is flattened to RGB."""
    from PIL import Image as PIL

    img = _rgba_image()
    result = _composite_alpha_to_background(img)
    assert result.mode == "RGB"
    assert result.size == (10, 10)


def test_composite_alpha_to_background_preserves_opaque_rgb():
    """Already-opaque RGB image passes through unchanged."""
    from PIL import Image as PIL

    img = PIL.new("RGB", (10, 10), (255, 0, 0))
    result = _composite_alpha_to_background(img)
    assert result is img  # same object — no-op


def test_composite_alpha_to_background_preserves_fully_opaque_rgba():
    """RGBA where every alpha channel is 255 passes through unchanged."""
    from PIL import Image as PIL

    img = PIL.new("RGBA", (10, 10), (255, 0, 0, 255))
    result = _composite_alpha_to_background(img)
    assert result is img


def test_composite_alpha_to_background_converts_p_mode():
    """Palette-mode image with transparency is flattened to RGB."""
    from PIL import Image as PIL

    img = PIL.new("P", (10, 10))
    img.info["transparency"] = 0  # mark color index 0 as transparent
    result = _composite_alpha_to_background(img)
    assert result.mode == "RGB"


def test_composite_garbage_under_transparent_pixels_removed():
    """Fully-transparent pixels with garbage RGB are overwritten with bg.

    The whole point of #64548: under a white background, the previously
    transparent (42,128,200) pixel becomes white, not garbage noise.
    """
    from PIL import Image as PIL

    img = PIL.new("RGBA", (3, 3), (0, 0, 0, 0))
    pixels = img.load()
    pixels[1, 1] = (42, 128, 200, 0)  # garbage under full transparency

    result = _composite_alpha_to_background(img)
    rpixels = result.load()
    # The previously-garbage pixel should now be white (the composite bg).
    assert rpixels[1, 1] == (255, 255, 255)


def test_composite_returns_rgb_without_alpha_band():
    """Result has no alpha channel — providers won't see RGBA garbage."""
    from PIL import Image as PIL

    img = PIL.new("RGBA", (5, 5), (128, 128, 128, 64))
    result = _composite_alpha_to_background(img)
    assert result.mode == "RGB"
    with pytest.raises((ValueError, AttributeError)):
        result.getchannel("A")  # no alpha channel in RGB image


# ---------------------------------------------------------------------------
# _composite_png_alpha  (normalization-path integration test)
# ---------------------------------------------------------------------------


@pytest.fixture
def transparent_png(tmp_path) -> Path:
    """Create a small PNG with transparent pixels."""
    from PIL import Image as PIL

    img = PIL.new("RGBA", (20, 20), (128, 128, 128, 0))  # fully transparent
    path = tmp_path / "transparent.png"
    img.save(path, format="PNG")
    return path


@pytest.fixture
def opaque_png(tmp_path) -> Path:
    """Create a small fully-opaque PNG."""
    from PIL import Image as PIL

    img = PIL.new("RGB", (20, 20), (255, 0, 0))
    path = tmp_path / "opaque.png"
    img.save(path, format="PNG")
    return path


def test_composite_png_alpha_creates_flattened_copy(transparent_png):
    """Transparent PNG produces a new temp flattened file."""
    path, mime, err = _composite_png_alpha(transparent_png, "image/png")
    assert err is None
    assert mime == "image/png"
    assert path != transparent_png  # new temp file
    assert path.exists()
    # Verify the result is actually flattened (RGB, not RGBA).
    from PIL import Image as PIL

    with PIL.open(path) as img:
        assert img.mode == "RGB"
        # Not a single pixel should be transparent now.
        rpixels = img.load()
        assert rpixels[0, 0] != (128, 128, 128)  # no longer gray garbage


def test_composite_png_alpha_skips_opaque_png(opaque_png):
    """Fully-opaque PNG passes through unchanged (no temp file)."""
    path, mime, err = _composite_png_alpha(opaque_png, "image/png")
    assert err is None
    assert mime == "image/png"
    assert path == opaque_png  # same path — no temp file created


def test_composite_png_alpha_handles_missing_pillow(transparent_png):
    """When Pillow is missing, pass through without error."""
    with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
        path, mime, err = _composite_png_alpha(transparent_png, "image/png")
    assert err is None
    assert mime == "image/png"
    assert path == transparent_png


# ---------------------------------------------------------------------------
# _normalize_to_supported_image  (integration — it calls _composite_png_alpha)
# ---------------------------------------------------------------------------


def test_normalize_flattens_transparent_png(transparent_png):
    """_normalize_to_supported_image on a PNG triggers alpha compositing."""
    path, mime, err = _normalize_to_supported_image(transparent_png, "image/png")
    assert err is None
    assert mime == "image/png"
    # For a transparent PNG, a new flattened file should be returned.
    assert path != transparent_png
    from PIL import Image as PIL

    with PIL.open(path) as img:
        assert img.mode == "RGB"


def test_normalize_passes_opaque_png_through(opaque_png):
    """_normalize_to_supported_image does NOT create a temp file for opaque PNG."""
    path, mime, err = _normalize_to_supported_image(opaque_png, "image/png")
    assert err is None
    assert mime == "image/png"
    assert path == opaque_png  # same file, no temp created


def test_normalize_non_png_supported_type_passes_through(tmp_path):
    """_normalize_to_supported_image on JPEG does NOT trigger alpha path."""
    from PIL import Image as PIL

    img = PIL.new("RGB", (10, 10), (0, 255, 0))
    jpg_path = tmp_path / "test.jpg"
    img.save(jpg_path, format="JPEG")
    path, mime, err = _normalize_to_supported_image(jpg_path, "image/jpeg")
    assert err is None
    assert mime == "image/jpeg"
    assert path == jpg_path


def test_normalize_transparent_png_get_text_no_glitch(transparent_png):
    """End-to-end: flattened PNG should no longer contain garbage pixels.

    The fully-transparent (128,128,128) pixel becomes (255,255,255) under
    the white composite. This is what the vision model should see.
    """
    path, mime, err = _normalize_to_supported_image(transparent_png, "image/png")
    assert err is None
    from PIL import Image as PIL

    with PIL.open(path) as img:
        pixels = img.load()
        assert pixels[0, 0] == (255, 255, 255), (
            f"Expected white background, got {pixels[0, 0]}"
        )
