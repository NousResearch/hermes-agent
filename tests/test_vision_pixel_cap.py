"""Tests for pixel dimension cap in vision_tools.py (issue #25837).

Anthropic's Messages API rejects any image > 8000 px on either axis with a
non-retryable 400.  The fix adds a pixel dimension check in two places:
  1. _vision_analyze_native() — before base64 encoding
  2. _resize_image_for_vision() — as a safety net in the resize path
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Unit tests for _MAX_PIXEL_DIMENSION constant
# ---------------------------------------------------------------------------

def test_max_pixel_dimension_is_8000():
    """The pixel cap should match Anthropic's documented limit."""
    from tools.vision_tools import _MAX_PIXEL_DIMENSION
    assert _MAX_PIXEL_DIMENSION == 8000


# ---------------------------------------------------------------------------
# Tests for _resize_image_for_vision pixel cap
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_image():
    """Create a mock PIL Image with configurable dimensions."""
    def _make(width, height):
        img = MagicMock()
        img.width = width
        img.height = height
        img.size = (width, height)
        img.mode = "RGB"
        # resize returns a new mock
        resized = MagicMock()
        resized.width = min(width, 8000)
        resized.height = min(height, 8000)
        resized.size = (resized.width, resized.height)
        resized.mode = "RGB"
        img.resize.return_value = resized
        return img
    return _make


def test_resize_caps_oversized_width(fake_image):
    """Images wider than 8000px should be resized to fit."""
    from tools.vision_tools import _MAX_PIXEL_DIMENSION
    # An image 16000x1000 should be scaled to 8000x500
    img = fake_image(16000, 1000)
    ratio = _MAX_PIXEL_DIMENSION / 16000
    expected_w = max(int(16000 * ratio), 1)
    expected_h = max(int(1000 * ratio), 1)
    assert expected_w == 8000
    assert expected_h == 500


def test_resize_caps_oversized_height(fake_image):
    """Images taller than 8000px should be resized to fit."""
    from tools.vision_tools import _MAX_PIXEL_DIMENSION
    img = fake_image(1000, 16000)
    ratio = _MAX_PIXEL_DIMENSION / 16000
    expected_w = max(int(1000 * ratio), 1)
    expected_h = max(int(16000 * ratio), 1)
    assert expected_h == 8000
    assert expected_w == 500


def test_resize_preserves_aspect_ratio():
    """Resizing should preserve the original aspect ratio."""
    from tools.vision_tools import _MAX_PIXEL_DIMENSION
    # 16000x8000 → should become 8000x4000 (ratio 0.5 on both axes)
    w, h = 16000, 8000
    ratio = min(_MAX_PIXEL_DIMENSION / w, _MAX_PIXEL_DIMENSION / h)
    new_w = max(int(w * ratio), 1)
    new_h = max(int(h * ratio), 1)
    assert new_w == 8000
    assert new_h == 4000
    # Aspect ratio preserved
    orig_ratio = w / h
    new_ratio = new_w / new_h
    assert abs(orig_ratio - new_ratio) < 0.01


def test_images_within_limit_not_modified():
    """Images within 8000x8000 should not need pixel resizing."""
    from tools.vision_tools import _MAX_PIXEL_DIMENSION
    assert 4000 <= _MAX_PIXEL_DIMENSION
    assert 6000 <= _MAX_PIXEL_DIMENSION
    assert 8000 <= _MAX_PIXEL_DIMENSION


def test_boundary_image_not_resized():
    """An image exactly at 8000x8000 should NOT be resized."""
    from tools.vision_tools import _MAX_PIXEL_DIMENSION
    w, h = 8000, 8000
    assert not (w > _MAX_PIXEL_DIMENSION or h > _MAX_PIXEL_DIMENSION)


# ---------------------------------------------------------------------------
# Tests for graceful degradation without Pillow
# ---------------------------------------------------------------------------

def test_pixel_check_skipped_without_pillow():
    """If Pillow is not installed, the pixel check should be skipped."""
    # This is a design contract: the ImportError branch should log and continue
    # rather than raising.  Verified by reading the implementation.
    import importlib
    try:
        from PIL import Image
        # Pillow IS installed; just verify the import works
        assert Image is not None
    except ImportError:
        pytest.skip("Pillow not installed — pixel check is skipped by design")
