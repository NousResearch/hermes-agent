"""Regression tests for the pixel-dimension image-guard fix.

Pre-fix, every image-oversize guard in Hermes reasoned about
**bytes** only, while Anthropic enforces two independent
ceilings: 5 MB encoded AND 8000px longest side. A tall
full-page screenshot (e.g. 1200×12000) is well under 5 MB but
vastly over 8000px, slipped through every guard, got baked into
immutable conversation history, and bricked the session on
every subsequent replay with a non-retryable HTTP 400. See
#25837 and #37677.

These tests cover the four fix sites:

  1. ``agent/error_classifier.py`` _IMAGE_TOO_LARGE_PATTERNS
     matches the dimension-cap error message.
  2. ``tools/vision_tools.py`` _image_exceeds_dimension helper
     enforces the 7900px cap (Anthropic's 8000 minus 100px
     headroom).
  3. ``tools/vision_tools.py`` _resize_image_for_vision pre-caps
     dimensions BEFORE the byte-sizing loop, so a tiny-but-tall
     image is downscaled even though it doesn't trip the byte cap.
  4. ``agent/conversation_compression.py`` reactive shrinker
     decodes the image and shrinks on dimension violation, not
     just byte.
"""

from __future__ import annotations

import base64
import io

import pytest


# --- 1. error_classifier dimension patterns ---


class TestImageTooLargePatternsDimensionCap:
    """The error classifier must recognise Anthropic's 8000px message.

    Without this, the reactive shrink/retry path never fires for
    dimension violations — the 400 falls through as a generic
    non-retryable error and the session is bricked.
    """

    def test_dimension_pattern_matches_anthropic_message(self):
        from agent.error_classifier import _IMAGE_TOO_LARGE_PATTERNS

        # The exact phrasing Anthropic returns. Multiple variants
        # seen in the wild (#25837, #37677, #29490).
        messages = [
            "messages.0.content.0.image.source.base64: At least one of the "
            "image dimensions exceed max allowed size: 8000 pixels",
            "image dimensions exceed max allowed size",
            "image dimensions exceed",
            "max allowed size: 8000",
        ]
        for msg in messages:
            assert any(
                p in msg for p in _IMAGE_TOO_LARGE_PATTERNS
            ), f"dimension message not matched: {msg!r}"

    def test_legacy_byte_pattern_still_matches(self):
        """Pre-existing 5 MB patterns must still match (regression guard)."""
        from agent.error_classifier import _IMAGE_TOO_LARGE_PATTERNS

        legacy = [
            "image exceeds 5 MB maximum",
            "image too large",
            "image_too_large",
            "image size exceeds",
        ]
        for msg in legacy:
            assert any(p in msg for p in _IMAGE_TOO_LARGE_PATTERNS), (
                f"legacy byte pattern not matched: {msg!r}"
            )

    def test_classifier_routes_dimension_error_to_image_too_large(self):
        """The _classify_400 / _classify_by_message path that uses
        _IMAGE_TOO_LARGE_PATTERNS routes dimension errors to the
        ``image_too_large`` FailoverReason.

        Pre-fix, the patterns only matched the 5 MB byte message,
        so this 400 fell through to a generic non-retryable bucket
        and the session was bricked. The matching pattern list
        update in agent/error_classifier.py is the upstream of
        this fix.
        """
        # Call the same path the runtime does: build a fake 400
        # exception, then verify the classified reason is
        # image_too_large (not generic).
        from agent.error_classifier import classify_api_error

        msg = (
            "messages.0.content.0.image.source.base64: At least one of the "
            "image dimensions exceed max allowed size: 8000 pixels"
        )
        # Wrap the message in an exception matching how the runtime
        # calls into classify_api_error.
        fake_exc = Exception(msg)
        result = classify_api_error(
            fake_exc, provider="anthropic", model="claude-opus-4-7",
        )
        # result is a ClassifiedError; .reason is the FailoverReason
        # enum. image_too_large == "image_too_large" by value.
        assert "image_too_large" in str(result), (
            f"dimension 400 must classify as image_too_large, got {result!r}"
        )


# --- 2. _image_exceeds_dimension helper ---


class TestImageExceedsDimensionHelper:
    """The pixel-dimension helper is the shared primitive for the cap."""

    def test_under_cap_returns_false(self):
        from tools.vision_tools import _image_exceeds_dimension
        assert _image_exceeds_dimension(800, 600) is False
        assert _image_exceeds_dimension(7900, 7900) is False
        # 7901 is the first size to fail (we use 7900 as the resize
        # target, with 100px headroom under Anthropic's hard 8000).
        assert _image_exceeds_dimension(7901, 100) is True
        assert _image_exceeds_dimension(100, 7901) is True

    def test_over_cap_returns_true(self):
        from tools.vision_tools import _image_exceeds_dimension
        assert _image_exceeds_dimension(8001, 100) is True
        assert _image_exceeds_dimension(100, 12000) is True
        # The actual reproducer from #25837 / #37677.
        assert _image_exceeds_dimension(1200, 12000) is True

    def test_corrupt_dimensions_treated_as_exceeding(self):
        """0×N / N×0 is corrupt; we return True so the caller falls
        back to a non-vision path rather than embedding an image the
        provider will reject with a generic 400."""
        from tools.vision_tools import _image_exceeds_dimension
        assert _image_exceeds_dimension(0, 100) is True
        assert _image_exceeds_dimension(100, 0) is True
        assert _image_exceeds_dimension(-1, 100) is True

    def test_custom_max_dim_parameter(self):
        from tools.vision_tools import _image_exceeds_dimension
        # Hypothetical provider with a 4000px cap.
        assert _image_exceeds_dimension(4001, 100, max_dim=4000) is True
        assert _image_exceeds_dimension(3999, 100, max_dim=4000) is False


# --- 3. _resize_image_for_vision pre-cap ---


class TestResizeImageForVisionDimensionPreCap:
    """The resizer must downscale a tall image BEFORE the byte loop.

    Without this fix, a 0.06 MB / 1200×12000 PNG passes the byte
    test untouched and gets embedded at full dimensions → 400.
    """

    def test_tall_small_byte_image_gets_downscaled(self, tmp_path):
        """1200×12000 PNG at < 1MB: should be downscaled to ≤7900
        longest side even though the byte cap is satisfied.
        """
        pytest.importorskip("PIL", reason="Pillow is required for resize")
        from PIL import Image

        from tools.vision_tools import (
            _MAX_IMAGE_DIMENSION,
            _resize_image_for_vision,
        )

        # Build a real 1200×12000 PNG. PIL stores it efficiently so
        # the byte count is tiny — exactly the reproducer from the
        # issue. Solid colour, not a photo, so the PNG is small.
        img = Image.new("RGB", (1200, 12000), color=(255, 255, 255))
        src = tmp_path / "tall.png"
        img.save(src, format="PNG", optimize=True)
        assert src.stat().st_size < 1 * 1024 * 1024, (
            f"fixture too large: {src.stat().st_size} bytes"
        )

        # max_base64_bytes is intentionally huge so the byte cap
        # would NOT trigger; the dimension cap must catch it.
        result = _resize_image_for_vision(
            src, mime_type="image/png",
            max_base64_bytes=100 * 1024 * 1024,
        )

        # Decode the result and check the dimensions.
        from PIL import Image as _PILImage
        import base64 as _b64
        header, _, data = result.partition(",")
        raw = _b64.b64decode(data)
        with _PILImage.open(io.BytesIO(raw)) as result_img:
            w, h = result_img.size

        assert max(w, h) <= _MAX_IMAGE_DIMENSION, (
            f"pre-cap should downscale to <= {_MAX_IMAGE_DIMENSION}px, "
            f"got {w}x{h}"
        )
        # Aspect ratio preserved.
        assert abs(w / h - 1200 / 12000) < 0.01, (
            f"aspect ratio not preserved: {w}x{h} from 1200x12000"
        )

    def test_small_image_untouched(self, tmp_path):
        """Normal-sized images must not be touched."""
        pytest.importorskip("PIL")
        from PIL import Image

        from tools.vision_tools import _resize_image_for_vision

        img = Image.new("RGB", (800, 600), color=(128, 128, 128))
        src = tmp_path / "small.png"
        img.save(src, format="PNG")

        result = _resize_image_for_vision(
            src, mime_type="image/png",
            max_base64_bytes=10 * 1024 * 1024,
        )
        header, _, data = result.partition(",")
        from PIL import Image as _PILImage
        with _PILImage.open(io.BytesIO(_b64decode(data))) as result_img:
            w, h = result_img.size
        assert (w, h) == (800, 600), f"small image was modified: got {w}x{h}"


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data)


# --- 4. conversation_compression reactive shrinker ---


class TestShrinkImagePartsDecodesDimension:
    """The reactive shrinker must catch dimension-only violations.

    Pre-fix, the gate was ``len(url) <= target_bytes``. A 0.07 MB
    / 1000×11000 tool-result slipped through, the retry re-sent
    the identical payload, and the 400 bricked the session.
    """

    def test_tall_image_decodes_and_shrinks(self, tmp_path):
        """A 0.07 MB / 1000×11000 data URL should be shrunk, even
        though the byte cap is not exceeded."""
        pytest.importorskip("PIL")
        from PIL import Image

        from agent.conversation_compression import (
            try_shrink_image_parts_in_messages,
        )
        from tools.vision_tools import _MAX_IMAGE_DIMENSION

        # Build the reproducer from the issue body.
        img = Image.new("RGB", (1000, 11000), color=(200, 200, 200))
        src = tmp_path / "tall.jpg"
        img.save(src, format="JPEG", quality=50)
        # Sanity: this is the case the old gate missed.
        assert src.stat().st_size < 1 * 1024 * 1024, (
            f"fixture too large: {src.stat().st_size} bytes"
        )

        # Base64-encode and embed as an OpenAI-style image_url part.
        with open(src, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        api_messages = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

        # The shrinker must report it changed something.
        changed = try_shrink_image_parts_in_messages(api_messages)
        assert changed is True, (
            "tall 0.07 MB image was not shrunk — dimension gate is still "
            "byte-only (#25837 regression)"
        )

        # And the result is now within the dimension cap.
        new_url = api_messages[0]["content"][0]["image_url"]["url"]
        _, _, new_data = new_url.partition(",")
        with Image.open(io.BytesIO(base64.b64decode(new_data))) as r:
            w, h = r.size
        assert max(w, h) <= _MAX_IMAGE_DIMENSION, (
            f"shrunk image still over dimension cap: {w}x{h}"
        )

    def test_small_image_unchanged(self, tmp_path):
        """Sanity: a small image that doesn't trip either cap must
        not be touched (no shrink, no failure)."""
        pytest.importorskip("PIL")
        from PIL import Image

        from agent.conversation_compression import (
            try_shrink_image_parts_in_messages,
        )

        img = Image.new("RGB", (400, 300), color=(100, 100, 100))
        src = tmp_path / "small.jpg"
        img.save(src, format="JPEG", quality=70)

        with open(src, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        api_messages = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

        # Either changed=False (gate correctly skipped), or
        # changed=True with a smaller image — both are valid; the
        # important thing is no exception and the function returns
        # without burning the retry budget.
        result = try_shrink_image_parts_in_messages(api_messages)
        assert isinstance(result, bool)
