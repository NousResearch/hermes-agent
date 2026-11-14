"""Behaviour contracts for alpha flattening in tools/vision_tools_image_prep.py.

Some OpenAI-compatible backends (a GLM/Zai route behind a LiteLLM proxy was measured)
hang instead of decoding alpha-channel rasters — RGBA PNG, the default clipboard-
screenshot format on many platforms. A vision embed is baked into immutable history
and re-sent every turn, so one RGBA screenshot wedges the session's vision tooling.
The prep layer therefore composites alpha PNG/WebP onto an opaque white background
(JPEG q85) before the embed, unless ``vision.flatten_alpha: false``.
"""

from __future__ import annotations

import pytest

from hermes_cli.config import get_config_path
from tools.vision_tools import _secure_cache_dir
from tools.vision_tools_image_prep import (
    _flatten_image_alpha,
    _normalize_to_supported_image,
    _vision_flatten_alpha_enabled,
)

PIL = pytest.importorskip("PIL.Image")


def _write_config(text: str) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rgba_png(path, size=(40, 30), color=(200, 30, 30), alpha=128):
    PIL.new("RGBA", size, color + (alpha,)).save(path, format="PNG")
    return path


def _opaque_png(path, size=(40, 30)):
    PIL.new("RGB", size, (30, 200, 30)).save(path, format="PNG")
    return path


def _opaque_jpeg(path, size=(40, 30)):
    PIL.new("RGB", size, (30, 30, 200)).save(path, format="JPEG")
    return path


def _vision_cache_dir():
    return _secure_cache_dir("cache/vision", "temp_vision_images")


class TestFlattenImageAlpha:
    def test_rgba_png_composited_onto_white_as_rgb_jpeg(self, tmp_path):
        """Alpha pixels composite onto white and re-encode as an opaque RGB JPEG —
        never a re-saved RGBA, which would reach the backend unchanged."""
        shot = _rgba_png(tmp_path / "shot.png")
        out = _flatten_image_alpha(shot)
        assert out is not None and out != shot and shot.exists()
        assert out.suffix == ".jpg"
        with PIL.open(out) as flat:
            assert flat.mode == "RGB" and flat.size == (40, 30)
            r, g, b = flat.getpixel((0, 0))
        # (200, 30, 30) at alpha 128/255 over white: 200*.502+127≈227, 30*.502+127≈142.
        assert abs(r - 227) <= 2 and abs(g - 142) <= 2 and abs(b - 142) <= 2

    def test_alpha_webp_flattens_too(self, tmp_path):
        webp = tmp_path / "shot.webp"
        PIL.new("RGBA", (40, 30), (0, 0, 255, 64)).save(webp, format="WEBP")
        out = _flatten_image_alpha(webp)
        assert out is not None
        with PIL.open(out) as flat:
            assert flat.mode == "RGB"

    def test_fully_opaque_images_skip_the_reencode(self, tmp_path):
        assert _flatten_image_alpha(_opaque_png(tmp_path / "a.png")) is None
        assert _flatten_image_alpha(_opaque_jpeg(tmp_path / "b.jpg")) is None

    def test_undecodable_input_returns_none_never_raises(self, tmp_path):
        bad = tmp_path / "broken.png"
        bad.write_bytes(b"not an image at all")
        assert _flatten_image_alpha(bad) is None


class TestNormalizeFlattensAlpha:
    def test_rgba_png_returns_jpeg_mime_matching_the_bytes(self, tmp_path):
        """The returned mime must describe the RE-ENCODED bytes (JPEG), not the input
        format — an ``image/png`` label on JPEG bytes is exactly the mismatch that
        fails provider-side decoding."""
        shot = _rgba_png(tmp_path / "shot.png")
        path, mime, err = _normalize_to_supported_image(shot, "image/png")
        assert err is None and mime == "image/jpeg" and path != shot
        assert path.read_bytes().startswith(b"\xff\xd8\xff")
        path.unlink()

    def test_opaque_jpeg_passthrough_unchanged(self, tmp_path):
        img = _opaque_jpeg(tmp_path / "photo.jpg")
        before = img.read_bytes()
        path, mime, err = _normalize_to_supported_image(img, "image/jpeg")
        assert err is None and mime == "image/jpeg" and path == img
        assert path.read_bytes() == before
        assert not list(_vision_cache_dir().glob("flattened_*"))

    def test_opaque_png_passthrough_no_reencode(self, tmp_path):
        img = _opaque_png(tmp_path / "photo.png")
        before = img.read_bytes()
        path, mime, err = _normalize_to_supported_image(img, "image/png")
        assert err is None and mime == "image/png" and path == img
        assert path.read_bytes() == before
        assert not list(_vision_cache_dir().glob("flattened_*"))

    def test_corrupt_alpha_candidate_fails_open_to_original_bytes(self, tmp_path):
        """Any flattening failure returns the ORIGINAL bytes, never an error — the
        embed path must not regress for images the backends may still accept."""
        bad = tmp_path / "broken.png"
        bad.write_bytes(b"definitely not a PNG")
        path, mime, err = _normalize_to_supported_image(bad, "image/png")
        assert err is None and mime == "image/png" and path == bad


class TestFlattenAlphaConfig:
    def test_flatten_alpha_false_disables_flattening(self, tmp_path):
        _write_config("vision:\n  flatten_alpha: false\n")
        assert _vision_flatten_alpha_enabled() is False
        shot = _rgba_png(tmp_path / "shot.png")
        before = shot.read_bytes()
        path, mime, err = _normalize_to_supported_image(shot, "image/png")
        assert err is None and mime == "image/png" and path == shot
        assert path.read_bytes() == before

    def test_key_unset_defaults_to_flatten_enabled(self, tmp_path):
        """Default is ON (fail-open): a config.yaml without the key still flattens."""
        _write_config("vision:\n  embed_target_bytes: 1048576\n")
        assert _vision_flatten_alpha_enabled() is True
        shot = _rgba_png(tmp_path / "shot.png")
        path, mime, err = _normalize_to_supported_image(shot, "image/png")
        assert err is None and mime == "image/jpeg" and path != shot
        path.unlink()

    def test_no_config_at_all_still_flattens(self, tmp_path):
        assert _vision_flatten_alpha_enabled() is True
        shot = _rgba_png(tmp_path / "shot.png")
        path, mime, err = _normalize_to_supported_image(shot, "image/png")
        assert err is None and mime == "image/jpeg" and path != shot
        path.unlink()
