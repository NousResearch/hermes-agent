"""The native-embed size cap is configurable.

``vision_analyze``'s native fast path shrinks every image to ``_EMBED_TARGET_BYTES``
(default 256 KB) so that a screenshot re-sent on every later turn stays cheap. That
default halves a 150 DPI document page to ~620 px wide, which is where table digits
stop being legible, so the cap is now ``HERMES_VISION_EMBED_TARGET_BYTES`` →
``auxiliary.vision.embed_target_bytes`` → 256 KB (and the same shape for the long-edge
cap). These tests pin the resolution order, the floor, the ceiling, and that the
default is unchanged when nothing is set.
"""

import pytest

import tools.vision_tools as vt


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_VISION_EMBED_TARGET_BYTES", raising=False)
    monkeypatch.delenv("HERMES_VISION_EMBED_MAX_DIMENSION", raising=False)
    monkeypatch.setattr(vt, "_cfg_auxiliary", lambda *keys, default=None: default)


class TestEmbedTargetBytes:
    def test_default_unchanged(self):
        assert vt._resolve_embed_target_bytes() == 256 * 1024
        assert vt._EMBED_TARGET_BYTES_DEFAULT == 256 * 1024

    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setattr(vt, "_cfg_auxiliary",
                            lambda *keys, default=None: 512 * 1024 if keys[-1] == "embed_target_bytes" else default)
        monkeypatch.setenv("HERMES_VISION_EMBED_TARGET_BYTES", str(2 * 1024 * 1024))
        assert vt._resolve_embed_target_bytes() == 2 * 1024 * 1024

    def test_config_key_used_without_env(self, monkeypatch):
        monkeypatch.setattr(vt, "_cfg_auxiliary",
                            lambda *keys, default=None: 1024 * 1024 if keys[-1] == "embed_target_bytes" else default)
        assert vt._resolve_embed_target_bytes() == 1024 * 1024

    @pytest.mark.parametrize("raw", ["0", "1024", "-5", "lots", ""])
    def test_bad_or_tiny_values_fall_back(self, monkeypatch, raw):
        monkeypatch.setenv("HERMES_VISION_EMBED_TARGET_BYTES", raw)
        assert vt._resolve_embed_target_bytes() == 256 * 1024

    def test_capped_at_hard_ceiling(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_EMBED_TARGET_BYTES", str(10 * vt._MAX_BASE64_BYTES))
        assert vt._resolve_embed_target_bytes() == vt._MAX_BASE64_BYTES


class TestEmbedMaxDimension:
    def test_default_unchanged(self):
        assert vt._resolve_embed_max_dimension() == 1568

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_EMBED_MAX_DIMENSION", "2200")
        assert vt._resolve_embed_max_dimension() == 2200

    @pytest.mark.parametrize("raw", ["0", "63", "wide"])
    def test_bad_values_fall_back(self, monkeypatch, raw):
        monkeypatch.setenv("HERMES_VISION_EMBED_MAX_DIMENSION", raw)
        assert vt._resolve_embed_max_dimension() == 1568

    def test_capped_at_anthropic_limit(self, monkeypatch):
        monkeypatch.setenv("HERMES_VISION_EMBED_MAX_DIMENSION", "20000")
        assert vt._resolve_embed_max_dimension() == 8000


class TestLadderHonoursRaisedCap:
    """The point of the knob: a dense page that the default halves is sent full-size once
    the cap is raised. Pure Pillow, no LLM."""

    def test_dense_page_not_halved_under_raised_cap(self, tmp_path, monkeypatch):
        PIL = pytest.importorskip("PIL")
        from PIL import Image, ImageDraw, ImageFont
        import random
        random.seed(7)
        # 1241x1755 (A4 at 150 DPI) of dense body text, like a rendered agreement page:
        # ~300-400 KB as JPEG q85, over 256 KB even at q50, comfortably under 2 MB.
        try:
            font = ImageFont.load_default(size=13)
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
        img = Image.new("RGB", (1241, 1755), "white")
        d = ImageDraw.Draw(img)
        words = ["ordinary", "hours", "37", "settlement", "7.5", "period", "APS6", "$97,412", "clause", "flex"]
        for y in range(40, 1720, 16):
            line = " ".join(random.choice(words) for _ in range(22))
            d.text((40, y), line, fill=(20, 20, 20), font=font)
        path = tmp_path / "page.png"
        img.save(path, "PNG")

        scale_default: dict = {}
        vt._resize_image_for_vision(path, max_base64_bytes=256 * 1024, max_dimension=1568,
                                    scale_out=scale_default, force_jpeg=True)
        assert scale_default, "fixture must be dense enough that the default cap downscales it"
        assert scale_default["new_width"] < 1241

        # The long-edge cap alone halves an A4 page (1755 > 1568, and the ladder only halves).
        scale_dim_only: dict = {}
        vt._resize_image_for_vision(path, max_base64_bytes=2 * 1024 * 1024, max_dimension=1568,
                                    scale_out=scale_dim_only, force_jpeg=True)
        assert scale_dim_only["new_height"] == 877

        scale_raised: dict = {}
        vt._resize_image_for_vision(path, max_base64_bytes=2 * 1024 * 1024, max_dimension=2000,
                                    scale_out=scale_raised, force_jpeg=True)
        assert not scale_raised, "with both caps raised the page must go out at its rendered size"
