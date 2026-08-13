"""Tests for the native-vision fast path inside vision_analyze.

When the active main model supports native vision AND the provider supports
image content inside tool-result messages, ``_handle_vision_analyze`` skips
the auxiliary LLM and returns a multimodal envelope so the main model sees
the pixels directly on its next turn.
"""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from unittest.mock import patch


from tools.vision_tools import (
    _build_native_vision_tool_result,
    _handle_vision_analyze,
    _supports_media_in_tool_results,
    _vision_analyze_native,
)


# Minimal valid 1x1 PNG bytes.
_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

# PNG-shaped but undecodable; resolver/native fast path must reject it.
_CORRUPT_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFElEQVR4nGP8z8Dwn4EIwESJ5gAAVQ4CH1evYJQAAAAASUVORK5CYII="
)


def _animated_gif_bytes(colors, *, size=(4, 4)):
    from PIL import Image

    frames = [Image.new("RGB", size, color) for color in colors]
    encoded = BytesIO()
    frames[0].save(
        encoded,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return encoded.getvalue()


def _track_validated_frame_loads(monkeypatch):
    from PIL import ImageSequence

    loaded_frames = []
    real_iterator = ImageSequence.Iterator

    class TrackedFrame:
        def __init__(self, frame, frame_number):
            self.frame = frame
            self.frame_number = frame_number
            self.width = frame.width
            self.height = frame.height

        def load(self):
            loaded_frames.append(self.frame_number)
            return self.frame.load()

    def tracking_iterator(image):
        for frame_number, frame in enumerate(real_iterator(image), start=1):
            yield TrackedFrame(frame, frame_number)

    monkeypatch.setattr(ImageSequence, "Iterator", tracking_iterator)
    return loaded_frames


# ─── _supports_media_in_tool_results ─────────────────────────────────────────


class TestSupportsMediaInToolResults:
    def test_anthropic_native_yes(self):
        assert _supports_media_in_tool_results("anthropic", "claude-opus-4-6") is True

    def test_openrouter_yes(self):
        assert _supports_media_in_tool_results("openrouter", "anthropic/claude-opus-4.6") is True


    def test_empty_provider_no(self):
        assert _supports_media_in_tool_results("", "anything") is False
        assert _supports_media_in_tool_results(None, "anything") is False  # type: ignore[arg-type]


# ─── _build_native_vision_tool_result ────────────────────────────────────────


class TestBuildNativeVisionToolResult:
    def test_envelope_shape(self):
        env = _build_native_vision_tool_result(
            image_url="/tmp/foo.png",
            question="what does it say?",
            image_data_url="data:image/png;base64,XYZ",
            image_size_bytes=1024,
        )
        assert env["_multimodal"] is True
        assert isinstance(env["content"], list)
        assert len(env["content"]) == 2
        assert env["content"][0]["type"] == "text"
        assert env["content"][1]["type"] == "image_url"
        assert env["content"][1]["image_url"]["url"] == "data:image/png;base64,XYZ"
        assert "what does it say?" in env["content"][0]["text"]
        assert "Image attached natively" in env["text_summary"]

    def test_no_question_omits_question_section(self):
        env = _build_native_vision_tool_result(
            image_url="/tmp/foo.png",
            question="",
            image_data_url="data:image/png;base64,XYZ",
            image_size_bytes=512,
        )
        text = env["content"][0]["text"]
        assert "Question:" not in text
        assert "Image loaded" in text


# ─── _vision_analyze_native ──────────────────────────────────────────────────


class TestVisionAnalyzeNative:
    def test_local_file_returns_multimodal_envelope(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(_TINY_PNG)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(img), "what is this?")
        )
        assert isinstance(result, dict)
        assert result.get("_multimodal") is True
        parts = result["content"]
        assert any(p.get("type") == "image_url" for p in parts)
        assert any(p.get("type") == "text" for p in parts)
        url = next(p["image_url"]["url"] for p in parts if p.get("type") == "image_url")
        assert url.startswith("data:image/")

    def test_truncated_supported_image_is_rejected_before_embedding(self, tmp_path):
        """A valid header must not let partially downloaded bytes poison history."""
        pytest = __import__("pytest")
        Image = pytest.importorskip(
            "PIL.Image", reason="Pillow is required for full raster decode validation"
        )

        truncated = tmp_path / "truncated.png"
        truncated.write_bytes(_TINY_PNG[:-22])

        # This is the production failure shape: header parsing succeeds, but a
        # complete decode fails after the partial download is read.
        with Image.open(truncated) as image:
            assert image.format == "PNG"
            with pytest.raises(OSError, match="truncated"):
                image.load()

        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(truncated), "describe")
        )

        assert isinstance(result, str), "corrupt image must not return a multimodal envelope"
        payload = json.loads(result)
        assert payload["success"] is False
        assert "decode" in payload["error"].lower()

    def test_truncated_animated_gif_frame_is_rejected_before_embedding(
        self, tmp_path, monkeypatch
    ):
        """Every frame must decode before an animated raster enters history."""
        pytest = __import__("pytest")
        pytest.importorskip(
            "PIL.Image", reason="Pillow is required for full raster decode validation"
        )
        from PIL import Image, ImageSequence

        truncated = tmp_path / "truncated.gif"
        truncated.write_bytes(_animated_gif_bytes(["red", "blue"])[:-3])
        monkeypatch.setattr(
            "tools.vision_tools._VISION_MAX_VALIDATED_FRAME_COUNT", 2
        )
        monkeypatch.setattr(
            "tools.vision_tools._VISION_MAX_VALIDATED_AGGREGATE_PIXELS", 32
        )

        with Image.open(truncated) as image:
            assert image.format == "GIF"
            assert getattr(image, "n_frames", 1) == 2
            with pytest.raises(OSError, match="truncated"):
                for frame in ImageSequence.Iterator(image):
                    frame.load()

        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(truncated), "describe")
        )

        assert isinstance(result, str), "corrupt animation must not return a multimodal envelope"
        payload = json.loads(result)
        assert payload["success"] is False
        assert "decode" in payload["error"].lower()

    def test_animation_over_frame_validation_limit_is_rejected(
        self, tmp_path, monkeypatch
    ):
        """A small animation is rejected before an excess frame is decoded."""
        pytest = __import__("pytest")
        pytest.importorskip(
            "PIL.Image", reason="Pillow is required for full raster decode validation"
        )

        animation = tmp_path / "too-many-frames.gif"
        animation.write_bytes(_animated_gif_bytes(["red", "green", "blue"]))
        monkeypatch.setattr(
            "tools.vision_tools._VISION_MAX_VALIDATED_FRAME_COUNT", 2
        )
        loaded_frames = _track_validated_frame_loads(monkeypatch)

        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(animation), "describe")
        )

        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload["success"] is False
        assert "frame 3" in payload["error"].lower()
        assert "maximum 2" in payload["error"].lower()
        assert loaded_frames == [1, 2]

    def test_animation_over_aggregate_pixel_validation_limit_is_rejected(
        self, tmp_path, monkeypatch
    ):
        """Aggregate decoded pixels are bounded independently of file size."""
        pytest = __import__("pytest")
        pytest.importorskip(
            "PIL.Image", reason="Pillow is required for full raster decode validation"
        )

        animation = tmp_path / "too-many-pixels.gif"
        animation.write_bytes(_animated_gif_bytes(["red", "blue"]))
        monkeypatch.setattr(
            "tools.vision_tools._VISION_MAX_VALIDATED_AGGREGATE_PIXELS", 31
        )
        loaded_frames = _track_validated_frame_loads(monkeypatch)

        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(animation), "describe")
        )

        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload["success"] is False
        assert "aggregate decoded pixel" in payload["error"].lower()
        assert "32" in payload["error"]
        assert "maximum 31" in payload["error"].lower()
        assert loaded_frames == [1]

    def test_file_url_scheme_resolves(self, tmp_path):
        img = tmp_path / "t.png"
        img.write_bytes(_TINY_PNG)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(f"file://{img}", "?")
        )
        assert isinstance(result, dict)
        assert result.get("_multimodal") is True

    def test_corrupt_png_rejected_before_native_embed(self, tmp_path):
        """Header-only PNG bytes must not enter conversation history."""
        img = tmp_path / "bad.png"
        img.write_bytes(_CORRUPT_PNG)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(img), "what is this?")
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "multimodal" not in parsed
        assert "recognized image" in parsed.get("error", "")

    def test_oversized_image_resized_under_embed_cap(self, tmp_path):
        """Regression for the wedged-session incident (May 2026).

        A vision tool-result image is baked into conversation history and
        re-sent on every subsequent turn.  The native fast path must
        proactively resize down to the history-reuse embed cap BEFORE
        embedding, not just at the 20 MB hard ceiling.  Skips if Pillow
        isn't available (resize is a no-op).
        """
        pytest = __import__("pytest")
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed — proactive resize is a no-op")

        from tools.vision_tools import _EMBED_TARGET_BYTES

        # Noisy PNG that base64-encodes to well over 5 MB (won't compress much).
        big = tmp_path / "big.png"
        Image.effect_noise((2600, 2600), 80).convert("RGB").save(big, format="PNG")
        assert big.stat().st_size * 4 // 3 > 5 * 1024 * 1024, "test image not big enough"

        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(big), "describe")
        )
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            p["image_url"]["url"]
            for p in result["content"]
            if p.get("type") == "image_url"
        )
        assert len(url) <= _EMBED_TARGET_BYTES, (
            f"embedded image {len(url) / 1024:.0f} KB exceeds embed cap "
            f"{_EMBED_TARGET_BYTES / 1024:.0f} KB — would bloat every later turn"
        )

    def test_embed_caps_are_sized_for_history_reuse(self):
        """Native embeds ride every later turn, so caps must stay well below
        the Anthropic 5 MB / 8000px reject limits (#92699)."""
        from tools.vision_tools import _EMBED_MAX_DIMENSION, _EMBED_TARGET_BYTES

        assert _EMBED_TARGET_BYTES <= 512 * 1024
        assert _EMBED_MAX_DIMENSION <= 2048


# ─── _handle_vision_analyze fast-path gating ─────────────────────────────────


class TestHandleVisionAnalyzeFastPath:
    """Verify the dispatcher chooses fast-path vs aux-LLM correctly."""

    def test_vision_capable_main_model_uses_fast_path(self, tmp_path, monkeypatch):
        """Main model supports native vision → fast path returns multimodal."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        # Set runtime override so the handler thinks we're on opus@openrouter
        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("openrouter", "anthropic/claude-opus-4.6")
        try:
            # Mock decide_image_input_mode to always return "native" so the
            # fast path fires regardless of model-catalog state in CI.
            with patch(
                "agent.image_routing.decide_image_input_mode",
                return_value="native",
            ):
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert isinstance(result, dict), \
            f"Expected multimodal envelope, got {type(result).__name__}: {str(result)[:200]}"
        assert result.get("_multimodal") is True


    def test_fast_path_disabled_for_unsupported_provider(self, tmp_path, monkeypatch):
        """Even with vision-capable model, unknown provider → fall through."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "anthropic/claude-opus-4.6")
        try:
            with patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel):
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert not (isinstance(result, dict) and result.get("_multimodal") is True), \
            "Fast path fired for unknown provider; should have fallen through"

    def test_supports_vision_override_bypasses_provider_allowlist(self, tmp_path):
        """supports_vision=true enables the fast path on an unlisted provider."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"model": {"supports_vision": True}},
            ), patch(
                "tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel,
            ) as mock_aux:
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert isinstance(result, dict) and result.get("_multimodal") is True
        mock_aux.assert_not_called()

    def test_text_mode_wins_over_supports_vision_override(self, tmp_path):
        """Explicit text routing blocks the fast path even with supports_vision."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={
                    "agent": {"image_input_mode": "text"},
                    "model": {"supports_vision": True},
                },
            ), patch(
                "tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel,
            ) as mock_aux:
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert isinstance(result, str)
        assert json.loads(result) == {"sentinel": "aux-path"}
        mock_aux.assert_called_once()
