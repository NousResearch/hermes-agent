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


# ─── _supports_media_in_tool_results ─────────────────────────────────────────


class TestSupportsMediaInToolResults:
    def test_anthropic_native_yes(self):
        assert _supports_media_in_tool_results("anthropic", "claude-opus-4-6") is True

    def test_openrouter_yes(self):
        assert _supports_media_in_tool_results("openrouter", "anthropic/claude-opus-4.6") is True

    def test_nous_yes(self):
        assert _supports_media_in_tool_results("nous", "anthropic/claude-sonnet-4.6") is True

    def test_openai_chat_yes(self):
        assert _supports_media_in_tool_results("openai", "gpt-5.4") is True

    def test_openai_codex_yes(self):
        assert _supports_media_in_tool_results("openai-codex", "gpt-5-codex") is True

    def test_gemini_3_yes(self):
        assert _supports_media_in_tool_results("google", "gemini-3-flash-preview") is True

    def test_gemini_2_no(self):
        assert _supports_media_in_tool_results("google", "gemini-2.5-pro") is False

    def test_unknown_provider_conservative_no(self):
        assert _supports_media_in_tool_results("brand-new-provider", "any-model") is False

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

    def test_missing_file_returns_error_string(self, tmp_path):
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(tmp_path / "nope.png"), "?")
        )
        # tool_error returns a JSON string, not the multimodal envelope
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "Invalid image source" in parsed.get("error", "")

    def test_empty_image_url_returns_error(self):
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native("", "?")
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "image_url is required" in parsed.get("error", "")

    def test_file_url_scheme_resolves(self, tmp_path):
        img = tmp_path / "t.png"
        img.write_bytes(_TINY_PNG)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(f"file://{img}", "?")
        )
        assert isinstance(result, dict)
        assert result.get("_multimodal") is True

    def test_oversized_image_resized_under_embed_cap(self, tmp_path):
        """Regression for the wedged-session incident (May 2026).

        A vision tool-result image is baked into conversation history and
        re-sent on every subsequent turn.  Anthropic rejects any single
        base64 image over 5 MB with a 400, and immutable history means the
        bad bytes can't be cleared by retrying — the session is permanently
        wedged.  The native fast path must proactively resize down to the
        embed cap (well under 5 MB) BEFORE embedding, not just at the 20 MB
        hard ceiling.  Skips if Pillow isn't available (resize is a no-op).
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
            f"embedded image {len(url) / 1024 / 1024:.1f} MB exceeds embed cap "
            f"{_EMBED_TARGET_BYTES / 1024 / 1024:.0f} MB — would wedge sessions on Anthropic"
        )


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

    def test_non_vision_main_model_falls_through_to_aux(self, tmp_path, monkeypatch):
        """Non-vision main model → fast path skipped, aux LLM path attempted."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("openrouter", "qwen/qwen3-coder")
        try:
            with patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel):
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert not (isinstance(result, dict) and result.get("_multimodal") is True), \
            "Fast path fired for non-vision model; should have fallen through to aux LLM"

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
