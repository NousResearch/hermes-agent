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


# ─── _handle_vision_analyze fast-path gating ─────────────────────────────────


class TestHandleVisionAnalyzeFastPath:
    """Verify the dispatcher chooses fast-path vs aux-LLM correctly."""

    def test_native_mode_with_supported_transport_uses_fast_path(self, tmp_path):
        """Explicit native mode + known transport returns multimodal."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("openrouter", "anthropic/claude-opus-4.6")
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"image_input_mode": "native"}},
            ), patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel) as mock_aux:
                result = asyncio.get_event_loop().run_until_complete(
                    _handle_vision_analyze({"image_url": str(img), "question": "?"})
                )
        finally:
            clear_runtime_main()

        assert isinstance(result, dict), (
            f"Expected multimodal envelope, got {type(result).__name__}: {str(result)[:200]}"
        )
        assert result.get("_multimodal") is True
        mock_aux.assert_not_called()

    def test_native_mode_with_unsupported_transport_falls_through(self, tmp_path):
        """Explicit native mode still respects the transport gate."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "opaque-model")
        try:
            with (
                patch(
                    "hermes_cli.config.load_config",
                    return_value={"agent": {"image_input_mode": "native"}},
                ),
                patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel) as mock_aux,
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    _handle_vision_analyze({"image_url": str(img), "question": "?"})
                )
        finally:
            clear_runtime_main()

        assert isinstance(result, str)
        assert json.loads(result) == {"sentinel": "aux-path"}
        mock_aux.assert_called_once()

    def test_supports_vision_bypasses_transport_gate(self, tmp_path):
        """supports_vision=True enables fast path even on unknown providers."""
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
            ), patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel) as mock_aux:
                result = asyncio.get_event_loop().run_until_complete(
                    _handle_vision_analyze({"image_url": str(img), "question": "?"})
                )
        finally:
            clear_runtime_main()

        assert isinstance(result, dict), (
            f"Expected multimodal envelope, got {type(result).__name__}: {str(result)[:200]}"
        )
        assert result.get("_multimodal") is True
        mock_aux.assert_not_called()

    def test_text_mode_still_blocks_fast_path_when_supports_vision_true(self, tmp_path):
        """Routing mode wins over supports_vision when text mode was chosen."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with (
                patch(
                    "hermes_cli.config.load_config",
                    return_value={
                        "agent": {"image_input_mode": "text"},
                        "model": {"supports_vision": True},
                    },
                ),
                patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel) as mock_aux,
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    _handle_vision_analyze({"image_url": str(img), "question": "?"})
                )
        finally:
            clear_runtime_main()

        assert isinstance(result, str)
        assert json.loads(result) == {"sentinel": "aux-path"}
        mock_aux.assert_called_once()
