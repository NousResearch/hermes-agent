"""Tests for the native-vision fast path inside camofox_vision.

When the active main model supports native vision, ``camofox_vision`` must
attach the screenshot directly as a multimodal tool-result envelope (so the
main model sees the pixels on its next turn) instead of delegating to the
auxiliary vision LLM.  This mirrors the fast path already present in
``browser_tool.browser_vision`` and keeps the two backends consistent.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import patch

from tools import browser_camofox
from tools.browser_camofox import camofox_vision


# Minimal valid 1x1 PNG bytes.
_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _fake_session():
    return {"tab_id": "tab-123", "user_id": "ashu"}


def _fake_response(content: bytes) -> SimpleNamespace:
    return SimpleNamespace(content=content)


class TestCamofoxVisionFastPath:
    """Verify camofox_vision chooses fast-path vs aux-LLM correctly."""

    def test_native_capable_model_returns_multimodal_envelope(self):
        """Main model supports native vision → screenshot attached directly."""
        with (
            patch.object(browser_camofox, "_get_session", return_value=_fake_session()),
            patch.object(
                browser_camofox, "_camofox_private_page_block", return_value=None
            ),
            patch.object(
                browser_camofox, "_get_raw", return_value=_fake_response(_TINY_PNG)
            ),
            patch(
                "tools.vision_tools._should_use_native_vision_fast_path",
                return_value=True,
            ),
        ):
            result = camofox_vision("what does it say?")

        # Fast path returns the multimodal envelope (dict), NOT a JSON string.
        assert isinstance(result, dict)
        assert result.get("_multimodal") is True
        parts = result["content"]
        assert any(p.get("type") == "image_url" for p in parts)
        assert any(p.get("type") == "text" for p in parts)
        url = next(
            p["image_url"]["url"] for p in parts if p.get("type") == "image_url"
        )
        assert url.startswith("data:image/png;base64,")
        # Screenshot path is surfaced in meta + text_summary for sharing.
        assert "screenshot_path" in result.get("meta", {})
        assert "Screenshot path:" in result.get("text_summary", "")

    def test_native_fast_path_preserves_annotation_context(self):
        """annotate=True → element refs appear in the native envelope text."""
        fake_snapshot = {"snapshot": "[@e1] button 'Submit'\n[@e2] link 'Home'"}
        with (
            patch.object(browser_camofox, "_get_session", return_value=_fake_session()),
            patch.object(
                browser_camofox, "_camofox_private_page_block", return_value=None
            ),
            patch.object(
                browser_camofox, "_get_raw", return_value=_fake_response(_TINY_PNG)
            ),
            patch.object(browser_camofox, "_get", return_value=fake_snapshot),
            patch(
                "tools.vision_tools._should_use_native_vision_fast_path",
                return_value=True,
            ),
        ):
            result = camofox_vision("what buttons are there?", annotate=True)

        assert isinstance(result, dict)
        assert result.get("_multimodal") is True
        text_part = next(
            p["text"] for p in result["content"] if p.get("type") == "text"
        )
        # Annotation context must be present for element interaction.
        assert "Accessibility tree" in text_part
        assert "[@e1]" in text_part
        assert "button 'Submit'" in text_part

    def test_text_only_model_falls_back_to_aux_llm(self):
        """No native vision → delegates to the auxiliary vision LLM."""
        fake_llm_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="a gray page"))]
        )
        with (
            patch.object(browser_camofox, "_get_session", return_value=_fake_session()),
            patch.object(
                browser_camofox, "_camofox_private_page_block", return_value=None
            ),
            patch.object(
                browser_camofox, "_get_raw", return_value=_fake_response(_TINY_PNG)
            ),
            patch(
                "tools.vision_tools._should_use_native_vision_fast_path",
                return_value=False,
            ),
            patch("agent.auxiliary_client.call_llm", return_value=fake_llm_response),
        ):
            result = camofox_vision("what does it say?")

        # Aux path returns a JSON string with the analysis text.
        import json

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed.get("success") is True
        assert parsed.get("analysis") == "a gray page"
