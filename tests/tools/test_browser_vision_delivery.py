"""Tests for browser_vision delivery hint to messaging gateways.

When a user on Telegram (or any messaging platform) asks the agent for a
screenshot, the gateway only forwards the file if the agent's reply
contains a ``MEDIA:<path>`` tag (see ``gateway/platforms/base.py`` —
``extract_local_files`` / ``extract_images``).

Previously, ``browser_vision`` returned ``screenshot_path`` as a JSON field
and relied on the tool description ("include MEDIA:<screenshot_path> in
your response") to nudge the model. In practice models routinely
paraphrased the analysis but dropped the path, leaving the user with no
image while the agent believed the screenshot was sent.

These tests verify that the tool result now includes a ready-to-paste
``delivery_tag`` plus an explicit ``delivery_instruction``, and that
suspiciously small (likely-blank) screenshots are flagged via
``size_warning``.
"""

import os


def _read_browser_tool_source() -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.join(base, "tools", "browser_tool.py")) as f:
        return f.read()


class TestBrowserVisionDeliveryHint:
    """tools/browser_tool.py — browser_vision() must surface MEDIA: tag."""

    def test_response_includes_delivery_tag_field(self):
        src = _read_browser_tool_source()
        assert '"delivery_tag": media_tag' in src, (
            "browser_vision response_data must include `delivery_tag` so "
            "models can echo the MEDIA:<path> verbatim into their reply"
        )

    def test_media_tag_uses_correct_prefix(self):
        src = _read_browser_tool_source()
        assert 'media_tag = f"MEDIA:{screenshot_path}"' in src, (
            "delivery_tag must use the MEDIA:<path> form recognised by "
            "gateway/platforms/base.py extract_local_files"
        )

    def test_response_includes_delivery_instruction(self):
        src = _read_browser_tool_source()
        assert '"delivery_instruction"' in src, (
            "browser_vision response_data must include a "
            "`delivery_instruction` string telling the model the tag is "
            "required for the user to actually receive the screenshot"
        )


class TestBrowserVisionBlankScreenshotWarning:
    """tools/browser_tool.py — flag near-empty screenshots."""

    def test_size_warning_threshold_present(self):
        src = _read_browser_tool_source()
        assert "if _shot_bytes < 5_000:" in src, (
            "browser_vision must warn when the captured screenshot is "
            "smaller than 5KB (typically a blank/unrendered page) so the "
            "agent can re-take instead of delivering blank to the user"
        )

    def test_size_warning_field_emitted(self):
        src = _read_browser_tool_source()
        assert '"size_warning"' in src, (
            "size_warning field must surface in response_data when the "
            "screenshot is suspiciously small"
        )
