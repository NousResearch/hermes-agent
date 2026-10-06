"""Unit tests for tools.computer_use.vision_routing.

``should_route_capture_to_aux_vision`` decides whether a ``computer_use`` capture is returned as a multimodal
envelope (the main model reads the pixels) or pre-analysed through ``auxiliary.vision`` (the main model sees text).
It is the negation of the shared native-tool-result gate, so these cases drive the real gate and patch only the
capability lookups underneath it. The end-to-end regression for #24015 lives in
``tests/tools/test_computer_use_capture_routing.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.computer_use.vision_routing import should_route_capture_to_aux_vision


def _route(provider, model, cfg, *, catalog, provider_takes_media, veto=False):
    with patch("agent.image_routing._lookup_supports_vision", return_value=catalog), \
         patch("tools.vision_tools._supports_media_in_tool_results", return_value=provider_takes_media), \
         patch("tools.vision_tools._profile_rejects_tool_media", return_value=veto):
        return should_route_capture_to_aux_vision(provider, model, cfg)


class TestRouteDecision:
    def test_explicit_aux_backend_wins_over_a_vision_main_model(self):
        """#24015: a configured auxiliary.vision backend is the de-facto image route in auto mode."""
        cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": "google/gemini-2.5-flash"}}}
        assert _route("anthropic", "claude-opus-4-5", cfg, catalog=True, provider_takes_media=True) is True

    def test_provider_auto_plus_explicit_model_counts_as_an_aux_backend(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto", "model": "claude-3-haiku"}}}
        assert _route("anthropic", "claude-opus-4-5", cfg, catalog=True, provider_takes_media=True) is True

    def test_vision_main_model_without_aux_backend_stays_native(self):
        assert _route("anthropic", "claude-opus-4-5", None, catalog=True, provider_takes_media=True) is False

    def test_catalog_text_only_model_on_a_media_provider_goes_to_aux(self):
        """deepseek-v3.2 on OpenRouter: the provider carries tool-result media, the model cannot see. Routing it
        native made the agent backstop swap the screenshot for a 'switch to a vision model' error."""
        assert _route("openrouter", "deepseek/deepseek-v3.2", {}, catalog=False, provider_takes_media=True) is True

    def test_unknown_provider_and_model_fail_closed_to_aux(self):
        assert _route("exotic-provider", "exotic-model", {}, catalog=None, provider_takes_media=False) is True

    def test_image_input_mode_text_keeps_pixels_out(self):
        cfg = {"agent": {"image_input_mode": "text"}}
        assert _route("anthropic", "claude-opus-4-5", cfg, catalog=True, provider_takes_media=True) is True

    def test_image_input_mode_native_keeps_a_catalog_unknown_model_native(self):
        """A proxy alias the catalog does not know: native mode is the user's explicit word for it."""
        cfg = {"agent": {"image_input_mode": "native"}}
        assert _route("anthropic", "my-proxy-claude", cfg, catalog=None, provider_takes_media=True) is False

    def test_declared_supports_vision_keeps_a_local_vlm_native(self):
        """Local/custom VLMs declare support in config; the real lookup reads it (no catalog patch)."""
        cfg = {"model": {"default": "Qwen3.6-35B-A3B-local-vlm", "provider": "omlx", "supports_vision": True}}
        with patch("tools.vision_tools._profile_rejects_tool_media", return_value=False):
            assert should_route_capture_to_aux_vision("custom", "Qwen3.6-35B-A3B-local-vlm", cfg) is False

    def test_profile_veto_routes_a_vision_model_to_aux(self):
        assert _route("xiaomi", "mimo-v2.5", {}, catalog=True, provider_takes_media=False, veto=True) is True


class TestGateAgreementWithVisionAnalyze:
    """Capture, vision_analyze, browser screenshots and MCP images share one predicate (#115248), so the lane never
    depends on which tool produced the image."""

    @pytest.mark.parametrize("cfg,catalog,provider_takes_media,veto", [
        ({}, True, False, False),                                   # catalog vision off the provider whitelist
        ({}, False, True, False),                                   # text-only model on a media provider
        ({"agent": {"image_input_mode": "text"}}, True, True, False),
        ({"agent": {"image_input_mode": "native"}}, None, True, False),
        ({}, True, False, True),                                    # profile veto
    ])
    def test_capture_route_is_the_shared_gate(self, cfg, catalog, provider_takes_media, veto):
        from tools.vision_tools import _native_tool_result_images

        with patch("agent.image_routing._lookup_supports_vision", return_value=catalog), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=provider_takes_media), \
             patch("tools.vision_tools._profile_rejects_tool_media", return_value=veto):
            assert should_route_capture_to_aux_vision("p", "m", cfg) is (not _native_tool_result_images("p", "m", cfg))
