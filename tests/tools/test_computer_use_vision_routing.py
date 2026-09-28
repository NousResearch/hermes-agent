"""Unit tests for tools.computer_use.vision_routing.

Cover the small ``should_route_capture_to_aux_vision`` policy helper that
decides whether a captured screenshot from ``computer_use(action='capture')``
should be returned as a multimodal envelope (main model handles vision
natively) or pre-analysed via the ``auxiliary.vision`` pipeline so the
main model only sees text.

The companion end-to-end regression for #24015 lives in
``tests/tools/test_computer_use_capture_routing.py``; this file pins the
unit contract of the helper in isolation so behaviour does not regress
silently if the surrounding ``computer_use`` plumbing is refactored.
"""

from __future__ import annotations

from unittest.mock import patch



# ---------------------------------------------------------------------------
# _explicit_aux_vision_override
# ---------------------------------------------------------------------------

class TestExplicitAuxVisionOverride:
    """Mirror agent.image_routing — config detection must agree across paths."""




    def test_returns_false_when_vision_block_missing(self):
        from tools.computer_use.vision_routing import _explicit_aux_vision_override
        cfg = {"auxiliary": {"compression": {"provider": "openai"}}}
        assert _explicit_aux_vision_override(cfg) is False


    def test_returns_true_for_provider_auto_plus_explicit_model(self):
        """``provider: auto`` + an explicit model still counts as override."""
        from tools.computer_use.vision_routing import _explicit_aux_vision_override
        cfg = {
            "auxiliary": {
                "vision": {"provider": "auto", "model": "claude-3-haiku"},
            }
        }
        assert _explicit_aux_vision_override(cfg) is True



# ---------------------------------------------------------------------------
# should_route_capture_to_aux_vision
# ---------------------------------------------------------------------------

class TestRouteDecision:
    """End-to-end policy: explicit override > tool-result support > vision caps."""

    def test_explicit_override_routes_to_aux_even_for_vision_main(self):
        """Issue #24015 core repro: explicit aux config must win.

        Even if the main model fully supports vision (Anthropic / Claude),
        an explicit ``auxiliary.vision`` block means the user wants their
        configured backend used. Don't silently bypass it.
        """
        from tools.computer_use import vision_routing

        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash",
                }
            }
        }
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision(
                "anthropic", "claude-opus-4-5", cfg
            ) is True

    def test_non_vision_main_model_routes_to_aux(self):
        """The reported #24015 scenario: tencent/hy3-preview has no vision."""
        from tools.computer_use import vision_routing

        cfg = {"model": {"default": "tencent/hy3-preview", "provider": "openrouter"}}
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=False):
            assert vision_routing.should_route_capture_to_aux_vision(
                "openrouter", "tencent/hy3-preview", cfg
            ) is True

    def test_vision_main_model_no_override_keeps_multimodal(self):
        """Default path: vision-capable main model + no aux override → native."""
        from tools.computer_use import vision_routing

        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision(
                "anthropic", "claude-opus-4-5", None
            ) is False


    def test_user_declared_vision_support_keeps_custom_provider_native(self):
        """Local/custom VLMs use config as their tool-result image escape hatch."""
        from tools.computer_use import vision_routing

        cfg = {
            "model": {
                "default": "Qwen3.6-35B-A3B-local-vlm",
                "provider": "omlx",
                "supports_vision": True,
            }
        }
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=False):
            assert vision_routing.should_route_capture_to_aux_vision(
                "custom", "Qwen3.6-35B-A3B-local-vlm", cfg
            ) is False


    def test_unknown_provider_capabilities_fail_closed(self):
        """When tool-result lookup returns None, route to aux (safe default)."""
        from tools.computer_use import vision_routing

        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=None):
            assert vision_routing.should_route_capture_to_aux_vision(
                "exotic-provider", "exotic-model", {}
            ) is True


    def test_explicit_override_wins_over_unknown_caps(self):
        """Explicit aux config wins regardless of unknown caps elsewhere."""
        from tools.computer_use import vision_routing

        cfg = {"auxiliary": {"vision": {"provider": "openrouter"}}}
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=None):
            assert vision_routing.should_route_capture_to_aux_vision(
                "openrouter", "tencent/hy3-preview", cfg
            ) is True


# ---------------------------------------------------------------------------
# Internal lookups — defensive paths
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------



class TestGateAgreementWithVisionAnalyze:
    """The capture route and the ``vision_analyze`` fast path derive from one predicate, so the lane never
    depends on which tool asked (#115248: deepseek/deepseek-flash went native in one and aux in the other)."""

    def test_catalog_vision_model_off_the_provider_whitelist_stays_native(self):
        from tools.computer_use import vision_routing

        cfg = {"agent": {"image_input_mode": "native"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=False), \
             patch("tools.vision_tools._profile_rejects_tool_media", return_value=False):
            assert vision_routing.should_route_capture_to_aux_vision("deepseek", "deepseek-flash", cfg) is False

    def test_profile_veto_still_routes_a_catalog_vision_model_to_aux(self):
        from tools.computer_use import vision_routing

        with patch("agent.image_routing._lookup_supports_vision", return_value=True), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=False), \
             patch("tools.vision_tools._profile_rejects_tool_media", return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision("xiaomi", "mimo-v2.5", {}) is True

    def test_whitelisted_provider_with_catalog_unknown_model_matches_vision_analyze(self):
        """provider on the tool-result-media whitelist, model absent from models.dev/config (a proxy alias):
        vision_analyze embeds natively, so capture must stay native too instead of demanding a second
        ``supports_vision is True`` from the catalog (review follow-up)."""
        from tools.computer_use import vision_routing
        from tools.vision_tools import _accepts_tool_result_images

        cfg = {"agent": {"image_input_mode": "native"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=None), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=True), \
             patch("tools.vision_tools._profile_rejects_tool_media", return_value=False):
            assert _accepts_tool_result_images("anthropic", "my-proxy-claude", cfg) is True
            assert vision_routing.should_route_capture_to_aux_vision("anthropic", "my-proxy-claude", cfg) is False
