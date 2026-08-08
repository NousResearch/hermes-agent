"""Tests for tools/image_generation_tool.py — FAL multi-model support.

Covers the pure logic of the new wrapper: catalog integrity, the three size
families (image_size_preset / aspect_ratio / gpt_literal), the supports
whitelist, default merging, GPT quality override, and model resolution
fallback. Does NOT exercise fal_client submission — that's covered by
tests/tools/test_managed_media_gateways.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_tool():
    """Fresh import of tools.image_generation_tool per test."""
    import importlib
    import tools.image_generation_tool as mod
    return importlib.reload(mod)


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------

class TestFalCatalog:
    """Every FAL_MODELS entry must have a consistent shape."""

    def test_default_model_is_klein(self, image_tool):
        assert image_tool.DEFAULT_MODEL == "fal-ai/flux-2/klein/9b"


    def test_nano_banana_2_in_catalog(self, image_tool):
        meta = image_tool.FAL_MODELS["fal-ai/nano-banana-2"]

        # Invariants (not value snapshots): NB2 is an aspect-ratio family
        # with an edit endpoint whose ref cap matches FAL's published limit.
        assert meta["size_style"] == "aspect_ratio"
        assert meta["edit_endpoint"].startswith("fal-ai/nano-banana-2")
        assert meta["max_reference_images"] >= 1
        assert meta["edit_supports"] >= {"prompt", "image_urls"}

    def test_all_entries_have_required_keys(self, image_tool):
        required = {
            "display", "speed", "strengths", "price",
            "size_style", "sizes", "defaults", "supports", "upscale",
        }
        for mid, meta in image_tool.FAL_MODELS.items():
            missing = required - set(meta.keys())
            assert not missing, f"{mid} missing required keys: {missing}"


    def test_only_flux2_pro_upscales_by_default(self, image_tool):
        """Upscaling should default to False for all new models to preserve
        the <1s / fast-render value prop. Only flux-2-pro stays True for
        backward-compat with the previous default."""
        for mid, meta in image_tool.FAL_MODELS.items():
            if mid == "fal-ai/flux-2-pro":
                assert meta["upscale"] is True, \
                    "flux-2-pro should keep upscale=True for backward-compat"
            else:
                assert meta["upscale"] is False, \
                    f"{mid} should default to upscale=False"


# ---------------------------------------------------------------------------
# Payload building — three size families
# ---------------------------------------------------------------------------

class TestImageSizePresetFamily:
    """Flux, z-image, qwen, recraft, ideogram all use preset enum sizes."""

    def test_klein_landscape_uses_preset(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hello", "landscape")
        assert p["image_size"] == "landscape_16_9"
        assert "aspect_ratio" not in p


    def test_klein_portrait_uses_preset(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hello", "portrait")
        assert p["image_size"] == "portrait_16_9"


class TestAspectRatioFamily:
    """Nano-banana uses aspect_ratio enum, NOT image_size."""

    def test_nano_banana_landscape_uses_aspect_ratio(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana-pro", "hello", "landscape")
        assert p["aspect_ratio"] == "16:9"
        assert "image_size" not in p


    def test_nano_banana_portrait_uses_aspect_ratio(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana-pro", "hello", "portrait")
        assert p["aspect_ratio"] == "9:16"

    def test_nano_banana_2_uses_aspect_ratio_and_flash_defaults(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana-2", "hello", "landscape")

        assert p["aspect_ratio"] == "16:9"
        assert p["resolution"] == "1K"
        assert p["limit_generations"] is True
        assert "image_size" not in p

    def test_nano_banana_2_allows_thinking_level(self, image_tool):
        p = image_tool._build_fal_payload(
            "fal-ai/nano-banana-2",
            "hello",
            "square",
            overrides={"thinking_level": "minimal"},
        )

        assert p["thinking_level"] == "minimal"


class TestGptLiteralFamily:
    """GPT-Image 1.5 uses literal size strings."""

    def test_gpt_landscape_is_literal(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hello", "landscape")
        assert p["image_size"] == "1536x1024"


    def test_gpt_portrait_is_literal(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hello", "portrait")
        assert p["image_size"] == "1024x1536"


class TestGptImage2Presets:
    """GPT Image 2 uses preset enum sizes (not literal strings like 1.5).
    Mapped to 4:3 variants so we stay above the 655,360 min-pixel floor
    (16:9 presets at 1024x576 = 589,824 would be rejected)."""

    def test_gpt2_landscape_uses_4_3_preset(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-2", "hello", "landscape")
        assert p["image_size"] == "landscape_4_3"


    def test_gpt2_strips_byok_and_unsupported_overrides(self, image_tool):
        """openai_api_key (BYOK) is deliberately not in supports — all users
        route through shared FAL billing. guidance_scale/num_inference_steps
        aren't in the model's API surface either."""
        p = image_tool._build_fal_payload(
            "fal-ai/gpt-image-2", "hi", "square",
            overrides={
                "openai_api_key": "sk-...",
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
            },
        )
        assert "openai_api_key" not in p
        assert "guidance_scale" not in p
        assert "num_inference_steps" not in p

    def test_gpt2_strips_seed_even_if_passed(self, image_tool):
        # seed isn't in the GPT Image 2 API surface either.
        p = image_tool._build_fal_payload("fal-ai/gpt-image-2", "hi", "square", seed=42)
        assert "seed" not in p


# ---------------------------------------------------------------------------
# Supports whitelist — the main safety property
# ---------------------------------------------------------------------------

class TestSupportsFilter:
    """No model should receive keys outside its `supports` set."""

    def test_payload_keys_are_subset_of_supports_for_all_models(self, image_tool):
        for mid, meta in image_tool.FAL_MODELS.items():
            payload = image_tool._build_fal_payload(mid, "test", "landscape", seed=42)
            unsupported = set(payload.keys()) - meta["supports"]
            assert not unsupported, \
                f"{mid} payload has unsupported keys: {unsupported}"


    def test_nano_banana_never_gets_image_size(self, image_tool):
        # Common bug: translator accidentally setting both image_size and aspect_ratio.
        p = image_tool._build_fal_payload("fal-ai/nano-banana-pro", "hi", "landscape", seed=1)
        assert "image_size" not in p
        assert p["aspect_ratio"] == "16:9"


# ---------------------------------------------------------------------------
# Default merging
# ---------------------------------------------------------------------------

class TestDefaults:
    """Model-level defaults should carry through unless overridden."""

    def test_klein_default_steps_is_4(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hi", "square")
        assert p["num_inference_steps"] == 4


    def test_none_override_does_not_replace_default(self, image_tool):
        """None values from caller should be ignored (use default)."""
        p = image_tool._build_fal_payload(
            "fal-ai/flux-2-pro", "hi", "square",
            overrides={"num_inference_steps": None},
        )
        assert p["num_inference_steps"] == 50


# ---------------------------------------------------------------------------
# GPT-Image quality is pinned to medium (not user-configurable)
# ---------------------------------------------------------------------------

class TestGptQualityPinnedToMedium:
    """GPT-Image quality is baked into the FAL_MODELS defaults at 'medium'
    and cannot be overridden via config. Pinning keeps Nous Portal billing
    predictable across all users."""

    def test_gpt_payload_always_has_medium_quality(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hi", "square")
        assert p["quality"] == "medium"


    def test_resolve_gpt_quality_function_is_gone(self, image_tool):
        """The _resolve_gpt_quality() helper was removed — quality is now
        a static default, not a runtime lookup."""
        assert not hasattr(image_tool, "_resolve_gpt_quality"), (
            "_resolve_gpt_quality should not exist — quality is pinned"
        )


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution:

    def test_no_config_falls_back_to_default(self, image_tool):
        with patch("hermes_cli.config.load_config", return_value={}):
            mid, meta = image_tool._resolve_fal_model()
        assert mid == "fal-ai/flux-2/klein/9b"


    def test_config_wins_over_env_var(self, image_tool, monkeypatch):
        monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/z-image/turbo")
        with patch("hermes_cli.config.load_config",
                   return_value={"image_gen": {"model": "fal-ai/nano-banana-pro"}}):
            mid, _ = image_tool._resolve_fal_model()
        assert mid == "fal-ai/nano-banana-pro"


# ---------------------------------------------------------------------------
# Aspect ratio handling
# ---------------------------------------------------------------------------

class TestAspectRatioNormalization:

    def test_invalid_aspect_defaults_to_landscape(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hi", "cinemascope")
        assert p["image_size"] == "landscape_16_9"


    def test_empty_aspect_defaults_to_landscape(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hi", "")
        assert p["image_size"] == "landscape_16_9"


# ---------------------------------------------------------------------------
# Schema + registry integrity
# ---------------------------------------------------------------------------

class TestRegistryIntegration:

    def test_schema_exposes_expected_agent_params(self, image_tool):
        """The agent-facing schema exposes the unified text+image surface:
        prompt (required), aspect_ratio, and the image-to-image inputs
        image_url + reference_image_urls. Model selection stays a user-level
        config choice, never an agent-level arg."""
        props = image_tool.IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
        assert set(props.keys()) == {
            "prompt", "aspect_ratio", "image_url", "reference_image_urls",
        }
        assert image_tool.IMAGE_GENERATE_SCHEMA["parameters"]["required"] == ["prompt"]

    def test_aspect_ratio_enum_is_three_values(self, image_tool):
        enum = image_tool.IMAGE_GENERATE_SCHEMA["parameters"]["properties"]["aspect_ratio"]["enum"]
        assert set(enum) == {"landscape", "square", "portrait"}


# ---------------------------------------------------------------------------
# Managed gateway 4xx translation
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _MockHttpxError(Exception):
    """Simulates httpx.HTTPStatusError which exposes .response.status_code."""
    def __init__(self, status_code: int, message: str = "Bad Request"):
        super().__init__(message)
        self.response = _MockResponse(status_code)


class TestExtractHttpStatus:
    """Status-code extraction should work across exception shapes."""

    def test_extracts_from_response_attr(self, image_tool):
        exc = _MockHttpxError(403)
        assert image_tool._extract_http_status(exc) == 403


    def test_response_attr_without_status_code_returns_none(self, image_tool):
        class OddResponse:
            pass
        exc = Exception("weird")
        exc.response = OddResponse()  # type: ignore[attr-defined]
        assert image_tool._extract_http_status(exc) is None


class TestManagedGatewayErrorTranslation:
    """4xx from the Nous managed gateway should be translated to a user-actionable message."""

    def test_4xx_translates_to_value_error_with_remediation(self, image_tool, monkeypatch):
        """403 from managed gateway → ValueError mentioning FAL_KEY + hermes tools."""
        from unittest.mock import MagicMock

        # Simulate: managed mode active, managed submit raises 4xx.
        managed_gateway = MagicMock()
        managed_gateway.gateway_origin = "https://fal-queue-gateway.example.com"
        managed_gateway.nous_user_token = "test-token"
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway",
                            lambda: managed_gateway)

        bad_request = _MockHttpxError(403, "Forbidden")
        mock_managed_client = MagicMock()
        mock_managed_client.submit.side_effect = bad_request
        monkeypatch.setattr(image_tool, "_get_managed_fal_client",
                            lambda gw: mock_managed_client)

        with pytest.raises(ValueError) as exc_info:
            image_tool._submit_fal_request("fal-ai/nano-banana-pro", {"prompt": "x"})

        msg = str(exc_info.value)
        assert "fal-ai/nano-banana-pro" in msg
        assert "403" in msg
        assert "FAL_KEY" in msg
        assert "hermes tools" in msg
        # Original exception chained for debugging
        assert exc_info.value.__cause__ is bad_request


    def test_non_http_exception_from_managed_bubbles_up(self, image_tool, monkeypatch):
        """Connection errors, timeouts, etc. from managed mode aren't 4xx —
        they should bubble up unchanged so callers can retry or diagnose."""
        from unittest.mock import MagicMock

        managed_gateway = MagicMock()
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway",
                            lambda: managed_gateway)

        conn_error = ConnectionError("network down")
        mock_managed_client = MagicMock()
        mock_managed_client.submit.side_effect = conn_error
        monkeypatch.setattr(image_tool, "_get_managed_fal_client",
                            lambda gw: mock_managed_client)

        with pytest.raises(ConnectionError):
            image_tool._submit_fal_request("fal-ai/flux-2-pro", {"prompt": "x"})


class TestKreaModelNormalization:
    """Native ``krea-2-*`` detection for managed Krea routing."""

    def test_native_models_detected(self, image_tool):
        for mid in ("krea-2-medium", "krea-2-large", "krea-2-medium-turbo"):
            assert image_tool.is_krea_model(mid) is True
            assert image_tool._normalize_krea_model(mid) == mid


    def test_non_krea_models_are_not_krea(self, image_tool):
        for mid in ("fal-ai/flux-2/klein/9b", "fal-ai/nano-banana-pro", None, "", 123):
            assert image_tool.is_krea_model(mid) is False
            assert image_tool._normalize_krea_model(mid) is None


class TestManagedKreaRouting:
    """`_maybe_route_managed_krea` only fires for Krea models in managed mode."""

    def test_no_route_when_model_not_krea(self, image_tool, monkeypatch):
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: None)
        monkeypatch.setattr(
            image_tool, "_read_configured_image_model", lambda: "fal-ai/flux-2/klein/9b"
        )
        assert image_tool._maybe_route_managed_krea("p", "square") is None


    def test_routes_native_krea_model_to_krea_plugin_in_managed_mode(
        self, image_tool, monkeypatch
    ):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import json as _json

        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: None)
        monkeypatch.setattr(
            image_tool,
            "_read_configured_image_model",
            lambda: "krea-2-large",
        )
        import plugins.image_gen.krea as krea_mod

        monkeypatch.setattr(
            krea_mod,
            "_resolve_managed_krea_gateway",
            lambda: SimpleNamespace(
                vendor="krea",
                gateway_origin="https://krea-gateway.example.com",
                nous_user_token="tok",
                managed_mode=True,
            ),
        )

        fake_provider = MagicMock()
        fake_provider.generate.return_value = {"success": True, "image": "/tmp/x.png"}
        monkeypatch.setattr(
            "agent.image_gen_registry.get_provider", lambda name: fake_provider
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None
        )

        out = image_tool._maybe_route_managed_krea("a cat", "portrait")
        assert out is not None
        assert _json.loads(out)["success"] is True
        kwargs = fake_provider.generate.call_args.kwargs
        assert kwargs["model"] == "krea-2-large"
        assert kwargs["prompt"] == "a cat"
        assert kwargs["aspect_ratio"] == "portrait"


class TestFalKreaCatalog:
    """Krea 2 on FAL remains in the FAL picker for FAL-billed users."""

    def test_fal_krea_models_in_fal_catalog(self, image_tool):
        assert "fal-ai/krea/v2/medium/text-to-image" in image_tool.FAL_MODELS
        assert "fal-ai/krea/v2/large/text-to-image" in image_tool.FAL_MODELS
