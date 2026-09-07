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


    def test_edit_capable_entries_declare_a_full_edit_contract(self, image_tool):
        """An `edit_endpoint` is useless without the whitelist and the
        reference-image cap that `_build_fal_edit_payload` reads."""
        for mid, meta in image_tool.FAL_MODELS.items():
            if "edit_endpoint" not in meta:
                continue
            assert meta.get("edit_supports"), f"{mid} has edit_endpoint but no edit_supports"
            assert "image_urls" in meta["edit_supports"], \
                f"{mid} edit_supports must allow image_urls"
            cap = meta.get("max_reference_images")
            assert isinstance(cap, int) and cap > 0, \
                f"{mid} needs a positive max_reference_images"


class TestAugust2026Catalog:
    """The Aug 2026 FAL catalog expansion, surfaced in the model picker."""

    NEW_MODELS = (
        "bytedance/seedream/v5/pro/text-to-image",
        "bytedance/seedream/v5/lite/text-to-image",
        "ideogram/v4/instant",
        "ideogram/v4/fast",
        "alibaba/qwen-image-3/text-to-image",
        "microsoft/mai-image-2.5-pro",
        "google/nano-banana-2-lite",
        "fal-ai/recraft/v4.1/text-to-image",
        "fal-ai/nano-banana-2",
    )

    def test_new_models_are_in_the_catalog(self, image_tool):
        missing = [m for m in self.NEW_MODELS if m not in image_tool.FAL_MODELS]
        assert not missing, f"missing from FAL_MODELS: {missing}"

    def test_paired_edit_endpoints_are_wired(self, image_tool):
        expected = {
            "bytedance/seedream/v5/pro/text-to-image": "bytedance/seedream/v5/pro/edit",
            "alibaba/qwen-image-3/text-to-image": "alibaba/qwen-image-3/edit",
            "google/nano-banana-2-lite": "google/nano-banana-2-lite/edit",
            "fal-ai/nano-banana-2": "fal-ai/nano-banana-2/edit",
        }
        for model_id, edit_endpoint in expected.items():
            assert image_tool.FAL_MODELS[model_id]["edit_endpoint"] == edit_endpoint

    def test_text_only_models_declare_no_edit_endpoint(self, image_tool):
        """These have no `/edit` app on FAL; claiming one would 404 mid-request."""
        for model_id in (
            "bytedance/seedream/v5/lite/text-to-image",
            "ideogram/v4/instant",
            "ideogram/v4/fast",
            "microsoft/mai-image-2.5-pro",
            "fal-ai/recraft/v4.1/text-to-image",
        ):
            assert "edit_endpoint" not in image_tool.FAL_MODELS[model_id]

    def test_recraft_v41_omits_keys_its_schema_lacks(self, image_tool):
        """Recraft V4.1 exposes no num_images/output_format/seed — the
        `supports` whitelist has to drop them rather than pass them upstream."""
        p = image_tool._build_fal_payload(
            "fal-ai/recraft/v4.1/text-to-image", "hello", "landscape"
        )
        assert p["image_size"] == "landscape_16_9"
        for absent in ("num_images", "output_format", "seed"):
            assert absent not in p

    def test_nano_banana_2_pins_the_1k_billing_tier(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana-2", "hello", "landscape")
        assert p["resolution"] == "1K"
        assert p["aspect_ratio"] == "16:9"
        assert "image_size" not in p

    def test_nano_banana_2_lite_has_no_resolution_knob(self, image_tool):
        """The lite tier renders at a fixed 1K and declares no `resolution`."""
        meta = image_tool.FAL_MODELS["google/nano-banana-2-lite"]
        assert "resolution" not in meta["supports"]
        assert "resolution" not in meta["defaults"]
        p = image_tool._build_fal_payload("google/nano-banana-2-lite", "hello", "square")
        assert "resolution" not in p
        assert p["aspect_ratio"] == "1:1"

    def test_seedream_lite_uses_documented_size_presets(self, image_tool):
        """Lite accepts FAL's preset enum; custom ImageSize dicts are unnecessary."""
        p = image_tool._build_fal_payload(
            "bytedance/seedream/v5/lite/text-to-image", "hello", "landscape"
        )
        assert p["image_size"] == "landscape_16_9"


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
        """The static registration schema stays minimal — prompt (required)
        + aspect_ratio. Capability args (image_url, reference_image_urls,
        upscale) are added per-model by the dynamic override so sessions
        whose active model can't honor them never see them (#95681 diet).
        Model selection stays a user-level config choice, never an
        agent-level arg."""
        props = image_tool.IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
        assert set(props.keys()) == {"prompt", "aspect_ratio"}
        assert image_tool.IMAGE_GENERATE_SCHEMA["parameters"]["required"] == ["prompt"]
        # The dynamic builder owns the capability args.
        dyn = image_tool._build_dynamic_image_schema()
        assert "parameters" in dyn and "prompt" in dyn["parameters"]["properties"]

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
            assert image_tool._normalize_krea_model(mid) == mid


    def test_non_krea_models_are_not_krea(self, image_tool):
        for mid in ("fal-ai/flux-2/klein/9b", "fal-ai/nano-banana-pro", None, "", 123):
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


# ---------------------------------------------------------------------------
# Opt-in upscale pass
# ---------------------------------------------------------------------------

class _FakeHandle:
    def __init__(self, result):
        self._result = result

    def get(self):
        return self._result


class TestUpscaleOptIn:
    """Explicit ``upscale`` overrides the per-model catalog default."""

    def _run(self, image_tool, monkeypatch, *, model, upscale, upscaler_called):
        monkeypatch.setenv("FAL_IMAGE_MODEL", model)
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)
        monkeypatch.setattr(
            image_tool, "_submit_fal_request",
            lambda endpoint, arguments=None: _FakeHandle(
                {"images": [{"url": "https://fal/native.png", "width": 1024, "height": 768}]}
            ),
        )
        calls = []

        def _fake_upscale(url, prompt):
            calls.append(url)
            return {
                "url": "https://fal/upscaled.png", "width": 2048, "height": 1536,
                "upscaled": True, "upscale_factor": 2,
            }

        monkeypatch.setattr(image_tool, "_upscale_image", _fake_upscale)

        import json as _json
        out = _json.loads(image_tool.image_generate_tool("a cat", upscale=upscale))
        assert out["success"] is True
        assert bool(calls) is upscaler_called
        assert out["upscaled"] is upscaler_called
        expected_url = "https://fal/upscaled.png" if upscaler_called else "https://fal/native.png"
        assert out["image"] == expected_url

    def test_explicit_true_upscales_native_hi_res_model(self, image_tool, monkeypatch):
        """Seedream Lite has upscale=False in the catalog (native 4K) —
        explicit True still wins."""
        self._run(image_tool, monkeypatch,
                  model="bytedance/seedream/v5/lite/text-to-image",
                  upscale=True, upscaler_called=True)

    def test_explicit_false_stays_off(self, image_tool, monkeypatch):
        """Explicit False and the catalog default agree: no upscale."""
        self._run(image_tool, monkeypatch,
                  model="fal-ai/flux-2/klein/9b", upscale=False, upscaler_called=False)

    def test_omitted_keeps_catalog_default_off(self, image_tool, monkeypatch):
        self._run(image_tool, monkeypatch,
                  model="bytedance/seedream/v5/lite/text-to-image",
                  upscale=None, upscaler_called=False)

    def test_omitted_is_off_for_previously_default_on_model(self, image_tool, monkeypatch):
        """flux-2-pro was the old default-on model — now off like the rest."""
        self._run(image_tool, monkeypatch,
                  model="fal-ai/flux-2-pro", upscale=None, upscaler_called=False)

    def test_upscale_failure_falls_back_to_native(self, image_tool, monkeypatch):
        monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/flux-2/klein/9b")
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)
        monkeypatch.setattr(
            image_tool, "_submit_fal_request",
            lambda endpoint, arguments=None: _FakeHandle(
                {"images": [{"url": "https://fal/native.png"}]}
            ),
        )
        monkeypatch.setattr(image_tool, "_upscale_image", lambda url, prompt: None)

        import json as _json
        out = _json.loads(image_tool.image_generate_tool("a cat", upscale=True))
        assert out["success"] is True
        assert out["image"] == "https://fal/native.png"
        assert out["upscaled"] is False


class TestUpscaleDispatchForwarding:
    """The tool handler forwards explicit upscale to plugin providers."""

    def test_dispatch_forwards_upscale(self, image_tool, monkeypatch):
        from unittest.mock import MagicMock
        import json as _json

        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "krea")
        monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: None)
        fake_provider = MagicMock()
        fake_provider.generate.return_value = {"success": True, "image": "/tmp/x.png"}
        monkeypatch.setattr(
            "agent.image_gen_registry.get_provider", lambda name: fake_provider
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None
        )

        out = image_tool._dispatch_to_plugin_provider("a cat", "square", upscale=True)
        assert _json.loads(out)["success"] is True
        assert fake_provider.generate.call_args.kwargs["upscale"] is True

    def test_dispatch_omits_upscale_when_unset(self, image_tool, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "krea")
        monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: None)
        fake_provider = MagicMock()
        fake_provider.generate.return_value = {"success": True, "image": "/tmp/x.png"}
        monkeypatch.setattr(
            "agent.image_gen_registry.get_provider", lambda name: fake_provider
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None
        )

        image_tool._dispatch_to_plugin_provider("a cat", "square")
        assert "upscale" not in fake_provider.generate.call_args.kwargs
