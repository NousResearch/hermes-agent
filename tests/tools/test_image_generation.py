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

@pytest.mark.parametrize("variant", ["flare", "sunburst"])
@pytest.mark.parametrize("aspect,size", [
    ("landscape", "landscape_4_3"), ("square", "square_hd"), ("portrait", "portrait_4_3"),
])
def test_image_25_selection_routes_generation_and_edits(image_tool, monkeypatch, variant, aspect, size):
    model = f"openai/gpt-image-2.5/{variant}/text-to-image"
    monkeypatch.setenv("FAL_IMAGE_MODEL", model)
    monkeypatch.setenv("FAL_KEY", "test-key")
    selected, meta = image_tool._resolve_fal_model()
    assert selected == model
    refs = [f"https://example.com/{i}.png" for i in range(17)]
    for sources, endpoint in (([], model), (refs, f"openai/gpt-image-2.5/{variant}/edit")):
        actual, payload = image_tool._prepare_fal_request(
            selected, meta, "a cup", aspect, 42, {"guidance_scale": 9}, sources,
        )
        assert actual == endpoint
        assert payload["quality"] == "medium"
        assert payload["image_size"] == size
        assert "seed" not in payload and "guidance_scale" not in payload
        assert payload.get("image_urls", []) == sources[:16]
    assert meta["upscale"] is False


class TestFalCatalog:
    """Every FAL_MODELS entry must have a consistent shape."""




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
            # Most edit endpoints take an `image_urls` list; entries with a
            # singular image key (Kling Image v3) declare edit_image_param.
            image_param = meta.get("edit_image_param") or "image_urls"
            assert image_param in meta["edit_supports"], \
                f"{mid} edit_supports must allow {image_param}"
            cap = meta.get("max_reference_images")
            assert isinstance(cap, int) and cap > 0, \
                f"{mid} needs a positive max_reference_images"






# ---------------------------------------------------------------------------
# Payload building — three size families
# ---------------------------------------------------------------------------

class TestImageSizePresetFamily:
    """Flux, z-image, qwen, recraft, ideogram all use preset enum sizes."""

    def test_klein_landscape_uses_preset(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/flux-2/klein/9b", "hello", "landscape")
        assert p["image_size"] == "landscape_16_9"
        assert "aspect_ratio" not in p




class TestAspectRatioFamily:
    """Nano-banana uses aspect_ratio enum, NOT image_size."""

    def test_nano_banana_landscape_uses_aspect_ratio(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/nano-banana-pro", "hello", "landscape")
        assert p["aspect_ratio"] == "16:9"
        assert "image_size" not in p






class TestGptLiteralFamily:
    """GPT-Image 1.5 uses literal size strings."""

    def test_gpt_landscape_is_literal(self, image_tool):
        p = image_tool._build_fal_payload("fal-ai/gpt-image-1.5", "hello", "landscape")
        assert p["image_size"] == "1536x1024"




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




# ---------------------------------------------------------------------------
# Default merging
# ---------------------------------------------------------------------------

class TestDefaults:
    """Model-level defaults should carry through unless overridden."""



    def test_none_override_does_not_replace_default(self, image_tool):
        """None values from caller should be ignored (use default)."""
        p = image_tool._build_fal_payload(
            "fal-ai/flux-2-pro", "hi", "square",
            overrides={"num_inference_steps": None},
        )
        assert p["num_inference_steps"] == image_tool.FAL_MODELS["fal-ai/flux-2-pro"]["defaults"]["num_inference_steps"]


# ---------------------------------------------------------------------------
# GPT-Image quality is pinned to medium (not user-configurable)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution:

    def test_no_config_falls_back_to_default(self, image_tool):
        with patch("hermes_cli.config.load_config", return_value={}):
            mid, meta = image_tool._resolve_fal_model()
        assert mid == image_tool.DEFAULT_MODEL


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




# ---------------------------------------------------------------------------
# Schema + registry integrity
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Managed gateway 4xx translation
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else __import__("json").dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _MockHttpxError(Exception):
    """Simulates httpx.HTTPStatusError which exposes .response.status_code."""
    def __init__(self, status_code: int, message: str = "Bad Request", payload=None):
        super().__init__(message)
        self.response = _MockResponse(status_code, payload)


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

    @pytest.fixture(autouse=True)
    def _fal_client_stub(self, image_tool, monkeypatch):
        # These tests drive a mocked managed client; the module-global loader
        # must not demand the optional fal extra (a version-pinned metadata
        # check under lazy installs) on the way there.
        monkeypatch.setattr(image_tool, "fal_client", object())

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
        # Original exception chained for debugging
        assert exc_info.value.__cause__ is bad_request

    def test_billing_meter_error_is_preserved_instead_of_called_model_unavailable(
        self, image_tool, monkeypatch
    ):
        """Portal billing configuration is the root cause, not a missing model."""
        from unittest.mock import MagicMock

        managed_gateway = MagicMock()
        managed_gateway.gateway_origin = "https://fal-queue-gateway.example.com"
        managed_gateway.nous_user_token = "test-token"
        monkeypatch.setattr(
            image_tool, "_resolve_managed_fal_gateway", lambda: managed_gateway
        )
        payload = {
            "error": {
                "code": "BILLING_ERROR",
                "message": "Charge authorization failed",
                "details": {
                    "upstreamPayload": {
                        "code": "unsupported_pricing_meter",
                        "error": "Unsupported resolver usage meter",
                    }
                },
            }
        }
        billing_error = _MockHttpxError(409, payload=payload)
        mock_managed_client = MagicMock()
        mock_managed_client.submit.side_effect = billing_error
        monkeypatch.setattr(
            image_tool, "_get_managed_fal_client", lambda gw: mock_managed_client
        )

        with pytest.raises(ValueError) as exc_info:
            image_tool._submit_fal_request(
                "openai/gpt-image-2.5/flare/text-to-image", {"prompt": "x"}
            )

        msg = str(exc_info.value)
        assert "Charge authorization failed" in msg
        assert "BILLING_ERROR" in msg
        assert "unsupported_pricing_meter" in msg
        assert "may not yet be enabled" not in msg


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

    @staticmethod
    def _rate_limited(retry_after):
        return _MockHttpxError(429, "Too Many Requests", payload={"error": {
            "code": "RATE_LIMIT_EXCEEDED", "message": "Rate limit exceeded.", "retryAfter": retry_after}})

    def test_short_429_is_retried_once_under_a_fresh_idempotency_key(self, image_tool, monkeypatch):
        """A gateway 429 with a short retryAfter is waited out and resubmitted once (new key)."""
        from unittest.mock import MagicMock

        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: MagicMock())
        mock_managed_client = MagicMock()
        mock_managed_client.submit.side_effect = [self._rate_limited(0), "handle"]
        monkeypatch.setattr(image_tool, "_get_managed_fal_client", lambda gw: mock_managed_client)

        assert image_tool._submit_fal_request("fal-ai/gpt-image-2", {"prompt": "x"}) == "handle"

        keys = [call.kwargs["headers"]["x-idempotency-key"] for call in mock_managed_client.submit.call_args_list]
        assert len(keys) == 2 and keys[0] != keys[1]

    def test_long_429_is_reported_as_a_rate_limit_not_a_missing_model(self, image_tool, monkeypatch):
        """A 429 beyond the retry cap names the rate limit; agents must not be told to switch models."""
        from unittest.mock import MagicMock

        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: MagicMock())
        mock_managed_client = MagicMock()
        mock_managed_client.submit.side_effect = self._rate_limited(120)
        monkeypatch.setattr(image_tool, "_get_managed_fal_client", lambda gw: mock_managed_client)

        with pytest.raises(ValueError) as exc_info:
            image_tool._submit_fal_request("fal-ai/gpt-image-2", {"prompt": "x"})

        msg = str(exc_info.value)
        assert "120" in msg
        assert "may not yet be enabled" not in msg
        assert mock_managed_client.submit.call_count == 1


class TestKreaModelNormalization:
    """Native ``krea-2-*`` detection for managed Krea routing."""

    def test_native_models_detected(self, image_tool):
        for mid in ("krea-2-medium", "krea-2-large", "krea-2-medium-turbo"):
            assert image_tool._normalize_krea_model(mid) == mid


    def test_non_krea_models_are_not_krea(self, image_tool):
        for mid in ("fal-ai/flux-2/klein/9b", "fal-ai/nano-banana-pro", None, "", 123):
            assert image_tool._normalize_krea_model(mid) is None


class TestManagedKreaRouting:
    """`_maybe_route_managed_model` only fires for Krea / Portal models in managed mode."""

    def test_no_route_when_model_not_krea(self, image_tool, monkeypatch):
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: None)
        monkeypatch.setattr(
            image_tool, "_read_configured_image_model", lambda: "fal-ai/flux-2/klein/9b"
        )
        assert image_tool._maybe_route_managed_model("p", "square") is None


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

        out = image_tool._maybe_route_managed_model("a cat", "portrait")
        assert out is not None
        assert _json.loads(out)["success"] is True
        kwargs = fake_provider.generate.call_args.kwargs
        assert kwargs["model"] == "krea-2-large"
        assert kwargs["prompt"] == "a cat"
        assert kwargs["aspect_ratio"] == "portrait"


class TestManagedPortalRouting:
    """A Portal model under the managed selection reaches the Portal plugin — never FAL."""

    def _fake_registry(self, monkeypatch, fake_provider):
        monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: fake_provider)
        monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None)

    def test_routes_portal_model_to_nous_plugin(self, image_tool, monkeypatch):
        import json as _json
        from unittest.mock import MagicMock

        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "nous")
        monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: "openai/gpt-5.4-image-2")
        fake_provider = MagicMock(display_name="Nous Portal")
        fake_provider.generate.return_value = {"success": True, "image": "/tmp/x.png"}
        self._fake_registry(monkeypatch, fake_provider)

        out = image_tool._maybe_route_managed_model("a cat", "square")

        assert _json.loads(out)["success"] is True
        assert fake_provider.generate.call_args.kwargs["model"] == "openai/gpt-5.4-image-2"

    def test_portal_model_without_plugin_errors_instead_of_billing_fal(self, image_tool, monkeypatch):
        import json as _json

        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "nous")
        monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: "openai/gpt-5.4-image-2")
        self._fake_registry(monkeypatch, None)

        out = image_tool._maybe_route_managed_model("a cat", "square")

        assert out is not None and _json.loads(out)["success"] is False




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
