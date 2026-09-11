"""Tests for the bundled xKiro model-provider profiles."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.error import URLError


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _catalog(*items):
    return {"object": "list", "data": list(items)}


def _profile(name):
    from providers import get_provider_profile

    profile = get_provider_profile(name)
    assert profile is not None
    return profile


def test_xkiro_profiles_are_registered_with_shared_credentials():
    profile = _profile("xkiro")
    anthropic = _profile("xkiro-anthropic")

    assert profile.name == "xkiro"
    assert profile.display_name == "xKiro"
    assert profile.api_mode == "chat_completions"
    assert profile.base_url == "https://api.xkiro.com/v1"
    assert profile.env_vars == ("XKIRO_API_KEY", "XKIRO_BASE_URL")
    assert profile.default_aux_model == "qwen/qwen3.5-flash:free"
    assert profile.prefer_live_model_catalog is True
    assert profile.public_model_catalog is True
    assert profile.pricing_cache_ttl_seconds == 300

    assert anthropic.display_name == "xKiro (Anthropic)"
    assert anthropic.api_mode == "anthropic_messages"
    assert anthropic.base_url == profile.base_url
    assert anthropic.env_vars == profile.env_vars
    assert anthropic.default_aux_model == "qwen/qwen3.5-flash:free"
    assert anthropic.preserve_anthropic_model_id is True


def test_xkiro_aliases_and_generic_cli_registration():
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS

    assert _profile("xkiro-ai").name == "xkiro"
    assert _profile("xkiro-claude").name == "xkiro-anthropic"

    config = PROVIDER_REGISTRY["xkiro"]
    assert config.auth_type == "api_key"
    assert config.api_key_env_vars == ("XKIRO_API_KEY",)
    assert config.base_url_env_var == "XKIRO_BASE_URL"
    assert config.inference_base_url == "https://api.xkiro.com/v1"
    assert {"XKIRO_API_KEY", "XKIRO_BASE_URL"}.issubset(OPTIONAL_ENV_VARS)
    assert {entry.slug for entry in CANONICAL_PROVIDERS}.issuperset({"xkiro", "xkiro-anthropic"})


def test_xkiro_fetch_models_uses_safe_credentialed_opener_and_custom_base():
    requests = []

    def fake_open(request, timeout):
        requests.append((request, timeout))
        return _Response(
            _catalog(
                {"id": "openai/gpt-5.6-luna"},
                {"id": "x-ai/grok-4.6"},
                {"id": "stability/image-model", "modality": "image"},
                {"id": 123},
                {"name": "missing-id"},
            )
        )

    profile = _profile("xkiro")
    with patch("hermes_cli.urllib_security.open_credentialed_url", side_effect=fake_open):
        models = profile.fetch_models(api_key="test-key", base_url="https://proxy.example/v1", timeout=3)

    assert models == ["openai/gpt-5.6-luna", "x-ai/grok-4.6"]
    request, timeout = requests[0]
    assert request.full_url == "https://proxy.example/v1/models"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("User-agent").startswith("hermes-cli/")
    assert timeout == 3


def test_xkiro_catalog_normalizes_vendor_origin_without_v1():
    profile = _profile("xkiro-anthropic")
    requests = []

    def fake_open(request, timeout):
        requests.append(request)
        return _Response(_catalog({"id": "anthropic/claude-sonnet-4.6"}))

    with patch("hermes_cli.urllib_security.open_credentialed_url", side_effect=fake_open):
        assert profile.fetch_models(base_url="https://api.xkiro.com") == ["anthropic/claude-sonnet-4.6"]

    assert requests[0].full_url == "https://api.xkiro.com/v1/models"


def test_xkiro_anthropic_exposes_all_chat_models():
    profile = _profile("xkiro-anthropic")
    payload = _catalog(
        {"id": "anthropic/claude-sonnet-5"},
        {"id": "qwen/qwen3.5-flash:free"},
        {"id": "openai/gpt-5.6-luna"},
    )

    with patch("hermes_cli.urllib_security.open_credentialed_url", return_value=_Response(payload)):
        models = profile.fetch_models(api_key="test-key")

    assert models == [
        "anthropic/claude-sonnet-5",
        "qwen/qwen3.5-flash:free",
        "openai/gpt-5.6-luna",
    ]


def test_xkiro_setup_prefers_complete_live_catalog_over_registry_subset():
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.model_setup_flows import _api_key_provider_model_list

    profile = _profile("xkiro")
    complete = ["anthropic/claude-sonnet-5", "openai/gpt-5.6-luna", "qwen/free-model"]
    with (
        patch.object(profile, "fetch_models", return_value=complete) as fetch,
        patch("hermes_cli.model_setup_flows._models_dev_merged", return_value=["registry/subset"]),
    ):
        result = _api_key_provider_model_list(
            "xkiro",
            PROVIDER_REGISTRY["xkiro"],
            "test-key",
            "XKIRO_API_KEY",
            "https://proxy.example/v1",
        )

    assert result == complete
    fetch.assert_called_once_with(api_key="test-key", base_url="https://proxy.example/v1")


def test_xkiro_empty_live_catalog_falls_back_to_registry():
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.model_setup_flows import _api_key_provider_model_list

    profile = _profile("xkiro")
    with (
        patch.object(profile, "fetch_models", return_value=[]),
        patch("hermes_cli.model_setup_flows._models_dev_merged", return_value=["registry/fallback"]),
    ):
        result = _api_key_provider_model_list(
            "xkiro",
            PROVIDER_REGISTRY["xkiro"],
            "test-key",
            "XKIRO_API_KEY",
            "https://api.xkiro.com/v1",
        )

    assert result == ["registry/fallback"]


def test_xkiro_public_catalog_populates_picker_without_a_key():
    from hermes_cli import models

    seen = []

    def fake_open(request, timeout):
        seen.append(request)
        return _Response(_catalog({"id": "openai/gpt-5.6-luna"}, {"id": "deepseek/free"}))

    with (
        patch.object(models, "_api_key_credentials", return_value=("", "https://api.xkiro.com/v1")),
        patch("hermes_cli.urllib_security.open_credentialed_url", side_effect=fake_open),
    ):
        result = models.provider_model_ids("xkiro", force_refresh=True)

    assert result == ["openai/gpt-5.6-luna", "deepseek/free"]
    assert seen[0].get_header("Authorization") is None


def test_xkiro_fetch_models_returns_none_on_failure_or_malformed_payload():
    profile = _profile("xkiro")

    with patch("hermes_cli.urllib_security.open_credentialed_url", side_effect=URLError("blocked")):
        assert profile.fetch_models(timeout=1) is None
    with patch(
        "hermes_cli.urllib_security.open_credentialed_url",
        return_value=_Response({"object": "list", "data": "not-a-list"}),
    ):
        assert profile.fetch_models(timeout=1) is None


def test_xkiro_pricing_normalizes_documented_per_million_units():
    profile = _profile("xkiro")
    payload = _catalog(
        {
            "id": "openai/gpt-5.6-luna",
            "pricing": {
                "currency": "USD",
                "unit": "per_1m_tokens",
                "input": 0.1,
                "output": 0.6,
                "cache_read": 0.05,
            },
        },
        {
            "id": "qwen/qwen3.5-flash:free",
            "pricing": {"currency": "USD", "unit": "per_1m_tokens", "input": 0, "output": 0},
        },
        {
            "id": "wrong-unit",
            "pricing": {"currency": "USD", "unit": "per_token", "input": 1, "output": 1},
        },
        {
            "id": "wrong-currency",
            "pricing": {"currency": "EUR", "unit": "per_1m_tokens", "input": 1, "output": 1},
        },
    )

    with patch("hermes_cli.urllib_security.open_credentialed_url", return_value=_Response(payload)):
        pricing = profile.fetch_model_pricing()

    assert pricing == {
        "openai/gpt-5.6-luna": {
            "prompt": "0.0000001",
            "completion": "0.0000006",
            "input_cache_read": "0.00000005",
        },
        "qwen/qwen3.5-flash:free": {"prompt": "0", "completion": "0"},
    }


def test_xkiro_pricing_hook_uses_shared_cache_and_cached_only_path():
    from hermes_cli import models_pricing

    payload = _catalog(
        {
            "id": "openai/gpt-5.6-luna",
            "pricing": {"currency": "USD", "unit": "per_1m_tokens", "input": 0.1, "output": 0.6},
        }
    )
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    models_pricing._pricing_provider_cache_keys.clear()

    with (
        patch("hermes_cli.models._api_key_credentials", return_value=("test-key", "https://api.xkiro.com/v1")),
        patch(
            "hermes_cli.urllib_security.open_credentialed_url", return_value=_Response(payload)
        ) as opener,
    ):
        fetched = models_pricing.get_pricing_for_provider("xkiro", force_refresh=True)
        cached = models_pricing.get_pricing_for_provider("xkiro", cached_only=True)

    assert fetched == cached == {
        "openai/gpt-5.6-luna": {"prompt": "0.0000001", "completion": "0.0000006"}
    }
    assert opener.call_count == 1
    assert models_pricing.pricing_cache_scope("xkiro").startswith("https://api.xkiro.com")


def test_xkiro_pricing_cache_scope_tracks_credential_rotation():
    from hermes_cli import models_pricing

    payload = _catalog({
        "id": "openai/gpt-5.6-luna",
        "pricing": {"currency": "USD", "unit": "per_1m_tokens", "input": 0.1, "output": 0.6},
    })
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    models_pricing._pricing_provider_cache_keys.clear()

    with (
        patch("hermes_cli.models._api_key_credentials", return_value=("old-key", "https://api.xkiro.com/v1")),
        patch("hermes_cli.urllib_security.open_credentialed_url", return_value=_Response(payload)),
    ):
        models_pricing.get_pricing_for_provider("xkiro", force_refresh=True)
        old_scope = models_pricing.pricing_cache_scope("xkiro")
        assert models_pricing.get_pricing_for_provider("xkiro", cached_only=True)

    with patch(
        "hermes_cli.models._api_key_credentials", return_value=("new-key", "https://api.xkiro.com/v1")
    ):
        new_scope = models_pricing.pricing_cache_scope("xkiro")
        configured_scope = models_pricing.pricing_cache_scope(
            "xkiro",
            current_provider="xkiro",
            current_base_url="https://stale.example/v1",
        )
        assert models_pricing.get_pricing_for_provider("xkiro", cached_only=True) == {}

    assert new_scope != old_scope
    assert configured_scope == new_scope


def test_xkiro_picker_formats_prices_without_inventing_sales():
    from hermes_cli.inventory import _apply_pricing

    rows: list[dict[str, Any]] = [
        {
            "slug": "xkiro",
            "models": ["paid", "free", "sale"],
        }
    ]
    raw = {
        "paid": {"prompt": "0.0000001", "completion": "0.0000006"},
        "free": {"prompt": "0", "completion": "0"},
        "sale": {
            "prompt": "0.0000001",
            "completion": "0.0000006",
            "original": {"prompt": "0.0000002", "completion": "0.0000012"},
        },
    }

    with patch("hermes_cli.models_pricing.get_pricing_for_provider", return_value=raw):
        _apply_pricing(rows)

    pricing: dict[str, Any] = rows[0]["pricing"]
    assert pricing["paid"] == {"input": "$0.10", "output": "$0.60", "cache": None, "free": False}
    assert pricing["free"] == {"input": "free", "output": "free", "cache": None, "free": True}
    assert pricing["sale"] == {
        "input": "$0.10",
        "output": "$0.60",
        "cache": None,
        "free": False,
    }


def test_xkiro_anthropic_wire_keeps_vendor_model_id_and_uses_bearer_auth():
    from agent.anthropic_adapter import build_anthropic_kwargs
    from agent.anthropic_endpoints import _requires_bearer_auth

    kwargs = build_anthropic_kwargs(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=1024,
        reasoning_config=None,
        base_url="https://api.xkiro.com/v1",
        preserve_model_id=_profile("xkiro-anthropic").preserve_anthropic_model_id,
    )

    assert kwargs["model"] == "anthropic/claude-sonnet-4.6"
    assert _requires_bearer_auth("https://api.xkiro.com/v1") is True
    assert _requires_bearer_auth("https://api.xkiro.com.evil.example/v1") is False
    assert _requires_bearer_auth("https://evil.example/api.xkiro.com/v1") is False


def test_profile_flag_preserves_vendor_id_for_custom_xkiro_base_url():
    from agent.anthropic_adapter import build_anthropic_kwargs

    profile = _profile("xkiro-anthropic")

    kwargs = build_anthropic_kwargs(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=1024,
        reasoning_config=None,
        base_url="https://proxy.example/v1",
        preserve_model_id=profile.preserve_anthropic_model_id,
    )

    assert kwargs["model"] == "anthropic/claude-sonnet-4.6"


def test_auxiliary_anthropic_adapter_uses_profile_wire_mode_and_preserves_model_id():
    from agent.auxiliary_client import _ResolveRequest, _wrap_transport, AnthropicAuxiliaryClient

    model = "anthropic/claude-sonnet-4.6"
    req = _ResolveRequest(
        provider="xkiro-anthropic",
        original_provider="xkiro-anthropic",
        model=model,
        async_mode=False,
        raw_codex=False,
        explicit_base_url="https://proxy.example/v1",
        explicit_api_key="test-key",
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task="title_generation",
    )
    fake_native_client = SimpleNamespace()
    normalized = SimpleNamespace(content="ok", tool_calls=None, reasoning=None, finish_reason="stop")
    fake_transport = SimpleNamespace(normalize_response=lambda response, **kwargs: normalized)

    with (
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=fake_native_client),
        patch("agent.anthropic_adapter.create_anthropic_message", return_value=SimpleNamespace(usage=None)) as create,
        patch("agent.transports.get_transport", return_value=fake_transport),
    ):
        wrapped = _wrap_transport(req, SimpleNamespace(), model, "https://proxy.example/v1", "test-key")
        assert isinstance(wrapped, AnthropicAuxiliaryClient)
        wrapped.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])

    assert create.call_args.args[1]["model"] == model


def test_xkiro_anthropic_resolves_through_real_runtime_and_auxiliary_paths(tmp_path, monkeypatch):
    from agent.auxiliary_client import AnthropicAuxiliaryClient, resolve_provider_client
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XKIRO_API_KEY", "test-key")
    monkeypatch.delenv("XKIRO_BASE_URL", raising=False)

    runtime = resolve_runtime_provider(
        requested="xkiro-anthropic", target_model="anthropic/claude-sonnet-4.6"
    )
    assert runtime["provider"] == "xkiro-anthropic"
    assert runtime["base_url"] == "https://api.xkiro.com/v1"
    assert runtime["api_mode"] == "anthropic_messages"

    client, model = resolve_provider_client("xkiro-anthropic")
    try:
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert model == "qwen/qwen3.5-flash:free"
        assert client.chat.completions._preserve_model_id is True
    finally:
        if client is not None:
            client.close()
