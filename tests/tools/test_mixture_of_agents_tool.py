import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

moa = importlib.import_module("tools.mixture_of_agents_tool")


@pytest.fixture(autouse=True)
def _clear_route_cache():
    moa._reset_route_cache()
    yield
    moa._reset_route_cache()


def _fake_completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, reasoning=None))]
    )


def _fake_client(create_mock):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )


def test_moa_defaults_are_well_formed():
    # Invariants, not a catalog snapshot: the exact model list churns with
    # OpenRouter availability (see PR #6636 where gemini-3-pro-preview was
    # removed upstream). What we care about is that the defaults are present
    # and valid vendor/model slugs.
    assert isinstance(moa.REFERENCE_MODELS, list)
    assert len(moa.REFERENCE_MODELS) >= 1
    for m in moa.REFERENCE_MODELS:
        assert isinstance(m, str) and "/" in m and not m.startswith("/")
    assert isinstance(moa.AGGREGATOR_MODEL, str)
    assert "/" in moa.AGGREGATOR_MODEL


def test_normalize_route_accepts_string():
    route = moa._normalize_route("anthropic/claude-opus-4.6")
    assert route["provider"] == "openrouter"
    assert route["model"] == "anthropic/claude-opus-4.6"
    assert route["label"] == "anthropic/claude-opus-4.6"


def test_normalize_route_accepts_dict():
    route = moa._normalize_route(
        {
            "provider": "ollama-cloud",
            "model": "kimi-k2.5",
            "temperature": 0.3,
            "max_tokens": 8192,
            "extra_body": {"custom": "value"},
        }
    )
    assert route["provider"] == "ollama-cloud"
    assert route["model"] == "kimi-k2.5"
    assert route["label"] == "ollama-cloud:kimi-k2.5"
    assert route["temperature"] == 0.3
    assert route["max_tokens"] == 8192
    assert route["extra_body"] == {"custom": "value"}


@pytest.mark.parametrize(
    "spec,expected_msg",
    [
        ({"model": "kimi-k2.5"}, "missing required 'provider'"),
        ({"provider": "ollama-cloud"}, "missing required 'model'"),
        ("", "must not be empty"),
        (123, "must be a string or mapping"),
        ({"provider": "x", "model": "y", "extra_body": "not a dict"}, "must be a mapping"),
    ],
)
def test_normalize_route_validation_errors(spec, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        moa._normalize_route(spec)


def test_extra_body_defaults_per_provider():
    or_route = moa._normalize_route("anthropic/claude-opus-4.6")
    assert moa._resolve_extra_body(or_route) == {
        "reasoning": {"enabled": True, "effort": "xhigh"}
    }
    ollama_route = moa._normalize_route(
        {"provider": "ollama-cloud", "model": "kimi-k2.5"}
    )
    assert moa._resolve_extra_body(ollama_route) is None
    explicit = moa._normalize_route(
        {"provider": "ollama-cloud", "model": "kimi-k2.5", "extra_body": {"x": 1}}
    )
    assert moa._resolve_extra_body(explicit) == {"x": 1}


def test_should_send_temperature_default_and_explicit_omit():
    # legacy "gpt-*" carve-out only fires for bare slugs (no vendor prefix);
    # OpenRouter vendor-prefixed slugs and other providers send temperature.
    bare_gpt_route = moa._normalize_route("gpt-4o-mini")
    assert moa._should_send_temperature(bare_gpt_route) is False
    claude_route = moa._normalize_route("anthropic/claude-opus-4.6")
    assert moa._should_send_temperature(claude_route) is True
    explicit_omit = moa._normalize_route(
        {"provider": "ollama-cloud", "model": "kimi-k2.5", "omit_temperature": True}
    )
    assert moa._should_send_temperature(explicit_omit) is False
    # ollama-cloud sends temperature by default
    ollama_route = moa._normalize_route(
        {"provider": "ollama-cloud", "model": "kimi-k2.5"}
    )
    assert moa._should_send_temperature(ollama_route) is True


def test_resolve_client_for_route_caches(monkeypatch):
    sentinel_client = SimpleNamespace(name="sentinel")
    resolve_calls = []

    def fake_resolve(provider, model=None, async_mode=False, **kw):
        resolve_calls.append((provider, model))
        return sentinel_client, model

    monkeypatch.setattr(moa, "resolve_provider_client", fake_resolve)
    route = moa._normalize_route({"provider": "ollama-cloud", "model": "kimi-k2.5"})
    c1, m1 = moa._resolve_client_for_route(route)
    c2, m2 = moa._resolve_client_for_route(route)
    assert c1 is c2 is sentinel_client
    assert m1 == m2 == "kimi-k2.5"
    assert len(resolve_calls) == 1


def test_resolve_client_for_route_raises_when_unconfigured(monkeypatch):
    monkeypatch.setattr(
        moa, "resolve_provider_client", lambda *a, **kw: (None, None)
    )
    route = moa._normalize_route({"provider": "requesty", "model": "qwen-3"})
    with pytest.raises(RuntimeError, match="provider 'requesty' is not configured"):
        moa._resolve_client_for_route(route)


@pytest.mark.asyncio
async def test_reference_model_retry_warnings_avoid_exc_info_until_terminal_failure(monkeypatch):
    # legacy test, updated to use the new client-resolution surface
    create_mock = AsyncMock(side_effect=RuntimeError("rate limited"))
    fake_client = _fake_client(create_mock)

    monkeypatch.setattr(
        moa, "_resolve_client_for_route", lambda route: (fake_client, route["model"])
    )
    warn = MagicMock()
    err = MagicMock()
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.logger, "error", err)

    label, message, success = await moa._run_reference_model_safe(
        "openai/gpt-5.4-pro", "hello", max_retries=2
    )

    assert label == "openai/gpt-5.4-pro"
    assert success is False
    assert "failed after 2 attempts" in message
    assert warn.call_count == 2
    assert all(call.kwargs.get("exc_info") is None for call in warn.call_args_list)
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_reference_model_unconfigured_provider_fails_fast(monkeypatch):
    monkeypatch.setattr(
        moa, "resolve_provider_client", lambda *a, **kw: (None, None)
    )
    label, message, success = await moa._run_reference_model_safe(
        {"provider": "requesty", "model": "qwen-3"}, "hello", max_retries=5
    )
    assert success is False
    assert label == "requesty:qwen-3"
    assert "requesty" in message and "not configured" in message


@pytest.mark.asyncio
async def test_run_reference_passes_extra_body_only_for_openrouter(monkeypatch):
    captured = {}

    async def capture(**kwargs):
        captured.update(kwargs)
        return _fake_completion("ok")

    monkeypatch.setattr(
        moa,
        "_resolve_client_for_route",
        lambda route: (_fake_client(AsyncMock(side_effect=capture)), route["model"]),
    )
    label, content, success = await moa._run_reference_model_safe(
        {"provider": "ollama-cloud", "model": "kimi-k2.5"}, "hello", max_retries=1
    )
    assert success is True
    assert content == "ok"
    assert label == "ollama-cloud:kimi-k2.5"
    assert "extra_body" not in captured
    assert captured["temperature"] == moa.REFERENCE_TEMPERATURE
    assert captured["model"] == "kimi-k2.5"


@pytest.mark.asyncio
async def test_run_reference_forwards_per_route_extra_body(monkeypatch):
    captured = {}

    async def capture(**kwargs):
        captured.update(kwargs)
        return _fake_completion("ok")

    monkeypatch.setattr(
        moa,
        "_resolve_client_for_route",
        lambda route: (_fake_client(AsyncMock(side_effect=capture)), route["model"]),
    )
    await moa._run_reference_model_safe(
        {
            "provider": "ollama-cloud",
            "model": "kimi-k2.5",
            "extra_body": {"reasoning": {"effort": "high"}},
            "temperature": 0.2,
        },
        "hello",
        max_retries=1,
    )
    assert captured["extra_body"] == {"reasoning": {"effort": "high"}}
    assert captured["temperature"] == 0.2


@pytest.mark.asyncio
async def test_moa_top_level_error_logs_single_traceback_on_aggregator_failure(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(moa, "_load_moa_config", lambda: {})
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(return_value=("anthropic/claude-opus-4.6", "ok", True)),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(side_effect=RuntimeError("aggregator boom")),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    err = MagicMock()
    monkeypatch.setattr(moa.logger, "error", err)

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=["anthropic/claude-opus-4.6"],
        )
    )

    assert result["success"] is False
    assert "Error in MoA processing" in result["error"]
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_moa_uses_config_routes_when_no_args(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_load_moa_config",
        lambda: {
            "references": [
                {"provider": "ollama-cloud", "model": "kimi-k2.5"},
                {"provider": "openrouter", "model": "anthropic/claude-opus-4.6"},
            ],
            "aggregator": {"provider": "ollama-cloud", "model": "kimi-k2.5"},
        },
    )
    captured_routes = []

    async def fake_ref(route, prompt, temperature=None):
        captured_routes.append(route)
        return route["label"], "draft", True

    captured_agg_route = {}

    async def fake_agg(system_prompt, user_prompt, temperature=None, route=None):
        captured_agg_route["route"] = route
        return "final"

    monkeypatch.setattr(moa, "_run_reference_model_safe", fake_ref)
    monkeypatch.setattr(moa, "_run_aggregator_model", fake_agg)
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(await moa.mixture_of_agents_tool("solve this"))

    assert result["success"] is True
    assert result["response"] == "final"
    assert result["models_used"]["aggregator_model"] == "ollama-cloud:kimi-k2.5"
    assert result["models_used"]["reference_models"] == [
        "ollama-cloud:kimi-k2.5",
        "anthropic/claude-opus-4.6",
    ]
    assert {r["provider"] for r in captured_routes} == {"ollama-cloud", "openrouter"}
    assert captured_agg_route["route"]["provider"] == "ollama-cloud"


@pytest.mark.asyncio
async def test_moa_partial_failure_still_succeeds(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_load_moa_config",
        lambda: {
            "references": [
                {"provider": "ollama-cloud", "model": "kimi-k2.5"},
                {"provider": "requesty", "model": "qwen-3"},
            ],
            "aggregator": {"provider": "ollama-cloud", "model": "kimi-k2.5"},
            "min_successful_references": 1,
        },
    )

    async def fake_ref(route, prompt, temperature=None):
        if route["provider"] == "requesty":
            return route["label"], "boom", False
        return route["label"], "draft", True

    async def fake_agg(*args, **kwargs):
        return "final"

    monkeypatch.setattr(moa, "_run_reference_model_safe", fake_ref)
    monkeypatch.setattr(moa, "_run_aggregator_model", fake_agg)
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(await moa.mixture_of_agents_tool("hello"))
    assert result["success"] is True
    assert result["response"] == "final"


@pytest.mark.asyncio
async def test_moa_bad_config_surfaces_clear_error(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_load_moa_config",
        lambda: {"references": [{"provider": "ollama-cloud"}]},  # missing model
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )
    result = json.loads(await moa.mixture_of_agents_tool("hello"))
    assert result["success"] is False
    assert "MoA configuration error" in result["error"]
    assert "missing required 'model'" in result["error"]


def test_check_moa_requirements_legacy_path(monkeypatch):
    monkeypatch.setattr(moa, "_load_moa_config", lambda: {})
    monkeypatch.setattr(moa, "check_openrouter_api_key", lambda: True)
    assert moa.check_moa_requirements() is True
    monkeypatch.setattr(moa, "check_openrouter_api_key", lambda: False)
    assert moa.check_moa_requirements() is False


def test_check_moa_requirements_config_driven(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_load_moa_config",
        lambda: {
            "references": [{"provider": "ollama-cloud", "model": "kimi-k2.5"}],
            "aggregator": {"provider": "ollama-cloud", "model": "kimi-k2.5"},
        },
    )
    monkeypatch.setattr(moa, "check_openrouter_api_key", lambda: False)
    monkeypatch.setattr(
        moa,
        "resolve_provider_client",
        lambda provider, **kw: (SimpleNamespace(), "kimi-k2.5"),
    )
    assert moa.check_moa_requirements() is True

    monkeypatch.setattr(
        moa, "resolve_provider_client", lambda *a, **kw: (None, None)
    )
    assert moa.check_moa_requirements() is False


def test_get_moa_configuration_reflects_overrides(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_load_moa_config",
        lambda: {
            "references": [
                {"provider": "ollama-cloud", "model": "kimi-k2.5"},
                {"provider": "openrouter", "model": "anthropic/claude-opus-4.6"},
            ],
            "aggregator": {"provider": "ollama-cloud", "model": "kimi-k2.5"},
            "reference_temperature": 0.7,
            "aggregator_temperature": 0.5,
            "min_successful_references": 2,
        },
    )
    cfg = moa.get_moa_configuration()
    assert cfg["aggregator_model"] == "ollama-cloud:kimi-k2.5"
    assert cfg["reference_models"] == [
        "ollama-cloud:kimi-k2.5",
        "anthropic/claude-opus-4.6",
    ]
    assert cfg["reference_temperature"] == 0.7
    assert cfg["aggregator_temperature"] == 0.5
    assert cfg["min_successful_references"] == 2
    assert cfg["total_reference_models"] == 2
    assert cfg["failure_tolerance"] == "0/2 models can fail"
