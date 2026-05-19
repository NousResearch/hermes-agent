import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

moa = importlib.import_module("tools.mixture_of_agents_tool")


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


@pytest.mark.asyncio
async def test_reference_model_retry_warnings_avoid_exc_info_until_terminal_failure(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=RuntimeError("rate limited"))
            )
        )
    )
    warn = MagicMock()
    err = MagicMock()

    monkeypatch.setattr(moa, "_get_moa_client", lambda: fake_client)
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.logger, "error", err)

    model, message, success = await moa._run_reference_model_safe(
        "openai/gpt-5.4-pro", "hello", max_retries=2
    )

    assert model == "openai/gpt-5.4-pro"
    assert success is False
    assert "failed after 2 attempts" in message
    assert warn.call_count == 2
    assert all(call.kwargs.get("exc_info") is None for call in warn.call_args_list)
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_moa_top_level_error_logs_single_traceback_on_aggregator_failure(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
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


def test_check_moa_requirements_accepts_ai_gateway_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MOA_PROVIDER", raising=False)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test")
    assert moa.check_moa_requirements() is True


def test_check_moa_requirements_accepts_openrouter_key(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("MOA_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    assert moa.check_moa_requirements() is True


def test_check_moa_requirements_false_without_keys(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MOA_PROVIDER", raising=False)
    assert moa.check_moa_requirements() is False


def test_resolve_provider_prefers_ai_gateway_when_both_keys_present(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.delenv("MOA_PROVIDER", raising=False)
    monkeypatch.setattr(moa, "_load_hermes_config", lambda: {})
    assert moa._resolve_provider_name() == "ai-gateway"


def test_resolve_provider_honors_explicit_moa_provider_env(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("MOA_PROVIDER", "openrouter")
    monkeypatch.setattr(moa, "_load_hermes_config", lambda: {})
    assert moa._resolve_provider_name() == "openrouter"
