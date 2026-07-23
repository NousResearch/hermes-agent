"""Tests for Hermes → ShinkaEvolve credential bridge (no live API)."""

from __future__ import annotations

from unittest.mock import patch

from tools import ai_scientist_env as ais
from tools import shinka_evolve_env as env_mod


def test_shinka_llm_model_openai_shim_uses_local_url() -> None:
    resolved = ais.ResolvedAiScientistLLM(
        provider_id="nous",
        sakana_model="gpt-4o-mini",
        api_model="DeepHermes-3-Llama-3-8B-Preview",
        api_key="nous-jwt",
        base_url="https://inference-api.nousresearch.com/v1",
        source="nous_auth",
        routing="openai_shim",
    )
    model = env_mod.shinka_llm_model_for_resolved(resolved)
    assert model.startswith("local/DeepHermes-3-Llama-3-8B-Preview@")
    assert "api_key_env=OPENAI_API_KEY" in model


def test_resolve_shinka_run_config_requires_overlay(monkeypatch) -> None:
    with patch.object(env_mod, "_read_shinka_config", return_value={}), patch.object(
        env_mod,
        "resolve_ai_scientist_run_config",
        return_value={
            "overlay": {},
            "provider_id": None,
            "source": None,
            "routing": None,
            "has_credentials": False,
            "sakana_model": "gpt-4o-mini",
        },
    ), patch.object(env_mod, "resolve_ai_scientist_llm", return_value=None):
        config = env_mod.resolve_shinka_run_config("auto")

    assert config["has_credentials"] is False
    assert config["llm_models"] == []


def test_resolve_shinka_run_config_with_nous(monkeypatch) -> None:
    resolved = ais.ResolvedAiScientistLLM(
        provider_id="nous",
        sakana_model="gpt-4o-mini",
        api_model="DeepHermes-3-Llama-3-8B-Preview",
        api_key="nous-jwt",
        base_url=ais.DEFAULT_NOUS_BASE,
        source="nous_auth",
        routing="openai_shim",
    )
    with patch.object(env_mod, "_read_shinka_config", return_value={}), patch.object(
        env_mod,
        "resolve_ai_scientist_run_config",
        return_value={
            "overlay": {"OPENAI_API_KEY": "nous-jwt"},
            "provider_id": "nous",
            "source": "nous_auth",
            "routing": "openai_shim",
            "has_credentials": True,
            "sakana_model": "gpt-4o-mini",
        },
    ), patch.object(env_mod, "resolve_ai_scientist_llm", return_value=resolved):
        config = env_mod.resolve_shinka_run_config("auto")

    assert config["has_credentials"] is True
    assert config["provider_id"] == "nous"
    assert config["llm_models"]
    assert config["llm_models"][0].startswith("local/")
