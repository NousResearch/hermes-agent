"""Bridge Hermes-resolved credentials into ShinkaEvolve subprocesses.

Reuses the AI-Scientist credential ladder (Codex OAuth, Nous, NVIDIA, Groq,
Gemini, Anthropic, …) so Shinka does not need a parallel secret store.
Behavioral knobs live under ``auxiliary.shinka`` in config.yaml.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlencode

from tools.ai_scientist_env import (
    ResolvedAiScientistLLM,
    resolve_ai_scientist_llm,
    resolve_ai_scientist_run_config,
    resolved_to_overlay,
)

logger = logging.getLogger(__name__)


def _read_shinka_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        aux = cfg.get("auxiliary") if isinstance(cfg.get("auxiliary"), dict) else {}
        section = aux.get("shinka") if isinstance(aux.get("shinka"), dict) else {}
        return section
    except Exception:
        return {}


def shinka_llm_model_for_resolved(resolved: ResolvedAiScientistLLM) -> str:
    """Map a Hermes-resolved provider onto a Shinka ``evo.llm_models`` entry."""
    if resolved.routing in {"openai_shim", "ollama"}:
        base = (resolved.base_url or "").rstrip("/")
        if not base:
            return resolved.api_model or resolved.sakana_model
        query = urlencode({"api_key_env": "OPENAI_API_KEY"})
        # local/<model>@http(s)://host/.../v1?api_key_env=OPENAI_API_KEY
        return f"local/{resolved.api_model}@{base}?{query}"

    if resolved.provider_id == "openrouter":
        return resolved.api_model or "meta-llama/llama-3.1-405b-instruct"

    if resolved.provider_id == "gemini":
        return resolved.api_model or resolved.sakana_model

    if resolved.provider_id == "anthropic":
        return resolved.api_model or resolved.sakana_model

    if resolved.provider_id == "deepseek":
        return resolved.api_model or "deepseek-chat"

    return resolved.api_model or resolved.sakana_model or "gpt-4o-mini"


def resolve_shinka_run_config(model: str | None = None) -> dict[str, Any]:
    """Return env overlay + preferred Shinka LLM model for a subprocess launch."""
    section = _read_shinka_config()
    explicit_models = section.get("llm_models")
    requested = (model or "").strip()
    if requested.lower() in {"", "auto"}:
        requested = ""

    run = resolve_ai_scientist_run_config(requested or None)
    overlay = dict(run.get("overlay") or {})
    resolved = resolve_ai_scientist_llm(requested or None)

    llm_models: list[str] = []
    if isinstance(explicit_models, list) and explicit_models and not requested:
        llm_models = [str(item).strip() for item in explicit_models if str(item).strip()]
    elif resolved is not None:
        llm_models = [shinka_llm_model_for_resolved(resolved)]
        overlay.update(resolved_to_overlay(resolved))
    elif requested:
        llm_models = [requested]

    return {
        "overlay": overlay,
        "llm_models": llm_models,
        "provider_id": run.get("provider_id"),
        "source": run.get("source"),
        "routing": run.get("routing"),
        "has_credentials": bool(overlay) or bool(llm_models),
        "sakana_model": run.get("sakana_model"),
    }


def build_shinka_env(
    *,
    base: dict[str, str] | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Merge Hermes credential overlay onto a child-process environment."""
    import os

    config = resolve_shinka_run_config(model)
    env = dict(base if base is not None else os.environ)
    env.update(config.get("overlay") or {})
    # Ensure Shinka dotenv does not clobber Hermes-bridged keys from a vendor .env.
    env.setdefault("SHINKA_HERMES_BRIDGE", "1")
    return env


def describe_shinka_credential_resolution(model: str | None = None) -> dict[str, Any]:
    """Non-secret summary for diagnostics and tests."""
    config = resolve_shinka_run_config(model)
    overlay = config.get("overlay") or {}
    return {
        "model": model,
        "llm_models": config.get("llm_models"),
        "provider_id": config.get("provider_id"),
        "routing": config.get("routing"),
        "source": config.get("source"),
        "env_keys": sorted(overlay.keys()),
        "has_credentials": bool(config.get("has_credentials")),
        "llm_models_json": json.dumps(config.get("llm_models") or []),
    }
