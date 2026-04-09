"""Session-scoped model inspection and switching tool."""

from __future__ import annotations

import json
import os
from typing import Any

from tools.registry import registry


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _normalize_allowed_entry(raw: Any, user_providers: dict | None = None) -> dict[str, str] | None:
    from hermes_cli.providers import resolve_provider_full

    provider = ""
    model = ""
    label = ""

    if isinstance(raw, dict):
        provider = str(raw.get("provider") or "").strip()
        model = str(raw.get("model") or "").strip()
        label = str(raw.get("label") or "").strip()
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        provider_candidate = ""
        model_candidate = text
        if ":" in text:
            prefix, suffix = text.split(":", 1)
            if "/" not in prefix:
                provider_def = resolve_provider_full(prefix.strip(), user_providers)
                if provider_def is not None and suffix.strip():
                    provider_candidate = provider_def.id
                    model_candidate = suffix.strip()
        provider = provider_candidate
        model = model_candidate

    if not model:
        return None

    if provider:
        provider_def = resolve_provider_full(provider, user_providers)
        if provider_def is not None:
            provider = provider_def.id

    display = f"{provider}:{model}" if provider else model
    return {
        "provider": provider,
        "model": model,
        "label": label,
        "display": display,
    }


def get_allowed_self_models() -> tuple[list[dict[str, str]], dict | None]:
    """Load the allowlist for agent-initiated session model switches."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        cfg = {}

    user_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    raw_entries = _coerce_list(agent_cfg.get("allowed_self_models"))
    raw_entries.extend(_coerce_list(os.getenv("HERMES_ALLOWED_SELF_MODELS", "")))

    allowed: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_entries:
        normalized = _normalize_allowed_entry(raw, user_providers=user_providers)
        if normalized is None:
            continue
        key = normalized["display"].lower()
        if key in seen:
            continue
        seen.add(key)
        allowed.append(normalized)
    return allowed, user_providers


def is_self_model_allowed(allowed_models: list[dict[str, str]], model: str, provider: str) -> bool:
    """Return True when the resolved model/provider pair is in the allowlist."""
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip().lower()
    for entry in allowed_models:
        entry_model = str(entry.get("model") or "").strip().lower()
        entry_provider = str(entry.get("provider") or "").strip().lower()
        if entry_model != normalized_model:
            continue
        if entry_provider and entry_provider != normalized_provider:
            continue
        return True
    return False


def check_session_model_requirements() -> bool:
    allowed_models, _user_providers = get_allowed_self_models()
    return bool(allowed_models)


def session_model_tool(args, **kwargs) -> str:
    """Stub handler; session_model must be intercepted by the agent loop."""
    del args, kwargs
    return json.dumps({"error": "session_model must be handled by the agent loop"}, ensure_ascii=False)


SESSION_MODEL_SCHEMA = {
    "name": "session_model",
    "description": (
        "Inspect or switch the current session's model within a preconfigured allowlist. "
        "Call without 'model' to see the current model and allowed targets. "
        "Switches are session-only, never persist global config, and should be used sparingly "
        "when the current model is a poor fit for the task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Requested target model. Omit to inspect the current model and allowed targets.",
            },
            "provider": {
                "type": "string",
                "description": "Optional provider slug when the target model should be resolved on a specific provider.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="session_model",
    toolset="session_model",
    schema=SESSION_MODEL_SCHEMA,
    handler=session_model_tool,
    check_fn=check_session_model_requirements,
    emoji="🧭",
)
