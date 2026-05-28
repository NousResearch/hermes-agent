#!/usr/bin/env python3
"""Agent-level tool for switching the active model mid-session."""

import json
from typing import Optional

SWITCH_MODEL_SCHEMA = {
    "name": "switch_model",
    "description": (
        "Change the active model for the remainder of this conversation. "
        "Useful when you need a different model's capabilities "
        "(stronger reasoning, faster response, vision, larger context).\n\n"
        "After switching, all subsequent messages use the new model. "
        "The change persists for the current session only.\n\n"
        "Available models:\n"
        "- gpt-4.1-nano — cheapest, fast, cache works. Use: greetings, status, confirmations, simple Q&A\n"
        "- gpt-5.4-mini — balanced. Use: code review, analysis, medium complexity\n"
        "- gpt-5.4 — strong reasoning, large context. Use: complex debug, planning, architecture\n"
        "- gpt-5.5 — strongest model. Use: frontier research, multi-file refactoring, critical issues\n\n"
        "Format: just the model name (e.g. \"gpt-5.4\"). Provider auto-detected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "new_model": {
                "type": "string",
                "description": "Model name to switch to (e.g. gpt-4.1-nano, gpt-5.4, gpt-5.5)",
            },
            "reason": {
                "type": "string",
                "description": "Optional explanation of why the switch is needed",
            },
            "provider": {
                "type": "string",
                "description": "Optional provider slug, equivalent to /model <model> --provider <provider>",
            },
        },
        "required": ["new_model"],
    },
}


def switch_model_tool(
    new_model: str,
    reason: Optional[str] = None,
    provider: Optional[str] = None,
    agent=None,
) -> str:
    """Switch the agent's active model at runtime."""
    if agent is None:
        return json.dumps({
            "success": False,
            "error": "switch_model must be handled by the agent loop",
        }, ensure_ascii=False)

    return switch_model_for_agent(agent, new_model, reason=reason, provider=provider)


def switch_model_for_agent(
    agent,
    new_model: str,
    *,
    reason: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Resolve and apply a session-local model switch for an AIAgent."""
    if not new_model or not new_model.strip():
        return json.dumps({
            "success": False,
            "error": "new_model is required",
        }, ensure_ascii=False)

    new_model = new_model.strip()
    old_model = getattr(agent, "model", "") or ""
    current_api_key = getattr(agent, "api_key", "") or ""
    if not isinstance(current_api_key, str):
        current_api_key = ""

    user_providers = None
    custom_providers = None
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        cfg = load_config()
        if isinstance(cfg, dict):
            providers_cfg = cfg.get("providers")
            if isinstance(providers_cfg, dict):
                user_providers = providers_cfg
            custom_providers = get_compatible_custom_providers(cfg)
    except Exception:
        user_providers = None
        custom_providers = None

    try:
        from hermes_cli.model_switch import switch_model

        result = switch_model(
            raw_input=new_model,
            current_provider=getattr(agent, "provider", "") or "",
            current_model=old_model,
            current_base_url=getattr(agent, "base_url", "") or "",
            current_api_key=current_api_key,
            is_global=False,
            explicit_provider=(provider or "").strip(),
            user_providers=user_providers,
            custom_providers=custom_providers,
        )
    except Exception as exc:
        return json.dumps({
            "success": False,
            "error": f"Failed to resolve model switch: {exc}",
        }, ensure_ascii=False)

    if not result.success:
        return json.dumps({
            "success": False,
            "error": result.error_message or "model switch failed",
        }, ensure_ascii=False)

    try:
        agent.switch_model(
            new_model=result.new_model,
            new_provider=result.target_provider,
            api_key=result.api_key,
            base_url=result.base_url,
            api_mode=result.api_mode,
        )
    except Exception as exc:
        return json.dumps({
            "success": False,
            "error": f"Failed to switch model: {exc}",
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "old_model": old_model,
        "new_model": result.new_model,
        "provider": result.target_provider,
        "provider_label": result.provider_label or result.target_provider,
        "reason": reason or "",
        "warning": result.warning_message or "",
    }, ensure_ascii=False)


def check_switch_model_requirements() -> bool:
    """No external requirements — always available."""
    return True


# =============================================================================
# Registry — auto-discovered by model_tools.py
# =============================================================================
from tools.registry import registry  # noqa: E402

registry.register(
    name="switch_model",
    toolset="model_control",
    schema=SWITCH_MODEL_SCHEMA,
    handler=lambda args, **kw: switch_model_tool(
        new_model=args.get("new_model", ""),
        reason=args.get("reason"),
        provider=args.get("provider"),
        agent=kw.get("agent")),
    check_fn=check_switch_model_requirements,
    emoji="🔄",
)
