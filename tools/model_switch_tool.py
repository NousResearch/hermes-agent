"""Agent-callable model_switch tool.

Allows an agent to change its own model and provider mid-session.
Delegates credential resolution to ``hermes_cli.model_switch.switch_model``
and runtime swap to ``agent.agent_runtime_helpers.switch_model``.

Issue: #16525
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MODEL_SWITCH_SCHEMA = {
    "name": "model_switch",
    "description": (
        "Switch the current agent's model and/or provider mid-session. "
        "Use this to adapt to different task requirements — e.g. switch to a "
        "faster model for simple tasks, or a more capable model for complex reasoning. "
        "The change persists for the rest of the session. "
        "Specify model (required) and optionally provider. "
        "If provider is omitted, it is auto-detected from the model name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": (
                    "The model to switch to (e.g. 'claude-sonnet-4', "
                    "'gemini-2.5-flash', 'gpt-4o'). Required."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "The provider to use (e.g. 'anthropic', 'google', 'openai'). "
                    "Optional — auto-detected from the model name when omitted."
                ),
            },
        },
        "required": ["model"],
    },
}


# ---------------------------------------------------------------------------
# Pipeline wrapper (thin adapter over hermes_cli.model_switch)
# ---------------------------------------------------------------------------

def _run_pipeline(
    raw_input: str,
    explicit_provider: str,
    current_provider: str,
    current_model: str,
    current_base_url: str,
    current_api_key: str,
    user_providers: dict | None,
    custom_providers: list | None,
) -> Any:
    """Call the shared model-switching pipeline.

    Wraps ``hermes_cli.model_switch.switch_model`` so the tool module
    stays testable (callers can patch this function instead of the
    heavy import chain).
    """
    from hermes_cli.model_switch import switch_model as switch_model_pipeline

    return switch_model_pipeline(
        raw_input=raw_input,
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
        explicit_provider=explicit_provider,
        user_providers=user_providers,
        custom_providers=custom_providers,
    )


def _run_runtime_switch(
    agent: Any,
    new_model: str,
    new_provider: str,
    api_key: str,
    base_url: str,
    api_mode: str,
) -> None:
    """Call the runtime model swap on the live agent."""
    from agent.agent_runtime_helpers import switch_model as switch_model_runtime

    switch_model_runtime(
        agent,
        new_model=new_model,
        new_provider=new_provider,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
    )


def _get_user_providers() -> dict | None:
    """Load user providers from config (lazy)."""
    try:
        from hermes_cli.config import get_config
        cfg = get_config()
        return cfg.get("providers")
    except Exception:
        return None


def _get_custom_providers() -> list | None:
    """Load custom providers from config (lazy)."""
    try:
        from hermes_cli.config import get_config
        cfg = get_config()
        return cfg.get("custom_providers")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def model_switch(
    model: str,
    provider: str = "",
    parent_agent: Any = None,
) -> Dict[str, Any]:
    """Switch the agent's model and/or provider.

    Args:
        model: Target model name (required).
        provider: Target provider (optional, auto-detected if omitted).
        parent_agent: The live agent instance (injected by the tool executor).

    Returns:
        Dict with success, message, and optional warning.
    """
    if parent_agent is None:
        return {
            "success": False,
            "message": "Cannot switch model: no agent instance available.",
        }

    # Short-circuit: same model + same provider
    if (
        model == getattr(parent_agent, "model", None)
        and (not provider or provider == getattr(parent_agent, "provider", None))
    ):
        return {
            "success": True,
            "message": f"Already using {model} on {getattr(parent_agent, 'provider', 'unknown')}. No switch needed.",
        }

    # Resolve credentials via the shared pipeline
    try:
        result = _run_pipeline(
            raw_input=model,
            explicit_provider=provider,
            current_provider=getattr(parent_agent, "provider", ""),
            current_model=getattr(parent_agent, "model", ""),
            current_base_url=getattr(parent_agent, "base_url", ""),
            current_api_key=getattr(parent_agent, "api_key", ""),
            user_providers=_get_user_providers(),
            custom_providers=_get_custom_providers(),
        )
    except Exception as exc:
        logger.warning("model_switch pipeline error: %s", exc)
        return {
            "success": False,
            "message": f"Failed to resolve model '{model}': {exc}",
        }

    if not result.success:
        return {
            "success": False,
            "message": result.error_message or f"Failed to switch to {model}.",
        }

    # Apply the runtime swap
    try:
        _run_runtime_switch(
            parent_agent,
            new_model=result.new_model,
            new_provider=result.target_provider,
            api_key=result.api_key,
            base_url=result.base_url,
            api_mode=result.api_mode,
        )
    except Exception as exc:
        logger.warning("model_switch runtime error: %s", exc)
        return {
            "success": False,
            "message": f"Resolved credentials for {result.new_model} but runtime swap failed: {exc}",
        }

    response = {
        "success": True,
        "message": (
            f"Switched to {result.new_model} "
            f"via {result.provider_label or result.target_provider}."
        ),
    }

    if result.warning_message:
        response["warning"] = result.warning_message

    return response


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_model_switch_requirements() -> bool:
    """The model_switch tool is always available when delegation is enabled."""
    return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="model_switch",
    toolset="delegation",
    schema=MODEL_SWITCH_SCHEMA,
    handler=lambda args, **kw: model_switch(
        model=args.get("model", ""),
        provider=args.get("provider", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_model_switch_requirements,
    emoji="🔀",
)
