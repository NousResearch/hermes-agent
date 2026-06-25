#!/usr/bin/env python3
"""Model Switch Tool — agent-driven complexity-based model routing.

Exposes the existing :func:`hermes_cli.model_switch.switch_model` pipeline as
an agent-callable tool so the model can escalate (or downgrade) its own backend
mid-conversation when it detects the current model is over- or under-powered for
the task at hand.  This is the concrete implementation behind the
``smart_model_routing`` config stub (upstream #16525), which previously had no
Python wiring.

Like ``todo`` / ``memory`` / ``delegate_task``, this tool needs live agent
state (it must mutate the running :class:`AIAgent`), so the registry handler is
a stub and the real work is intercepted in the agent loop
(``agent.tool_executor`` / ``agent.agent_runtime_helpers``), which calls
:func:`model_switch_tool` with the live agent.

Scopes:
  - ``session`` — persists until the session is reset or another switch.
    Implemented via :meth:`AIAgent.switch_model`, which updates
    ``_primary_runtime`` so the change survives across turns.
  - ``turn`` — one-shot.  The pre-switch runtime is snapshotted onto the agent
    and restored at the start of the next turn (see ``conversation_loop``).
"""

import json
import logging

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# Agent attribute holding the turn-scoped revert snapshot.  Read by the
# conversation loop's per-turn reset block.  Kept here as the single source of
# truth so the loop and this tool can't drift on the attribute name.
TURN_REVERT_ATTR = "_model_switch_turn_revert"


MODEL_SWITCH_SCHEMA = {
    "name": "model_switch",
    "description": (
        "Switch your own model when the current one is mismatched to the task — "
        "escalate to a more capable model for hard reasoning/coding, or downgrade "
        "to a cheaper/faster model for simple work. Takes effect on your NEXT turn. "
        "Use sparingly and only when the complexity clearly warrants it; explain "
        "the switch to the user via 'reason'. Examples of slugs: "
        "'deepseek/deepseek-r1', 'gpt-4o', 'anthropic/claude-opus-4', 'sonnet'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Target model identifier or short alias (e.g. "
                    "'deepseek/deepseek-r1', 'gpt-4o', 'opus', 'haiku'). "
                    "Aliases resolve against your authenticated providers."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "One-sentence justification for the switch, surfaced to the "
                    "user (e.g. 'Escalating to a reasoning model for this proof')."
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["session", "turn"],
                "description": (
                    "'session' (default) persists the switch until reset; "
                    "'turn' applies it for the next turn only, then reverts."
                ),
            },
        },
        "required": ["slug", "reason"],
    },
}


def _load_providers_config() -> tuple[dict | None, list | None]:
    """Return ``(user_providers, custom_providers)`` from config for resolution.

    Why: ``switch_model`` resolves aliases / credentials against the user's
    configured ``providers:`` and ``custom_providers:`` just like the /model
    command does; without them, custom endpoints wouldn't resolve.
    What: Loads the persistent config and extracts both sections, mirroring the
    gateway's ``_handle_model_command``.
    Test: Patch ``hermes_cli.config.load_config`` to return a config with
    ``providers``/``custom_providers`` and assert both are returned.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        user_providers = cfg.get("providers")
        try:
            from hermes_cli.config import get_compatible_custom_providers

            custom_providers = get_compatible_custom_providers(cfg)
        except Exception:
            custom_providers = cfg.get("custom_providers")
        return user_providers, custom_providers
    except Exception:
        return None, None


def _allowlist() -> list[str] | None:
    """Return the ``model_switch_allowlist`` config list, or None if unset.

    Why: Optional safeguard (#16525) so operators can restrict which models the
    agent may self-route to; default (unset) allows any configured model.
    What: Reads the top-level ``model_switch_allowlist`` key and normalizes it
    to a lowercased list, or None when absent/empty.
    Test: Patch config to ``{"model_switch_allowlist": ["gpt-4o"]}`` and assert
    ['gpt-4o'] is returned; unset config returns None.
    """
    try:
        from hermes_cli.config import load_config

        raw = (load_config() or {}).get("model_switch_allowlist")
    except Exception:
        return None
    if not raw or not isinstance(raw, list):
        return None
    return [str(m).strip().lower() for m in raw if str(m).strip()]


def model_switch_tool(agent, slug: str, reason: str, scope: str = "session") -> str:
    """Switch the live agent's model. Called from the agent loop with the agent.

    Why: Lets the model self-escalate/downgrade based on task complexity, the
    concrete behavior behind the ``smart_model_routing`` stub (#16525).
    What: Resolves *slug* via :func:`hermes_cli.model_switch.switch_model`,
    applies it with :meth:`AIAgent.switch_model`, and for ``scope='turn'``
    snapshots the prior runtime so the loop reverts it next turn. Returns a JSON
    string with old/new model, scope, and ``applied_at: 'next_turn'``.
    Test: Build a stub agent with ``switch_model`` recorded, call with a known
    slug, and assert the returned dict carries the resolved new_model and that
    the agent's switch_model was invoked; turn scope sets TURN_REVERT_ATTR.
    """
    from hermes_cli.model_switch import switch_model as _switch_model

    slug = (slug or "").strip()
    reason = (reason or "").strip()
    scope = (scope or "session").strip().lower()
    if scope not in {"session", "turn"}:
        scope = "session"
    if not slug:
        return tool_error("model_switch requires a non-empty 'slug'.")

    # Optional allowlist safeguard — default allows all configured models.
    allow = _allowlist()
    if allow is not None and slug.lower() not in allow:
        return tool_error(
            f"Model '{slug}' is not in model_switch_allowlist "
            f"({', '.join(allow)}). Pick an allowed model or remove the allowlist."
        )

    old_model = getattr(agent, "model", "")
    old_provider = getattr(agent, "provider", "")
    old_base_url = getattr(agent, "base_url", "")
    old_api_key = getattr(agent, "api_key", "")

    user_providers, custom_providers = _load_providers_config()

    # Resolve the target model + credentials through the shared pipeline.
    result = _switch_model(
        raw_input=slug,
        current_provider=old_provider,
        current_model=old_model,
        current_base_url=old_base_url,
        current_api_key=old_api_key,
        is_global=False,
        explicit_provider="",
        user_providers=user_providers,
        custom_providers=custom_providers,
    )
    if not result.success:
        return tool_error(
            f"Could not switch to '{slug}': {result.error_message}"
        )

    # For turn scope, snapshot the current runtime so the loop can restore it
    # at the start of the next turn (one-shot). Stored before mutation.
    if scope == "turn":
        setattr(agent, TURN_REVERT_ATTR, {
            "model": old_model,
            "provider": old_provider,
            "base_url": old_base_url,
            "api_key": old_api_key,
            "api_mode": getattr(agent, "api_mode", ""),
        })
    else:
        # A session switch supersedes any pending turn revert.
        if getattr(agent, TURN_REVERT_ATTR, None) is not None:
            setattr(agent, TURN_REVERT_ATTR, None)

    # Apply the swap in-place. switch_model() updates _primary_runtime so a
    # session switch persists across turns; a turn switch is reverted by the
    # loop before the next turn begins.
    try:
        agent.switch_model(
            new_model=result.new_model,
            new_provider=result.target_provider,
            api_key=result.api_key,
            base_url=result.base_url,
            api_mode=result.api_mode,
        )
    except Exception as exc:
        # Roll back the turn snapshot so a failed apply doesn't strand a revert.
        if scope == "turn":
            setattr(agent, TURN_REVERT_ATTR, None)
        logger.warning("model_switch apply failed (%s -> %s): %s",
                       old_model, result.new_model, exc)
        return tool_error(f"Model switch to '{result.new_model}' failed: {exc}")

    logger.info(
        "Agent self-switched model: %s -> %s (scope=%s, reason=%s)",
        old_model, result.new_model, scope, reason,
    )
    return json.dumps({
        "old_model": old_model,
        "new_model": result.new_model,
        "scope": scope,
        "applied_at": "next_turn",
    }, ensure_ascii=False)


def revert_turn_model_switch(agent) -> bool:
    """Restore a turn-scoped model switch at the start of the next turn.

    Why: ``scope='turn'`` switches must auto-revert after one turn; centralizing
    the restore here keeps the conversation loop's hook a one-liner.
    What: If TURN_REVERT_ATTR holds a snapshot, re-applies it via
    :meth:`AIAgent.switch_model` and clears the snapshot. Returns True if a
    revert happened.
    Test: Set TURN_REVERT_ATTR to a prior-runtime dict, call this, assert
    switch_model was invoked with those values and the attribute is cleared.
    """
    snapshot = getattr(agent, TURN_REVERT_ATTR, None)
    if not snapshot:
        return False
    setattr(agent, TURN_REVERT_ATTR, None)
    try:
        agent.switch_model(
            new_model=snapshot.get("model", ""),
            new_provider=snapshot.get("provider", ""),
            api_key=snapshot.get("api_key", ""),
            base_url=snapshot.get("base_url", ""),
            api_mode=snapshot.get("api_mode", ""),
        )
        logger.info("Reverted turn-scoped model switch -> %s", snapshot.get("model"))
        return True
    except Exception as exc:
        logger.warning("Failed to revert turn-scoped model switch: %s", exc)
        return False


def check_model_switch_requirements() -> bool:
    """Gate the tool on the ``agent.allow_self_model_switch`` config flag.

    Why: Self-model-switching is powerful and cost-affecting, so it ships
    opt-in (matching how send_message / kanban gate via check_fn).
    What: Returns True only when ``agent.allow_self_model_switch`` is truthy.
    Test: Patch config with the flag true/false and assert the return value
    tracks it; missing flag returns False.
    """
    try:
        from hermes_cli.config import load_config
        from utils import is_truthy_value

        agent_cfg = (load_config() or {}).get("agent") or {}
        if not isinstance(agent_cfg, dict):
            return False
        return is_truthy_value(agent_cfg.get("allow_self_model_switch", False))
    except Exception:
        return False


# --- Registry ---
# The schema is registered so the model sees the tool, but execution is
# intercepted by the agent loop (model_switch is in model_tools._AGENT_LOOP_TOOLS)
# because the handler needs the live agent. This stub only fires if a call
# somehow bypasses the loop interception.
registry.register(
    name="model_switch",
    toolset="model_switch",
    schema=MODEL_SWITCH_SCHEMA,
    handler=lambda *_args, **_kw: tool_error(
        "model_switch must be handled by the agent loop"
    ),
    check_fn=check_model_switch_requirements,
    emoji="🔀",
)
