"""Agent-facing session model override tool.

Lets a skill force the CURRENT thread onto a different model for a multi-turn
task, then switch back — without delegation, without global config mutation,
and without a separate process.

Semantics mirror the gateway ``/model`` session override (same storage, same
resolution priority: session override > channel override > global default):

  - ``action="set"``: resolve model/provider credentials via the shared
    ``switch_model`` pipeline and write the override for this session. The
    thread stays on the new model across turns until cleared. The previous
    override (if any) is snapshotted so ``clear`` can restore it.
  - ``action="clear"``: restore the pre-set override (or remove it when there
    was none), reverting the thread to whatever model it had before.
  - ``action="status"``: report the current effective model + override.

Only works inside a live gateway session (the override store lives on the
gateway runner). CLI/cron/subagent contexts return a clear error.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gateway bridge — mirrors tools/clarify_gateway.py: module-level registry
# populated lazily from the live gateway runner weakref.
# ---------------------------------------------------------------------------


def _get_gateway_runner():
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        return runner() if runner is not None else None
    except Exception:
        return None


def _session_override_state(runner, session_key: str) -> Optional[Dict[str, Any]]:
    """Read the live conversation override for a session (in-memory)."""
    try:
        peek = runner._peek_session_state(session_key)
        if peek is not None and peek.conversation is not None:
            return peek.conversation.model_override
    except Exception:
        logger.debug("model_override: peek failed", exc_info=True)
    return None


def _set_session_override(runner, session_key: str, override: Optional[Dict[str, Any]]) -> None:
    """Write the override to the gateway session state + session store."""
    try:
        state = runner._session_state(session_key)
        state.conversation.model_override = dict(override) if override else None
    except Exception:
        logger.debug("model_override: in-memory write failed", exc_info=True)
    # Persist non-secret parts so the switch survives a gateway restart.
    try:
        store = getattr(runner, "async_session_store", None)
        if store is not None:
            import asyncio

            try:
                asyncio.get_running_loop()
                # Inside the gateway: fire-and-forget; the in-memory write
                # above is authoritative for the running gateway, the store
                # write is best-effort (survives restart).
                async def _persist():
                    await store.set_model_override(session_key, override)

                asyncio.ensure_future(_persist())
            except RuntimeError:
                # No running loop (sync test or CLI context): run directly.
                asyncio.run(store.set_model_override(session_key, override))
    except Exception:
        logger.debug("model_override: store write failed", exc_info=True)
    # Evict the cached agent so the next turn rebuilds from the override.
    try:
        runner._evict_cached_agent(session_key)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


def model_override(
    action: str = "status",
    model: str = "",
    provider: str = "",
    parent_agent=None,
) -> str:
    """Set, clear, or inspect the session model override for this thread.

    Args:
        action: "set" (force model for this thread across turns), "clear"
            (restore the previous model), or "status" (show current).
        model: Model id for action="set", e.g. "grok-4.5".
        provider: Provider name for action="set", e.g. "xai-oauth".
        parent_agent: The running agent (injected by run_agent dispatch).

    Returns:
        JSON string with success/error + the effective model.
    """
    action = (action or "status").strip().lower()
    if action not in ("set", "clear", "status"):
        return json.dumps(
            {"error": f"Unknown action '{action}'. Use 'set', 'clear', or 'status'."},
            ensure_ascii=False,
        )

    runner = _get_gateway_runner()
    session_key = getattr(parent_agent, "_gateway_session_key", None) if parent_agent else None
    if runner is None or not session_key:
        return json.dumps(
            {
                "error": (
                    "model_override requires a live gateway session (this thread's "
                    "override store lives on the gateway runner). Not available from "
                    "CLI, cron, or subagent contexts."
                )
            },
            ensure_ascii=False,
        )

    if action == "set":
        if not model:
            return json.dumps(
                {"error": "action='set' requires a 'model'."}, ensure_ascii=False
            )
        try:
            from hermes_cli.model_switch import switch_model

            result = switch_model(
                raw_input=model,
                current_provider=getattr(parent_agent, "provider", "") or "",
                current_model=getattr(parent_agent, "model", "") or "",
                current_base_url=getattr(parent_agent, "base_url", "") or "",
                current_api_key=getattr(parent_agent, "api_key", "") or "",
                explicit_provider=provider or "",
            )
        except Exception as exc:
            logger.debug("model_override: switch_model failed", exc_info=True)
            return json.dumps(
                {"error": f"Could not resolve model '{model}': {exc}"}, ensure_ascii=False
            )
        if not result.success:
            return json.dumps(
                {"error": result.error_message or "Model resolution failed."},
                ensure_ascii=False,
            )

        # Snapshot the pre-existing override so clear() restores it exactly.
        prior = _session_override_state(runner, session_key)
        override = {
            "model": result.new_model,
            "provider": result.target_provider,
            "api_key": result.api_key,
            "base_url": result.base_url,
            "api_mode": result.api_mode,
        }
        # Tag the restore target in the stored override (non-secret, not
        # sent to the model). Survives restart via the session store.
        _restore_marker = dict(prior) if prior is not None else None
        if _restore_marker is not None:
            _restore_marker.pop("_restore_override", None)
        override["_restore_override"] = _restore_marker  # type: ignore[typeddict-item]
        _set_session_override(runner, session_key, override)

        display = model
        try:
            from hermes_cli.model_switch import format_model_for_display

            display = format_model_for_display(result.new_model)
        except Exception:
            pass
        return json.dumps(
            {
                "ok": True,
                "action": "set",
                "model": display,
                "provider": result.target_provider,
                "note": (
                    "This thread is now pinned to the new model across turns. "
                    "Call model_override(action='clear') when the task is done "
                    "to restore the previous model."
                ),
            },
            ensure_ascii=False,
        )

    if action == "clear":
        state = _session_override_state(runner, session_key) or {}
        restore = state.get("_restore_override")
        if restore is not None and isinstance(restore, dict):
            restore.pop("_restore_override", None)
            _set_session_override(runner, session_key, restore)
            restored_model = restore.get("model") or "previous"
        else:
            _set_session_override(runner, session_key, None)
            restored_model = "global default"
        return json.dumps(
            {"ok": True, "action": "clear", "restored": restored_model},
            ensure_ascii=False,
        )

    # status
    state = _session_override_state(runner, session_key)
    effective = getattr(parent_agent, "model", "") or ""
    if state:
        return json.dumps(
            {
                "ok": True,
                "action": "status",
                "override_model": state.get("model"),
                "override_provider": state.get("provider"),
                "effective_model": effective,
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "ok": True,
            "action": "status",
            "override_model": None,
            "effective_model": effective,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_OVERRIDE_SCHEMA = {
    "name": "model_override",
    "description": (
        "Switch the CURRENT thread's model for a multi-turn task, then switch "
        "back. Unlike delegate_task (separate subagent), this forces THIS "
        "session onto the chosen model for subsequent turns: call "
        "model_override(action='set', model='<m>', provider='<p>') before the "
        "heavy work, then model_override(action='clear') when done to restore "
        "the previous model. Same storage as the /model session override, no "
        "global config change. Also supports action='status' to show the "
        "current effective model. Only works inside a live gateway session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "clear", "status"],
                "description": (
                    "'set' forces the model on this thread across turns; 'clear' "
                    "restores the previous model; 'status' shows the current one."
                ),
            },
            "model": {
                "type": "string",
                "description": "Model id for action='set' (e.g. 'grok-4.5').",
            },
            "provider": {
                "type": "string",
                "description": "Provider name for action='set' (e.g. 'xai-oauth').",
            },
        },
        "required": [],
    },
}


def check_model_override_requirements() -> None:
    """No hard requirements — the tool reports a clear error outside gateway."""
    pass


try:
    from tools.registry import registry

    registry.register(
        name="model_override",
        toolset="delegation",
        schema=MODEL_OVERRIDE_SCHEMA,
        handler=lambda args, **kw: model_override(
            action=args.get("action", "status"),
            model=args.get("model", ""),
            provider=args.get("provider", ""),
            parent_agent=kw.get("parent_agent"),
        ),
        check_fn=check_model_override_requirements,
        emoji="🔄",
    )
except Exception:
    logger.debug("model_override: registry registration failed", exc_info=True)
