#!/usr/bin/env python3
"""
Context Usage & Agent-Triggered Compression Tool Module

Provides two tools the agent can call to monitor and manage context usage:

1. ``context_status()`` — reports current context window usage (tokens used,
   percent consumed, context length, compression threshold, etc.) without
   side effects.

2. ``request_compression(reason)`` — requests the runtime to trigger context
   compression now, with a human-readable reason the agent provides.  Gated
   by config (``compression.allow_agent_trigger``) and an optional minimum
   usage threshold (``compression.agent_suggest_threshold``).

Design:
- Tools are lightweight read / request — no state of their own.
- ``request_compression`` delegates to the agent's ``_compress_context()``
  which already exists for ``/compress`` slash commands and auto-compression.
- Safety: tool is opt-in via config, runtime may reject if already in
  progress or cooldown is active.
"""

import json
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


def _json_response(payload: Dict[str, Any]) -> str:
    """Serialize a tool response without hand-building JSON strings."""
    return json.dumps(payload, ensure_ascii=False)


def _tool_call_id_from_entry(tool_call: Any) -> str:
    """Extract a tool call ID from a persisted dict or SDK-ish object."""
    if isinstance(tool_call, dict):
        return str(tool_call.get("call_id") or tool_call.get("id") or "")
    return str(getattr(tool_call, "call_id", "") or getattr(tool_call, "id", "") or "")


def _latest_inflight_tool_tail_start(messages: list, current_tool_call_id: str) -> int | None:
    """Return the index of the current assistant tool-call group, if any.

    During tool execution, ``messages`` already contains the assistant message
    with its tool_calls, while the current tool result has not been appended
    yet.  Compressing that assistant message makes the compressor's tool-pair
    sanitizer insert synthetic "missing result" stubs for the in-flight call.
    To avoid corrupting the transcript, request_compression compacts only the
    history *before* the current assistant tool-call group and re-attaches the
    in-flight assistant/tool-result tail verbatim afterward.
    """
    if not current_tool_call_id:
        return None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        call_ids = {
            _tool_call_id_from_entry(tc)
            for tc in (msg.get("tool_calls") or [])
        }
        if current_tool_call_id in call_ids:
            return idx
    return None


# --- Schema ---

CONTEXT_STATUS_SCHEMA = {
    "name": "context_status",
    "description": (
        "Read the current context window usage for this session. "
        "Returns tokens used, percent of context consumed, the model's "
        "maximum context length, the compression threshold, whether "
        "compression is enabled, and whether the agent is allowed to "
        "request compression.  No side effects."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

COMPRESS_SCHEMA = {
    "name": "request_compression",
    "description": (
        "Request context compression to free up space in the current "
        "conversation window.  Middle turns are summarized by a fast "
        "auxiliary model while head and tail context are preserved.  "
        "Gated by configuration — the runtime may reject the request "
        "if compression is already in progress, too recent, or usage "
        "is below the agent-suggest threshold.  Provide a reason so "
        "the runtime can log why compression was requested.  "
        "Pass force=true to bypass the agent_suggest_threshold gate "
        "and use aggressive message-count-based tail protection for "
        "maximum compression at any context percentage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Why compression is needed.  Examples: "
                    "\"approaching context limit in long PM workflow\", "
                    "\"about to receive large tool result\""
                ),
            },
            "force": {
                "type": "boolean",
                "description": (
                    "When true, bypass the agent_suggest_threshold gate "
                    "and use aggressive message-count-based tail "
                    "protection (protect_last_n) instead of the default "
                    "token-budget tail.  Use this when the agent knows "
                    "it is entering an idle/poll loop and can safely "
                    "compress most context away."
                ),
            },
        },
        "required": ["reason"],
    },
}


# --- Handler used by tool_executor (receives agent reference via kwargs) ---

def context_status_handler(args: Dict[str, Any], **kw: Any) -> str:
    """
    Handler for ``context_status`` tool.

    Expects ``kw['agent']`` — the AIAgent instance whose context to inspect.
    """
    agent = kw.get("agent")
    if agent is None:
        return _json_response({
            "error": "context_status requires agent reference",
            "available": False,
        })

    prompt_tokens = getattr(agent, "session_prompt_tokens", 0)
    total_tokens = getattr(agent, "session_total_tokens", 0)

    # Read context length from the context compressor, falling back to the
    # model metadata helper if the compressor is not yet initialised.
    context_length = _resolve_context_length(agent)

    # Current actual prompt tokens (from context compressor, NOT session cumul)
    compressor = getattr(agent, "context_compressor", None)
    last_prompt_tokens = 0
    if compressor is not None:
        lp = getattr(compressor, "last_prompt_tokens", None)
        awaiting_after_compression = bool(
            getattr(compressor, "awaiting_real_usage_after_compression", False)
        )
        if lp == -1 and awaiting_after_compression:
            # Compression just ran; no provider-reported post-compression
            # prompt count exists yet. Report the rough compressed estimate
            # instead of falling back to the stale pre-compression real usage.
            lct = getattr(compressor, "last_compression_rough_tokens", None)
            if lct is not None and lct > 0:
                last_prompt_tokens = int(lct)
        elif lp is not None and lp >= 0:
            last_prompt_tokens = lp
        if not last_prompt_tokens:
            lrp = getattr(compressor, "last_real_prompt_tokens", None)
            if lrp is not None and lrp > 0:
                last_prompt_tokens = lrp

    # Read compression config from agent or reach into agent's config store.
    compression_config = _read_compression_config(agent)
    threshold = compression_config.get("threshold", 0.50)
    enabled = compression_config.get("enabled", True)
    agent_suggest = compression_config.get("agent_suggest_threshold", None)
    allow_agent_trigger = compression_config.get("allow_agent_trigger", False)

    percent_used = round(last_prompt_tokens / context_length * 100, 1) if context_length > 0 else 0.0
    threshold_pct = round(threshold * 100, 1)
    session_percent = round(prompt_tokens / context_length * 100, 1) if context_length > 0 else 0.0

    return _json_response({
        "tokens_used": last_prompt_tokens,
        "session_prompt_tokens": prompt_tokens,
        "total_tokens_this_session": total_tokens,
        "context_length": context_length,
        "percent_used": percent_used,
        "session_percent": session_percent,
        "compression_threshold_pct": threshold_pct,
        "compression_enabled": enabled,
        "allow_agent_trigger": allow_agent_trigger,
        "agent_suggest_threshold": agent_suggest,
        "tool": "context_status",
    })


def request_compression_handler(args: Dict[str, Any], **kw: Any) -> str:
    """
    Handler for ``request_compression`` tool.

    Expects ``kw['agent']`` plus, for live tool-loop compression,
    ``kw['messages']`` — the mutable message list currently being sent
    through ``run_conversation``.  Falling back to agent-level snapshots is
    best-effort only; the live list is what makes compression take effect in
    the current turn.
    """
    agent = kw.get("agent")
    if agent is None:
        return _json_response({
            "error": "request_compression requires agent reference",
            "compressed": False,
        })

    reason = args.get("reason", "").strip()
    if not reason:
        return _json_response({"error": "reason is required", "compressed": False})

    force = bool(args.get("force", False))

    # Gate 1: check config allows agent-triggered compression
    compression_config = _read_compression_config(agent)
    if not compression_config.get("allow_agent_trigger", False):
        return _json_response({
            "error": (
                "agent-triggered compression is disabled in config "
                "(compression.allow_agent_trigger)"
            ),
            "compressed": False,
        })

    if not compression_config.get("enabled", True):
        return _json_response({
            "error": "compression is disabled in config",
            "compressed": False,
        })

    # Gate 2: check agent_suggest_threshold (skip when force=True)
    agent_suggest = compression_config.get("agent_suggest_threshold")
    if agent_suggest is not None and not force:
        context_length = _resolve_context_length(agent)
        # Use current prompt tokens (from compressor), not session cumul
        compressor = getattr(agent, "context_compressor", None)
        current_tokens = 0
        if compressor is not None:
            lp = getattr(compressor, "last_prompt_tokens", None)
            if lp is not None and lp >= 0:
                current_tokens = lp
            if not current_tokens:
                lrp = getattr(compressor, "last_real_prompt_tokens", None)
                if lrp is not None and lrp > 0:
                    current_tokens = lrp
        percent_used = current_tokens / context_length if context_length > 0 else 0.0
        if percent_used < agent_suggest:
            return _json_response({
                "error": (
                    "usage below agent_suggest_threshold "
                    f"({agent_suggest * 100:.0f}%), currently at "
                    f"{percent_used * 100:.1f}%"
                ),
                "compressed": False,
                "percent_used": round(percent_used * 100, 1),
            })

    # Gate 3: check compression lock / cooldown via context_compressor
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        # The compressor may raise or return a non-None error string
        # if compression is already in progress or locked.
        try:
            can_compress = getattr(compressor, "can_compress", None)
            if callable(can_compress) and not can_compress():
                return _json_response({
                    "error": "compression is not available right now (lock or cooldown active)",
                    "compressed": False,
                })
        except Exception as exc:
            return _json_response({
                "error": f"compression check failed: {exc}",
                "compressed": False,
            })

    # Trigger compression — delegate to the agent's existing method.
    try:
        _compress_context = getattr(agent, "_compress_context", None)
        if _compress_context is None:
            return _json_response({
                "error": "_compress_context not available on agent",
                "compressed": False,
            })

        live_messages = kw.get("messages")
        if live_messages is None:
            live_messages = getattr(agent, "_session_messages", None)
        if live_messages is None:
            live_messages = getattr(agent, "conversation_history", None)
        if not isinstance(live_messages, list):
            return _json_response({
                "error": (
                    "live conversation messages are not available; "
                    "request_compression must be called from the tool loop"
                ),
                "compressed": False,
            })

        pre_count = len(live_messages)
        if pre_count == 0:
            return _json_response({
                "error": "no live messages available to compress",
                "compressed": False,
            })

        current_tool_call_id = str(kw.get("tool_call_id") or "")
        tail_start = _latest_inflight_tool_tail_start(live_messages, current_tool_call_id)
        protected_inflight_tail = []
        compression_messages = live_messages
        if tail_start is not None:
            protected_inflight_tail = list(live_messages[tail_start:])
            compression_messages = list(live_messages[:tail_start])
            if not compression_messages:
                return _json_response({
                    "error": "no pre-tool history available to compress",
                    "compressed": False,
                })

        old_session_id = getattr(agent, "session_id", "") or ""
        current_system_prompt = getattr(agent, "_cached_system_prompt", "") or ""
        before_tokens = _estimate_request_tokens(agent, live_messages, current_system_prompt)

        # Pass None as system_message so _compress_context rebuilds from the
        # normal prompt source. Passing the cached full prompt here duplicates
        # identity/instruction blocks on rebuild (same reason /compress passes
        # None in the CLI/TUI manual paths).
        result = _compress_context(
            compression_messages,
            None,
            approx_tokens=before_tokens or None,
            task_id=kw.get("task_id") or getattr(agent, "_current_task_id", None) or "default",
            force=force,
            aggressive=force,  # use aggressive tail when caller passes force=True
            focus_topic=f"Agent-requested: {reason}",
        )

        post_messages = []
        new_system_prompt = current_system_prompt
        if result and isinstance(result, (list, tuple)) and len(result) >= 1:
            if isinstance(result[0], list):
                post_messages = result[0]
            if len(result) >= 2 and isinstance(result[1], str):
                new_system_prompt = result[1]

        if not post_messages:
            return _json_response({
                "compressed": False,
                "reason": reason,
                "tokens_saved": 0,
                "messages_before": pre_count,
                "messages_after": 0,
                "error": "compression returned no messages",
            })

        # Make the compression effective for the *current* tool loop by
        # replacing the mutable live list in-place. The outer run loop holds
        # this exact list and will append the request_compression tool result
        # after this handler returns. If the current assistant tool-call group
        # is already in the transcript, re-attach it after the compressed
        # prefix so the executor can append the real tool result without a
        # synthetic sanitizer stub for the in-flight call.
        if protected_inflight_tail:
            live_messages[:] = list(post_messages) + protected_inflight_tail
        elif post_messages is not live_messages:
            live_messages[:] = post_messages
        try:
            agent._session_messages = live_messages
        except Exception:
            pass
        try:
            agent.conversation_history = live_messages
        except Exception:
            pass
        if new_system_prompt:
            agent._cached_system_prompt = new_system_prompt

        after_tokens = _estimate_request_tokens(agent, live_messages, new_system_prompt)
        dropped = max(0, pre_count - len(live_messages))
        if before_tokens and after_tokens:
            saved_tokens = max(0, before_tokens - after_tokens)
        else:
            # Fallback only when rough estimator is unavailable.
            saved_tokens = dropped * 150

        new_session_id = getattr(agent, "session_id", "") or ""
        applied = bool(dropped or saved_tokens > 0 or new_session_id != old_session_id)
        if applied:
            # The run loop uses this to clear its caller-history baseline and
            # refresh the active system prompt after tool execution.
            agent._request_compression_applied = True

        return _json_response({
            "compressed": applied,
            "applied": applied,
            "reason": reason,
            "tokens_saved": saved_tokens,
            "tokens_before_estimate": before_tokens,
            "tokens_after_estimate": after_tokens,
            "messages_before": pre_count,
            "messages_after": len(live_messages),
            "messages_removed": dropped,
            "protected_inflight_tool_messages": len(protected_inflight_tail),
            "old_session_id": old_session_id,
            "new_session_id": new_session_id,
        })
    except Exception as exc:
        return _json_response({
            "error": f"compression failed: {exc}",
            "compressed": False,
        })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_context_length(agent: Any) -> int:
    """Resolve the effective context length for the current model."""
    # Try context_compressor first
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        cl = getattr(compressor, "context_length", None)
        if cl and isinstance(cl, (int, float)) and cl > 0:
            return int(cl)

    # Fallback: model_metadata helper
    try:
        from agent.model_metadata import get_model_context_length
        model = getattr(agent, "model", None)
        provider = getattr(agent, "provider", None)
        if model:
            return get_model_context_length(model, provider=provider)
    except Exception:
        pass

    return 200_000  # safest fallback — worst case is 200K


def _estimate_request_tokens(agent: Any, messages: list, system_prompt: str = "") -> int:
    """Best-effort rough request token estimate for compression diagnostics."""
    try:
        from agent.model_metadata import estimate_request_tokens_rough

        return int(estimate_request_tokens_rough(
            messages,
            system_prompt=system_prompt or "",
            tools=getattr(agent, "tools", None) or None,
        ))
    except Exception:
        return 0


def _read_compression_config(agent: Any) -> Dict[str, Any]:
    """Read the compression section of the active profile config.

    AIAgent does not expose ``self.config``, so we load it directly
    from ``load_config()`` via the CLI config module.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return dict(cfg.get("compression", {}))
    except Exception:
        return {}


# ============================================================================
# Registry — self-registering module (discovered by tools.registry)
# ============================================================================

from tools.registry import registry  # noqa: E402

# Availability checks are config-gated, not env-var-gated.
def _check_context_status_reqs() -> bool:
    """context_status is safe and read-only, so expose it whenever the toolset is enabled."""
    return True


def _check_request_compression_reqs() -> bool:
    """Expose request_compression only when the user opted into agent-triggered compaction."""
    cfg = _read_compression_config(None)
    return bool(cfg.get("enabled", True) and cfg.get("allow_agent_trigger", False))

# context_status: pure read, no state needed (agent injected at call time)
registry.register(
    name="context_status",
    toolset="context_usage",
    schema=CONTEXT_STATUS_SCHEMA,
    handler=context_status_handler,
    check_fn=_check_context_status_reqs,
    requires_env=[],
    is_async=False,
    description="Check current context window usage for this session",
    emoji="📊",
)

# request_compression: agent can request compression voluntarily
registry.register(
    name="request_compression",
    toolset="context_usage",
    schema=COMPRESS_SCHEMA,
    handler=request_compression_handler,
    check_fn=_check_request_compression_reqs,
    requires_env=[],
    is_async=False,
    description="Request context compression to free up the conversation window",
    emoji="🗜️",
)
