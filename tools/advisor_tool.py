#!/usr/bin/env python3
"""
Advisor Tool — Agent-internal "ask a smarter model for advice" capability.

When the executing model encounters a difficult decision, complex architecture
choice, or repeated failure, it can call ``ask_advisor`` to get guidance from a
more capable model — *without* yielding control.  The advisor returns text
advice; the executor decides what to do with it.

Design inspired by Anthropic's Advisor Strategy:
    https://claude.com/blog/the-advisor-strategy

Key properties
--------------
* **Executor retains control** — the advisor never calls tools or modifies state.
* **Lightweight** — a single chat-completions call (~500-1000 tokens per query).
* **Model-agnostic** — works with any OpenAI-compatible advisor model.
* **Message sanitization** — executor messages (tool_calls, tool roles) are
  cleaned into plain text before sending to the advisor.

Config (config.yaml)::

    advisor:
      enabled: true
      model: "deepseek-chat"          # advisor model (required when enabled)
      provider: "custom:deepseek"     # provider key (optional — uses main provider)
      max_uses_per_task: 5            # per-task invocation cap
      temperature: 0.3                # advisor sampling temperature
      max_tokens: 2048                # max tokens per advisor response
      system_prompt: null             # custom system prompt (optional)
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_DEFAULT_ADVISOR_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "max_uses_per_task": 5,
    "temperature": 0.3,
    "max_tokens": 2048,
    "system_prompt": None,
}


def load_advisor_config() -> Dict[str, Any]:
    """Merge user config onto defaults (config.yaml → env vars → defaults)."""
    cfg = dict(_DEFAULT_ADVISOR_CONFIG)

    # 1) config.yaml ``advisor:`` section
    try:
        from cli import CLI_CONFIG
        user_cfg = CLI_CONFIG.get("advisor") or {}
    except Exception:
        try:
            from hermes_cli.config import load_config
            user_cfg = load_config().get("advisor") or {}
        except Exception:
            user_cfg = {}

    for k, v in user_cfg.items():
        if v is not None:
            cfg[k] = v

    # 2) env-var overrides
    env_model = os.environ.get("HERMES_ADVISOR_MODEL")
    if env_model:
        cfg["model"] = env_model
    env_key = os.environ.get("HERMES_ADVISOR_API_KEY")
    if env_key:
        cfg["api_key"] = env_key

    return cfg


# ---------------------------------------------------------------------------
# Message sanitization
# ---------------------------------------------------------------------------

def _sanitize_for_advisor(messages: List[dict]) -> List[dict]:
    """Strip tool_calls / tool roles so the advisor sees plain text only.

    The advisor model does *not* support tools — feeding it raw executor
    messages with ``tool_calls`` dicts or ``role: "tool"`` would cause API
    errors or garbled output.
    """
    out: List[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # --- system ---
        if role == "system":
            if isinstance(content, str) and content.strip():
                out.append({"role": "system", "content": content})

        # --- user ---
        elif role == "user":
            if isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(texts)
            if isinstance(content, str) and content.strip():
                out.append({"role": "user", "content": content})

        # --- assistant ---
        elif role == "assistant":
            text = content if isinstance(content, str) else ""
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                summaries = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args_preview = str(fn.get("arguments", ""))[:120]
                    summaries.append(f"{name}({args_preview})")
                action_text = "[Actions: " + "; ".join(summaries) + "]"
                text = f"{text}\n{action_text}" if text.strip() else action_text
            if text.strip():
                out.append({"role": "assistant", "content": text})

        # --- tool results → user ---
        elif role == "tool":
            text = content if isinstance(content, str) else str(content)
            # Truncate very long tool outputs
            if len(text) > 600:
                text = text[:600] + "…(truncated)"
            if text.strip():
                out.append({"role": "user", "content": f"[Tool output] {text}"})

    return out


# ---------------------------------------------------------------------------
# Default advisor system prompt
# ---------------------------------------------------------------------------

_DEFAULT_ADVISOR_SYSTEM_PROMPT = """\
You are a senior technical advisor. An AI assistant (the "executor") is working \
on a task and has asked for your guidance.

Your job:
1. Analyse the current progress and identify the core difficulty.
2. Give concrete, actionable advice the executor can follow immediately.
3. Point out blind spots or risks the executor may have missed.
4. If the current approach is sound, confirm briefly — no need to pad.

Rules:
- You do NOT execute anything. Only give advice.
- Be specific enough that the executor can act on your suggestion directly.
- Keep it concise — the executor pays per token.
- If the question is unclear, ask for clarification rather than guessing."""


# ---------------------------------------------------------------------------
# Core advisor call
# ---------------------------------------------------------------------------

def call_advisor(
    *,
    messages: List[dict],
    question: str,
    urgency: str,
    config: Dict[str, Any],
    credential_pool: Any = None,
    parent_agent: Any = None,
) -> str:
    """Call the advisor model and return its guidance as a string.

    Returns a JSON string with ``advice``, ``model``, ``tokens_in``,
    ``tokens_out``, and ``latency_ms`` so the executor can see the cost.
    """
    import openai

    # ---- Resolve API credentials ----
    advisor_model = config.get("model")
    if not advisor_model:
        return json.dumps({"error": "advisor.model is not configured — set it in config.yaml or HERMES_ADVISOR_MODEL"})

    # Try credential pool first (supports key rotation)
    api_key = config.get("api_key")
    base_url = config.get("base_url")

    if credential_pool and not api_key:
        try:
            provider = config.get("provider", "")
            # strip "custom:" prefix if present
            pool_name = provider.replace("custom:", "") if provider else ""
            cred = credential_pool.get_credentials(pool_name) if pool_name else None
            if cred:
                api_key = cred.get("api_key") or cred.get("key")
                base_url = base_url or cred.get("base_url")
        except Exception:
            pass

    # Fall back to parent agent's credentials if advisor shares the same provider
    if not api_key and parent_agent:
        parent_provider = getattr(parent_agent, "provider", "") or ""
        advisor_provider = config.get("provider", "") or ""
        if not advisor_provider or parent_provider == advisor_provider:
            api_key = getattr(parent_agent, "api_key", None)
            base_url = base_url or getattr(parent_agent, "base_url", None)

    if not api_key:
        return json.dumps({"error": "No API key found for advisor — configure advisor.api_key or advisor.provider in config.yaml"})

    # ---- Sanitize executor messages ----
    clean = _sanitize_for_advisor(messages)
    # Keep a rolling window of the last 20 messages to stay within context
    context_msgs = clean[-20:]

    # ---- Build advisor prompt ----
    advisor_messages: List[dict] = [
        {"role": "system", "content": config.get("system_prompt") or _DEFAULT_ADVISOR_SYSTEM_PROMPT},
    ]

    if context_msgs:
        context_summary = json.dumps(
            [{"role": m["role"], "content": m["content"][:300]} for m in context_msgs],
            ensure_ascii=False,
        )
        advisor_messages.append({
            "role": "user",
            "content": f"Executor's conversation context (recent):\n{context_summary}",
        })

    advisor_messages.append({
        "role": "user",
        "content": f"Urgency: {urgency}\n\nMy question: {question}",
    })

    # ---- API call ----
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    start = time.time()
    try:
        client = openai.OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=advisor_model,
            messages=advisor_messages,
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 2048),
        )
    except Exception as exc:
        logger.warning("advisor API call failed: %s", exc)
        return json.dumps({"error": f"Advisor API call failed: {exc}"})

    latency_ms = int((time.time() - start) * 1000)
    choice = resp.choices[0]
    advice = choice.message.content or "(advisor returned empty response)"
    usage = resp.usage

    return json.dumps({
        "advice": advice,
        "model": advisor_model,
        "tokens_in": usage.prompt_tokens if usage else 0,
        "tokens_out": usage.completion_tokens if usage else 0,
        "latency_ms": latency_ms,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_advisor_requirements() -> bool:
    """Return True only when advisor is explicitly enabled in config."""
    return load_advisor_config().get("enabled", False)


# ---------------------------------------------------------------------------
# Per-task invocation counter (reset per conversation)
# ---------------------------------------------------------------------------

def _make_use_counter():
    """Return a simple mutable counter dict."""
    return {"count": 0}


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# ---------------------------------------------------------------------------

ASK_ADVISOR_SCHEMA = {
    "name": "ask_advisor",
    "description": (
        "Ask a more capable AI model for advice. Use when you are unsure of the "
        "best approach, facing a complex architecture decision, or stuck after "
        "multiple failed attempts. The advisor sees your conversation context and "
        "returns actionable guidance — it does NOT execute anything. "
        "State: (1) what you've tried, (2) the difficulty, (3) what kind of "
        "advice you need."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Your specific question for the advisor. Include: "
                    "current progress, what's difficult, and the type of guidance you need."
                ),
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": (
                    "How badly you need help. "
                    "low = curious for a second opinion; "
                    "medium = would appreciate guidance; "
                    "high = stuck, need direction to proceed."
                ),
            },
        },
        "required": ["question"],
    },
}


# ---------------------------------------------------------------------------
# Registry entry — auto-discovered at import time
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error  # noqa: E402

# The handler is a no-op placeholder; actual dispatch happens in
# ``agent.agent_runtime_helpers.invoke_tool`` because the advisor needs
# access to agent-level state (messages, credential pool).
registry.register(
    name="ask_advisor",
    toolset="advisor",
    schema=ASK_ADVISOR_SCHEMA,
    handler=lambda args, **kw: tool_error(
        "ask_advisor", "must be dispatched via invoke_tool (agent-level tool)"
    ),
    check_fn=check_advisor_requirements,
    emoji="🧠",
)
