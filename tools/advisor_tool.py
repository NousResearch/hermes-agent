#!/usr/bin/env python3
"""
Advisor Tool — Agent-internal "ask a smarter model for advice" capability.

When the executing model encounters a difficult decision, complex architecture
choice, or repeated failure, it can call ``ask_advisor`` to get guidance from a
more capable model — *without* yielding control.  The advisor returns text
advice; the executor decides what to do with it.

Design based on Anthropic's Advisor Strategy:
    https://claude.com/blog/the-advisor-strategy
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool

Key properties
--------------
* **Executor retains control** — the advisor never calls tools or modifies state.
* **Lightweight** — a single chat-completions call (~400–700 text tokens per query).
* **Model-agnostic** — works with any OpenAI-compatible advisor model.
* **Message sanitization** — executor messages (tool_calls, tool roles) are
  cleaned into plain text before sending to the advisor.
* **Output trimming** — advisor is instructed to keep guidance concise (under
  ~80 words target) to minimize token cost, following Anthropic's recommendation.

Config (config.yaml)::

    advisor:
      model: "deepseek-v4-pro"           # advisor model (defaults to agent's own model)
      max_uses_per_task: 5                # per-task invocation cap (default: 5)
      max_context_messages: 20            # rolling window of recent messages
      max_context_chars: 300              # per-message content truncation length
      max_tool_output_chars: 600          # tool result truncation length
      timeout: 30                         # API call timeout (seconds)
      temperature: 0.3                    # advisor sampling temperature
      max_tokens: 2048                    # max tokens per advisor response
      system_prompt: null                 # custom system prompt (optional)

Zero-config: if no ``advisor:`` section exists in config.yaml, the tool
automatically inherits the agent's own model, API key, and base URL.
Only set explicit values when you want a *different* model as advisor.
Set ``advisor.enabled: false`` to hide the tool entirely (not recommended).
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
    "max_context_messages": 20,     # rolling window of recent messages
    "max_context_chars": 300,       # per-message content truncation
    "max_tool_output_chars": 600,   # tool result truncation
    "timeout": 30,                  # API call timeout (seconds)
    "temperature": 0.3,
    "max_tokens": 2048,
    "system_prompt": None,
}


def load_advisor_config() -> Dict[str, Any]:
    """Merge user config onto defaults (config.yaml -> env vars -> defaults)."""
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

def _sanitize_for_advisor(messages: List[dict], max_tool_output_chars: int = 600) -> List[dict]:
    """Strip tool_calls / tool roles so the advisor sees plain text only.

    The advisor model does *not* support tools — feeding it raw executor
    messages with ``tool_calls`` dicts or ``role: "tool"`` would cause API
    errors or garbled output.

    Args:
        messages: Executor conversation messages.
        max_tool_output_chars: Truncation limit for tool result content.
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

        # --- tool results -> assistant-annotated user context ---
        elif role == "tool":
            text = content if isinstance(content, str) else str(content)
            if len(text) > max_tool_output_chars:
                text = text[:max_tool_output_chars] + "...(truncated)"
            if text.strip():
                out.append({"role": "user", "content": f"[Tool result]\n{text}"})

    return out


# ---------------------------------------------------------------------------
# Default advisor system prompt
# Based on Anthropic's recommended advisor system prompt:
# https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
# ---------------------------------------------------------------------------

_DEFAULT_ADVISOR_SYSTEM_PROMPT = """\
You are a senior technical advisor. An AI coding assistant (the "executor") is \
working on a task and has asked for your guidance.

You see the executor's full conversation context: the task, every action taken, \
and every result observed. Your job is to provide focused, actionable guidance.

Rules:
- You do NOT execute anything. Only give advice.
- Be concise and concrete — the executor needs to act, not read.
- If the current approach is sound, confirm briefly and point out pitfalls to watch for.
- If the approach has issues, explain specifically what to change and why.
- If the question is unclear, ask for clarification rather than guessing.
- Target under 80 words unless the complexity genuinely requires more."""


# ---------------------------------------------------------------------------
# Executor system prompt appendix
# Based on Anthropic's suggested system prompt for coding tasks:
# https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool#suggested-system-prompt-for-coding-tasks
#
# This text should be injected into the executor's system prompt when
# ask_advisor is available in the tool list.
# ---------------------------------------------------------------------------

EXECUTOR_ADVISOR_PROMPT = """\

Timing guidance: You have access to an `ask_advisor` tool backed by a stronger \
reviewer model. When you call ask_advisor, your conversation history is \
automatically forwarded — they see the task, every tool call you've made, \
every result you've seen.

Call ask_advisor BEFORE substantive work — before writing, before committing \
to an interpretation, before building on an assumption. If the task requires \
orientation first (finding files, fetching a source, seeing what's there), do \
that, then call ask_advisor. Orientation is not substantive work. Writing, \
editing, and declaring an answer are.

Also call ask_advisor:
- When you believe the task is complete. BEFORE this call, make your \
deliverable durable: write the file, save the result, commit the change. The \
advisor call takes time; if the session ends during it, a durable result \
persists and an unwritten one doesn't.
- When stuck — errors recurring, approach not converging, results that don't fit.
- When considering a change of approach.

On tasks longer than a few steps, call ask_advisor at least once before \
committing to an approach and once before declaring done. On short reactive \
tasks where the next action is dictated by tool output you just read, you \
don't need to keep calling — the advisor adds most of its value on the first \
call, before the approach crystallizes.

How to treat the advice: Give the advice serious weight. If you follow a step \
and it fails empirically, or you have primary-source evidence that contradicts \
a specific claim (the file says X, the paper states Y), adapt. A passing \
self-test is not evidence the advice is wrong — it's evidence your test \
doesn't check what the advice is checking.

If you've already retrieved data pointing one way and the advisor points \
another: don't silently switch. Surface the conflict in one more advisor call — \
"I found X, you suggest Y, which constraint breaks the tie?" The advisor saw \
your evidence but may have underweighted it; a reconcile call is cheaper than \
committing to the wrong branch."""


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

    # ---- Resolve model ----
    advisor_model = config.get("model")
    if not advisor_model and parent_agent:
        # Zero-config: inherit model from the executing agent itself
        advisor_model = getattr(parent_agent, "model", None)
    if not advisor_model:
        return json.dumps({"error": "advisor.model is not configured — set it in config.yaml or HERMES_ADVISOR_MODEL"})

    # ---- Resolve API credentials ----
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

    # Fall back to parent agent's credentials (zero-config path)
    if not api_key and parent_agent:
        api_key = getattr(parent_agent, "api_key", None)
        base_url = base_url or getattr(parent_agent, "base_url", None)
        # If parent also has a credential pool, try that too
        if not api_key:
            try:
                parent_pool = getattr(parent_agent, "_credential_pool", None)
                if parent_pool:
                    parent_provider = getattr(parent_agent, "provider", "")
                    cred = parent_pool.get_credentials(parent_provider) if parent_provider else None
                    if cred:
                        api_key = cred.get("api_key") or cred.get("key")
            except Exception:
                pass

    if not api_key:
        return json.dumps({"error": "No API key found for advisor — configure advisor.api_key or advisor.provider in config.yaml"})

    # ---- Sanitize executor messages ----
    max_tool_chars = config.get("max_tool_output_chars", 600)
    clean = _sanitize_for_advisor(messages, max_tool_output_chars=max_tool_chars)

    # ---- Rolling window ----
    max_context_msgs = config.get("max_context_messages", 20)
    context_msgs = clean[-max_context_msgs:]

    # ---- Build advisor prompt ----
    system_content = config.get("system_prompt") or _DEFAULT_ADVISOR_SYSTEM_PROMPT
    advisor_messages: List[dict] = [
        {"role": "system", "content": system_content},
    ]

    if context_msgs:
        max_chars = config.get("max_context_chars", 300)
        safe_context = []
        for m in context_msgs:
            content = (m.get("content") or "")[:max_chars]
            safe_context.append({"role": m["role"], "content": content})
        context_summary = json.dumps(safe_context, ensure_ascii=False)
        advisor_messages.append({
            "role": "user",
            "content": f"Executor's conversation context (recent):\n{context_summary}",
        })

    # ---- User question with urgency ----
    user_content = f"Urgency: {urgency}\n\nMy question: {question}"

    # Anthropic's recommended trimming technique: embed a direct instruction
    # to the advisor in the user message for concise output. They found this
    # more effective than relying on max_tokens alone.
    # https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool#trimming-advisor-output-length
    user_content += "\n\n(Advisor: please keep your guidance under 80 words — I need a focused starting point, not a comprehensive plan.)"

    advisor_messages.append({
        "role": "user",
        "content": user_content,
    })

    # ---- API call ----
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    # Timeout: build from config
    timeout = config.get("timeout", 30)
    client_kwargs["timeout"] = timeout

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
    advice = choice.message.content or ""
    # GLM-5.1 / DeepSeek thinking mode: content may be empty, fallback to reasoning_content
    if not advice.strip() and hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
        advice = choice.message.reasoning_content
    usage = resp.usage

    return json.dumps({
        "advice": advice,
        "model": advisor_model,
        "tokens_in": usage.prompt_tokens if usage else 0,
        "tokens_out": usage.completion_tokens if usage else 0,
        "latency_ms": latency_ms,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-task invocation counter (reset per task)
# ---------------------------------------------------------------------------

def _make_use_counter():
    """Return a simple mutable counter dict."""
    return {"count": 0}


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# Updated based on Anthropic's recommended "no parameters" approach:
# The original Anthropic advisor tool takes NO parameters — the server
# constructs the advisor's view automatically. Since we're client-side,
# we keep `question` and `urgency` for non-Claude models that benefit from
# explicit prompts, but the description emphasizes that context is automatic.
# ---------------------------------------------------------------------------

ASK_ADVISOR_SCHEMA = {
    "name": "ask_advisor",
    "description": (
        "Ask a more capable AI model for guidance. Your full conversation "
        "history is automatically forwarded — they see the task, every action "
        "you've taken, and every result. Call BEFORE substantive work (writing, "
        "committing to an approach) and BEFORE declaring done. Also call when "
        "stuck or considering a change of approach. State what you need guidance on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What you need guidance on. Be specific: current progress, "
                    "what's difficult, and the type of advice you need."
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
    emoji="\U0001f9e0",
)
