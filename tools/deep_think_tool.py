"""Deep thinking tool — on-demand reasoning via a heavyweight model.

The main agent stays on its fast model with prompt cache intact.
When it encounters a problem that needs deeper reasoning (complex
debugging, architectural decisions, mathematical proofs, multi-step
logic), it calls this tool to consult a stronger model.

The deep model sees only the focused question — not the full
conversation history — so context stays tight and cost stays low.

Config: auxiliary.deep_think in config.yaml controls the provider
and model. Falls back to the main provider's best available model.
"""

import json
import logging
import time
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)

_DEEP_THINK_SYSTEM = (
    "You are a deep reasoning assistant. You have been called because the "
    "primary agent determined this problem requires careful, thorough analysis. "
    "Think step by step. Be precise and rigorous. If you're uncertain about "
    "something, say so explicitly rather than guessing. Focus your response on "
    "the specific question asked — don't repeat background the caller already knows."
)


def deep_think(question: str, context: str = "", **kwargs) -> str:
    """Call a deep reasoning model with a focused question.

    Args:
        question: The specific problem or question requiring deep analysis.
        context: Optional relevant context (code snippet, error message, etc.).

    Returns:
        JSON string with the reasoning model's analysis.
    """
    from agent.auxiliary_client import call_llm

    if not question or not question.strip():
        return json.dumps({"error": "No question provided"})

    user_content = question.strip()
    if context and context.strip():
        user_content = f"{user_content}\n\nContext:\n{context.strip()}"

    messages = [
        {"role": "system", "content": _DEEP_THINK_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    start = time.monotonic()
    try:
        response = call_llm(
            task="deep_think",
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
    except RuntimeError:
        # No auxiliary provider available — try with explicit main provider
        # passthrough so it at least works with whatever the user has.
        try:
            response = call_llm(
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as e:
            return json.dumps({
                "error": f"No model available for deep thinking: {e}",
                "suggestion": "Configure auxiliary.deep_think in config.yaml with a reasoning-capable model.",
            })

    elapsed_ms = int((time.monotonic() - start) * 1000)

    content = ""
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        content = getattr(msg, "content", "") or ""
        # Some reasoning models put analysis in a reasoning field
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        if reasoning and not content:
            content = reasoning
        elif reasoning:
            content = f"{reasoning}\n\n---\n\n{content}"

    resp_model = getattr(response, "model", "unknown")

    result = {
        "analysis": content,
        "model": resp_model,
        "elapsed_ms": elapsed_ms,
    }

    logger.info("deep_think completed in %dms via %s (%d chars)",
                elapsed_ms, resp_model, len(content))

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

DEEP_THINK_SCHEMA = {
    "name": "deep_think",
    "description": (
        "Consult a stronger reasoning model on a focused question. "
        "Use when a problem needs rigorous step-by-step analysis: "
        "root-cause debugging, architectural trade-offs, math proofs, "
        "or multi-step plans where mistakes are costly. "
        "Send only the question and minimal context — not the full conversation. "
        "Returns analysis you can summarize for the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The specific question or problem requiring deep analysis. "
                    "Be precise about what you need help with."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional relevant context: a code snippet, error message, "
                    "data sample, or constraint list. Keep it focused — only "
                    "include what the reasoning model needs to answer the question."
                ),
            },
        },
        "required": ["question"],
    },
}


# ---------------------------------------------------------------------------
# Check function — tool is available when any auxiliary provider is configured
# ---------------------------------------------------------------------------

def _check_deep_think() -> bool:
    try:
        from agent.auxiliary_client import _resolve_task_provider_model
        provider, model, base_url, api_key, _ = _resolve_task_provider_model("deep_think")
        if provider and provider != "auto":
            return True
        # "auto" means it will try the fallback chain — still likely to work
        return True
    except Exception:
        return True  # fail-open: let the actual call surface the error


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="deep_think",
    toolset="delegation",  # consult a heavier model — groups with delegate_task
    schema=DEEP_THINK_SCHEMA,
    handler=lambda args, **kw: deep_think(
        question=args.get("question", ""),
        context=args.get("context", ""),
    ),
    check_fn=_check_deep_think,
    emoji="🧠",
)
