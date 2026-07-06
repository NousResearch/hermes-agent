"""delegate_completion tool: one-shot text completion via the auxiliary provider chain.

Companion to ``delegate_task`` for work that needs no tools, no agent loop,
and no subagent context: summarization, classification, reformatting,
extraction. Each prompt becomes a single ``call_llm()`` request routed by the
existing auxiliary resolution chain, so every provider that chain resolves
(OpenRouter, Nous Portal, native Anthropic, direct API-key providers,
``provider: "main"`` for local/OpenAI-compatible endpoints) works here with
no new integration code.

Config lives at ``auxiliary.delegate_completion`` (``provider`` / ``model`` /
``timeout``), the same per-task shape as ``auxiliary.vision`` and
``auxiliary.web_extract``. See #59070.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, extract_content_or_reasoning
from tools.registry import registry

logger = logging.getLogger(__name__)

_TASK = "delegate_completion"


DELEGATE_COMPLETION_SCHEMA = {
    "name": "delegate_completion",
    "description": (
        "Run one-shot text completions on the configured auxiliary backend "
        "without spawning a subagent. Use for self-contained text transforms "
        "-- summarize, classify, reformat, extract, draft -- that need no "
        "tools, files, web access, or follow-up turns; for those, use "
        "delegate_task instead. Prompts must be fully self-contained: the "
        "backend model sees nothing from this conversation. Backend is "
        "resolved from auxiliary.delegate_completion config (provider/model/"
        "timeout), falling back to the auto-detected auxiliary provider."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "A single self-contained prompt. Mutually exclusive "
                    "with `batch`."
                ),
            },
            "batch": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of self-contained prompts, each run as its own "
                    "completion on the same backend. Results come back in "
                    "input order. Mutually exclusive with `prompt`."
                ),
            },
            "system": {
                "type": "string",
                "description": (
                    "Optional system message applied to every prompt in "
                    "the call."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "Override the configured provider for this call "
                    "(e.g. 'openrouter', 'nous', 'main', 'anthropic')."
                ),
            },
            "model": {
                "type": "string",
                "description": "Override the configured model for this call.",
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Request timeout in seconds. Defaults to "
                    "auxiliary.delegate_completion.timeout."
                ),
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature (provider default when omitted).",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (provider default when omitted).",
            },
        },
        "required": [],
    },
}


def _usage_error(message: str) -> str:
    return json.dumps({"success": False, "error": message})


def delegate_completion(
    prompt: Optional[str] = None,
    batch: Optional[List[str]] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Run one or more prompts as plain completions through the auxiliary router.

    Returns a JSON string. Results are in input order; each entry carries
    either ``text`` (the completion) or ``error`` (that prompt failed)::

        {"success": true, "count": 2, "failed": 0,
         "results": [{"text": "...", "model": "..."}, ...]}

    ``success`` is true only when every prompt succeeded. Argument-shape
    errors return ``{"success": false, "error": "..."}`` with no results.
    """
    if prompt is not None and batch is not None:
        return _usage_error("Pass either `prompt` or `batch`, not both.")
    if prompt is None and batch is None:
        return _usage_error("One of `prompt` or `batch` is required.")

    if batch is not None:
        if isinstance(batch, str):
            # Tolerate a bare string in the batch slot.
            prompts = [batch]
        elif isinstance(batch, list) and batch:
            prompts = [str(p) for p in batch]
        else:
            return _usage_error("`batch` must be a non-empty list of strings.")
    else:
        prompts = [str(prompt)]

    base_messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": system}] if system else []
    )

    results: List[Dict[str, Any]] = []
    failed = 0
    for i, text in enumerate(prompts):
        try:
            response = call_llm(
                _TASK,
                messages=base_messages + [{"role": "user", "content": text}],
                provider=provider,
                model=model,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(
                {
                    "text": extract_content_or_reasoning(response),
                    "model": getattr(response, "model", None) or model,
                }
            )
        except Exception as e:
            failed += 1
            logger.warning(
                "delegate_completion: prompt %d/%d failed: %s",
                i + 1, len(prompts), e,
            )
            results.append({"error": f"{type(e).__name__}: {e}"})

    return json.dumps(
        {
            "success": failed == 0,
            "count": len(results),
            "failed": failed,
            "results": results,
        },
        ensure_ascii=False,
    )


# No check_fn: the auxiliary router auto-resolves a backend at call time and
# raises an informative error when none is reachable, which surfaces as a
# tool error. Gating on provider detection at schema-build time would make
# the tool silently vanish instead.
registry.register(
    name="delegate_completion",
    toolset="delegation",
    schema=DELEGATE_COMPLETION_SCHEMA,
    handler=lambda args, **kw: delegate_completion(
        prompt=args.get("prompt"),
        batch=args.get("batch"),
        provider=args.get("provider"),
        model=args.get("model"),
        timeout=args.get("timeout"),
        system=args.get("system"),
        temperature=args.get("temperature"),
        max_tokens=args.get("max_tokens"),
    ),
    emoji="⚡",
)
