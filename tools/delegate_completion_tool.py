"""``delegate_completion`` tool — cheap one-shot text completion via the existing auxiliary router.

Companion to ``delegate_task`` for the "no tools, no agent loop, no system prompt"
case: pure text transforms, classification, summarization, reformatting. Routes
through :mod:`agent.auxiliary_client`, so it inherits every provider the
auxiliary chain already resolves (OpenRouter, Nous Portal, native Anthropic,
direct API-key providers, ``provider: "main"`` for local/Ollama, or any
OpenAI-compatible ``base_url`` configured under ``auxiliary.delegate_completion``).

Why a separate tool rather than reaching into ``delegate_task``:
  * One ``delegate_task`` call spins up a full child AIAgent with its own
    system prompt, tool schemas in context, multi-turn iteration loop,
    file-state tracking. That's a lot of overhead for a prompt-in /
    completion-out task. ``delegate_completion`` does the same single
    completion call the auxiliary subsystem already performs for
    compression / vision / web-extract, just exposed at the tool layer.
  * Routing decisions stay in config. Users add an
    ``auxiliary.delegate_completion`` block to point this tool at
    OpenRouter, a specific paid API key, or a local model. New provider
    integrations land as detection/fallback updates in
    ``agent/auxiliary_client.py`` and apply here for free.

Configured at::

    auxiliary:
      delegate_completion:
        provider: openrouter       # main | openrouter | nous | custom | anthropic
        model:    "xai/grok-mini"
        timeout:  60               # optional, seconds
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Tool description surfaced to the model. Pushed to ``delegate_task`` when the
# sub-task genuinely needs tools or multi-turn reasoning. Keep it short and
# direct — the agent decides which tool to call based on this string.
TOOL_DESCRIPTION = (
    "Run a SINGLE text-completion call on a configured backend without "
    "spawning a subagent. Use this for text-transform / classification / "
    "summarization / reformatting work that does NOT need terminal, files, "
    "web, or multi-turn reasoning. Pass either a single prompt (\"prompt\") "
    "or a JSON array of (\"batch\") prompts; in batch mode the responses are "
    "returned in the same order. Resolves the configured backend from "
    "`auxiliary.delegate_completion.provider` (default: \"main\") and "
    "`.model` (optional). For tasks that need tools or conversation "
    "memory, use `delegate_task` instead."
)


def _normalize_prompts(prompts: Optional[List[Any]], prompt: Optional[Any]) -> List[str]:
    """Collapse the singular ``prompt`` and plural ``batch`` shapes into a list."""
    if prompt is not None and prompts is not None:
        raise ValueError(
            "Pass exactly one of `prompt` (str) or `batch` (list of strings); "
            "both were provided."
        )
    if prompts is None and prompt is None:
        raise ValueError("Either `prompt` (str) or `batch` (list of strings) is required.")
    if prompts is not None:
        if isinstance(prompts, str):
            # Tolerate "batch=\"single string\"" by treating it as a singleton.
            return [prompts]
        if not isinstance(prompts, list):
            raise ValueError(
                "`batch` must be a list of strings "
                f"(got {type(prompts).__name__})."
            )
        return [str(p) for p in prompts]
    return [str(prompt)]


def _call_one(
    text: str,
    *,
    timeout: Optional[float],
    system: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> str:
    """Hit the auxiliary router once. Returns the assistant text content.

    Provider and model are resolved exclusively from the
    ``auxiliary.delegate_completion`` config via ``call_llm(task=...)``;
    this tool intentionally does not expose per-call overrides.
    """
    from agent.auxiliary_client import call_llm

    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})

    # ``task="delegate_completion"`` makes ``call_llm`` read
    # ``auxiliary.delegate_completion`` config for provider/model/extra_body
    # by name. Routing stays entirely in config — there are no per-call
    # overrides on purpose.
    response = call_llm(
        "delegate_completion",
        messages=messages,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    from agent.auxiliary_client import (
        extract_content_or_reasoning,
    )

    # ``call_llm`` returns a normalized chat-completions response with
    # ``choices[0].message``; ``extract_content_or_reasoning`` reads that
    # shape (and falls back to reasoning fields for models that emit
    # content=None), so it works for every provider the auxiliary chain
    # resolves — Responses-API wrappers, OpenRouter, native Anthropic,
    # OpenAI-compatible backends, etc. Using
    # ``_extract_aux_response_text`` here would silently return ``""``
    # for any standard provider.
    return extract_content_or_reasoning(response) or ""


def delegate_completion(
    prompt: str = None,
    batch=None,
    *,
    timeout: float = None,
    system: str = None,
    temperature: float = None,
    max_tokens: int = None,
    task_id: str = None,
) -> str:
    """Run a single prompt (or a small batch of prompts) through the auxiliary router.

    Routing (provider + model) is resolved exclusively from the
    ``auxiliary.delegate_completion`` config block — this tool deliberately
    does NOT expose ``provider``/``model`` overrides, consistent with the
    standing delegation-model-routing policy. Override routing by editing
    ``config.yaml``, never by passing it through the tool.

    Args:
        prompt: A single user-side prompt. Mutually exclusive with ``batch``.
        batch:  A list of prompts to run with the same routing; responses are
                returned in the same order. Keeps the tool usable from any
                provider that disallows nested rounds.
        timeout:  Seconds. Falls back to ``auxiliary.delegate_completion.timeout``.
        system:   Optional system message prepended to the prompt.
        temperature: Optional sampling temperature.
        max_tokens:  Optional response cap. Honor ``agent.auxiliary_client``
                     semantics (``max_tokens`` vs ``max_completion_tokens``).
        task_id:  Optional task identifier used to probe the active backend.

    Returns:
        JSON string. On success::

            {
                "success": true,
                "results": [
                    {... "text": "<assistant text>" ...},
                    ...
                ],
                "count": <int>
            }

        ``results`` carries ONE entry for the ``prompt`` shape and N entries
        for the ``batch`` shape, in the same order as input. On failure::

            {"success": false, "error": "<description>"}
    """
    try:
        prompts = _normalize_prompts(batch, prompt)
    except ValueError as e:
        return json.dumps({"success": False, "error": str(e)})

    try:
        outputs: List[Dict[str, Any]] = []
        for text in prompts:
            text_out = _call_one(
                text,
                timeout=timeout,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            outputs.append({"text": text_out, "input": text})
        return json.dumps(
            {"success": True, "results": outputs, "count": len(outputs)},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("delegate_completion failed: %s", e, exc_info=True)
        return json.dumps(
            {"success": False, "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=False,
        )


__all__ = [
    "TOOL_DESCRIPTION",
    "delegate_completion",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Tool schema is the JSON_schema one-pass emits to the LLM. The agent
# decides between this tool and ``delegate_task`` based on the description
# string below. Mirrors the documented "use this for no-tools text
# transforms, use delegate_task for tool-using agents" guidance from #59070.
_DELEGATE_COMPLETION_SCHEMA = {
    "name": "delegate_completion",
    "description": TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "A single prompt. Use when there's exactly one prompt "
                    "to resolve. Mutually exclusive with ``batch``."
                ),
            },
            "batch": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of prompts to run through the same backend. "
                    "Responses are returned in the same order as input. "
                    "Mutually exclusive with ``prompt``."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (overrides config).",
            },
            "system": {
                "type": "string",
                "description": (
                    "Optional system message prepended to the user prompt."
                ),
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens.",
            },
        },
        "required": [],
    },
}


def _delegate_completion_check():
    """Delegate completion is always available — gating it on a provider
    would make the tool itself disappear from the schema on cold start,
    leaving the LLM without a discoverable cheap-completion path. The
    underlying ``call_llm`` raises when no provider is reachable, which
    surfaces as a tool error and is more informative than a missing tool.
    """
    return True


try:
    from tools.registry import registry, tool_error  # type: ignore

    registry.register(
        name="delegate_completion",
        toolset="delegation",
        schema=_DELEGATE_COMPLETION_SCHEMA,
        handler=lambda args, **kw: delegate_completion(
            prompt=args.get("prompt"),
            batch=args.get("batch"),
            provider=args.get("provider"),
            model=args.get("model"),
            timeout=args.get("timeout"),
            system=args.get("system"),
            temperature=args.get("temperature"),
            max_tokens=args.get("max_tokens"),
            task_id=args.get("task_id") or kw.get("task_id"),
        ),
        check_fn=_delegate_completion_check,
        emoji="⚡",
    )
except (ImportError, Exception) as e:
    # Tool registry hosts run inside the CLI; defer registration until
    # the in-process runtime wires the registry. The ToolEntry exists
    # either way; this branch keeps import-side-effects safe during
    # unit tests and ad-hoc repls that don't import ``tools.registry``.
    logger.debug(
        "delegate_completion: registry.attach skipped (%s)", e,
    )
