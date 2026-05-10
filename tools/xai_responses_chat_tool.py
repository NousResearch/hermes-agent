"""xAI Responses API tool — single-shot chat call exposing the distinctive
features of ``POST /v1/responses`` that aren't surfaced by the regular
``codex_responses`` transport: ``store=true``, ``previous_response_id``,
configurable ``max_turns`` and ``parallel_tool_calls``.

Why expose this as a standalone tool
------------------------------------
The main Hermes transport for xAI uses Responses API under the hood
(``api_mode='codex_responses'``), but it hardcodes ``store=False``, never
threads ``previous_response_id``, and exposes only ``reasoning.effort``
through the normal config knobs. That's the right default — the transport
is for stateless interactive turns where the client owns the conversation
history.

But the Responses API also has a *stateful* mode designed exactly for the
scenarios Hermes excels at: long autonomous sessions, parallel sub-agents,
agentic tool-calling loops with budget caps. Storing the conversation
server-side via ``store=true`` and chaining turns with
``previous_response_id`` saves re-uploading the full context each turn —
the longer the session, the bigger the saving.

This tool gives an agent (or a delegating orchestrator) direct access to
that stateful surface without changing the global transport behavior.

Reference
---------
- POST /v1/responses — https://docs.x.ai/docs/api-reference#responses-create
- store: 30-day server-side retention per xAI docs.
- previous_response_id: continues a conversation without re-uploading context.
- max_turns: caps agentic tool-calling loops (server-side budget).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import httpx

from tools.xai_http import hermes_xai_user_agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
DEFAULT_MODEL = "grok-4.3"
DEFAULT_TIMEOUT_SECONDS = 300

VALID_REASONING_EFFORTS = {"none", "low", "medium", "high"}


def _config_section() -> Dict[str, Any]:
    """Read the ``xai_responses:`` block from the active Hermes config, if any."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return {}
    section = cfg.get("xai_responses") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def _resolve(name: str, default: Any) -> Any:
    section = _config_section()
    val = section.get(name)
    if val is None or (isinstance(val, str) and not val.strip()):
        return default
    return val


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class XaiResponsesError(RuntimeError):
    """Raised when a Responses API call fails."""


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

def xai_responses_chat(
    prompt: Optional[str] = None,
    *,
    input: Optional[Union[str, List[Dict[str, Any]]]] = None,
    instructions: Optional[str] = None,
    model: Optional[str] = None,
    store: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
    max_turns: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    user: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single-shot call to xAI's ``POST /v1/responses``.

    Parameters
    ----------
    prompt : str, optional
        Convenience: when provided, used as a single-message string ``input``.
        Mutually exclusive with ``input``.
    input : str or list of dict, optional
        Raw Responses API ``input`` payload (string or array of input items).
    instructions : str, optional
        System prompt. xAI's Responses API alternative to a system message.
    model : str, optional
        xAI model identifier. Defaults to config or ``"grok-4.3"``.
    store : bool, optional
        Persist input + response server-side for 30 days (enables
        ``previous_response_id`` chaining). Default: server default
        (currently False on xAI).
    previous_response_id : str, optional
        Continues a stored conversation without re-uploading prior context.
        Requires the prior call to have used ``store=True``.
    max_turns : int, optional
        Cap on the agentic tool-calling loop. Ignored for non-agentic calls.
    parallel_tool_calls : bool, optional
        Allow the model to dispatch multiple tool calls in parallel.
    reasoning_effort : str, optional
        One of ``"none"``, ``"low"``, ``"medium"``, ``"high"``. Translated to
        the ``reasoning: {effort: ...}`` payload.
    max_output_tokens, temperature, top_p, user
        Standard parameters passed through unchanged.
    tools, tool_choice
        Standard tool-calling parameters passed through unchanged.
    timeout_seconds : int, optional
        HTTP timeout for the single POST call. Defaults to 300s.
    extra_body : dict, optional
        Extra fields merged into the request body (e.g. ``search_parameters``,
        ``service_tier``, ``include``).

    Returns
    -------
    dict
        ``{"response_id", "status", "output_text", "raw"}`` where:
        - ``response_id``: id of this response (use for chaining).
        - ``status``: ``"completed"`` | ``"in_progress"`` | ``"incomplete"``.
        - ``output_text``: best-effort extraction of plain text output.
        - ``raw``: full Responses API body (``id``, ``output``, ``usage``...).

    Raises
    ------
    XaiResponsesError
        On HTTP errors, malformed responses, or invalid arguments.
    """
    if (prompt is None or not str(prompt).strip()) and input is None:
        raise ValueError("either 'prompt' (str) or 'input' must be provided")
    if prompt is not None and input is not None:
        raise ValueError("'prompt' and 'input' are mutually exclusive")

    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        raise XaiResponsesError("XAI_API_KEY is not set")

    if reasoning_effort is not None and reasoning_effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {sorted(VALID_REASONING_EFFORTS)}"
        )

    resolved_model = model or _resolve("model", DEFAULT_MODEL)
    resolved_timeout = int(timeout_seconds or _resolve("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    if resolved_timeout <= 0:
        raise ValueError("timeout_seconds must be positive")

    body: Dict[str, Any] = {
        "model": resolved_model,
        "input": input if input is not None else prompt,
    }
    if instructions is not None:
        body["instructions"] = instructions
    if store is not None:
        body["store"] = store
    if previous_response_id is not None:
        body["previous_response_id"] = previous_response_id
    if max_turns is not None:
        body["max_turns"] = max_turns
    if parallel_tool_calls is not None:
        body["parallel_tool_calls"] = parallel_tool_calls
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if user is not None:
        body["user"] = user
    if extra_body:
        for k, v in extra_body.items():
            # input/model are caller-defined here — don't let extra_body clobber them.
            if k in ("model", "input"):
                continue
            body[k] = v

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }
    base_url = DEFAULT_BASE_URL.rstrip("/")
    url = f"{base_url}/responses"

    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=resolved_timeout)
    except httpx.HTTPError as exc:
        raise XaiResponsesError(f"responses POST failed: {exc}") from exc

    if resp.status_code >= 400:
        raise XaiResponsesError(
            f"responses POST returned {resp.status_code}: {resp.text[:300]}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise XaiResponsesError(
            f"responses POST returned non-JSON body: {resp.text[:300]}"
        ) from exc

    if not isinstance(payload, dict):
        raise XaiResponsesError(f"responses POST returned non-object body: {payload!r}")

    return {
        "response_id": payload.get("id"),
        "status": payload.get("status"),
        "output_text": _extract_text(payload),
        "raw": payload,
    }


def _extract_text(payload: Dict[str, Any]) -> str:
    """Best-effort plain-text extraction from a Responses API body.

    Walks ``output[*].content[*]`` (the canonical shape) and concatenates
    every ``text``-style item it finds. Returns "" if nothing readable.
    """
    out = payload.get("output")
    if not isinstance(out, list):
        # Some xAI variants surface a top-level convenience field.
        legacy = payload.get("output_text")
        return str(legacy) if isinstance(legacy, str) else ""

    chunks: List[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        # New-style: item with content array
        content = item.get("content")
        if isinstance(content, list):
            for piece in content:
                if not isinstance(piece, dict):
                    continue
                text = piece.get("text") or piece.get("value")
                if isinstance(text, str) and text:
                    chunks.append(text)
            continue
        # Some shapes carry text directly on the item
        text = item.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def check_xai_responses_requirements() -> bool:
    return bool(os.environ.get("XAI_API_KEY", "").strip())


XAI_RESPONSES_SCHEMA = {
    "name": "xai_responses_chat",
    "description": (
        "One-shot call to xAI's POST /v1/responses with full access to the "
        "Responses API: store=true (server-side history, 30-day retention), "
        "previous_response_id (continue a stored conversation without "
        "re-uploading context), max_turns (cap agentic tool-calling loops), "
        "parallel_tool_calls. Use for long autonomous workflows where "
        "re-uploading context every turn is wasteful, or when chaining "
        "agentic legs that share state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Single-message text input (alternative to 'input').",
            },
            "input": {
                "description": "Raw Responses API input (string or list of input items).",
            },
            "instructions": {
                "type": "string",
                "description": "System prompt — xAI's Responses API alternative to a system message.",
            },
            "model": {
                "type": "string",
                "description": "xAI model. Defaults to config xai_responses.model or 'grok-4.3'.",
            },
            "store": {
                "type": "boolean",
                "description": "Persist input + response server-side for 30 days (enables previous_response_id chaining).",
            },
            "previous_response_id": {
                "type": "string",
                "description": "Continue a stored conversation. Requires the prior call to have used store=true.",
            },
            "max_turns": {
                "type": "integer",
                "description": "Cap on the agentic tool-calling loop. Ignored for non-agentic calls.",
            },
            "parallel_tool_calls": {
                "type": "boolean",
                "description": "Allow the model to dispatch multiple tool calls in parallel.",
            },
            "reasoning_effort": {
                "type": "string",
                "enum": ["none", "low", "medium", "high"],
                "description": "Reasoning intensity. Translated to reasoning.effort.",
            },
            "max_output_tokens": {"type": "integer"},
            "temperature": {"type": "number"},
            "top_p": {"type": "number"},
            "tools": {"type": "array", "items": {"type": "object"}},
            "tool_choice": {},
            "user": {"type": "string"},
            "timeout_seconds": {"type": "integer", "description": "HTTP timeout for the call. Defaults to 300s."},
        },
    },
}


def _handle_xai_responses_tool_call(args: Dict[str, Any], **_kw: Any) -> Dict[str, Any]:
    return xai_responses_chat(
        prompt=args.get("prompt"),
        input=args.get("input"),
        instructions=args.get("instructions"),
        model=args.get("model"),
        store=args.get("store"),
        previous_response_id=args.get("previous_response_id"),
        max_turns=args.get("max_turns"),
        parallel_tool_calls=args.get("parallel_tool_calls"),
        reasoning_effort=args.get("reasoning_effort"),
        max_output_tokens=args.get("max_output_tokens"),
        temperature=args.get("temperature"),
        top_p=args.get("top_p"),
        tools=args.get("tools"),
        tool_choice=args.get("tool_choice"),
        user=args.get("user"),
        timeout_seconds=args.get("timeout_seconds"),
        extra_body=args.get("extra_body"),
    )


# Self-register at import time.
from tools.registry import registry  # noqa: E402

registry.register(
    name="xai_responses_chat",
    toolset="xai_responses",
    schema=XAI_RESPONSES_SCHEMA,
    handler=_handle_xai_responses_tool_call,
    check_fn=check_xai_responses_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🧠",
)
