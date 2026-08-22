#!/usr/bin/env python3
"""Native Ollama ``/api/generate`` transport for the Vision Orchestrator
(inactive adapter).

This module adds exactly one alternative *transport* for the local vision
pipeline. It does NOT change any default: the Vision Orchestrator still uses
the existing OpenAI-compatible auxiliary-client path
(``tools.ollama_vision_client.invoke_vision_model``) unless an explicit
private invocation selects ``TRANSPORT_OLLAMA_NATIVE_GENERATE``.

Native transport contract (validated in qwen36-route-format-exp-v0_1):

- endpoint ``/api/generate`` with ``stream=false``;
- ``images`` carries RAW base64 bytes with no ``data:image/...;base64,``
  prefix;
- ``prompt`` is the exact validated generate Prompt rendering;
- ``options`` carries ``num_ctx`` / ``num_predict`` / ``temperature`` /
  ``seed`` explicitly — no reliance on Ollama defaults;
- ``format`` is either the literal ``"json"`` or the validated strict JSON
  Schema object;
- qwen3.6 is a thinking model: when ``response`` is empty the usable
  structured answer lives in ``thinking`` — extraction is deterministic and
  explicit (``content_source`` / ``thinking_fallback_used``).

Privacy boundary: the provider-result contract exposes only normalized safe
metadata (timing, counts, character lengths, content_source); raw response
text, full thinking, base64 and private endpoint identity never enter the
result contract.

No live inference is performed by import — tests are fully mocked.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from tools.vision_policy import (
    ExecutionStatus,
    TRANSPORT_OPENAI_COMPATIBLE,
    TRANSPORT_OLLAMA_NATIVE_GENERATE,
)

# ---------------------------------------------------------------------------
# Transport identities (defined in tools.vision_policy — single source of
# truth; re-exported here for backward compatibility with earlier imports).
# ---------------------------------------------------------------------------

# Validated strict JSON Schema (experiment PROFILE_S, aligned to the
# canonical internal field per task HERMES_VISION_UI_READ_TEXT_FIELD_
# CONTRACT_ALIGNMENT_V0_1). The REQUIRED field is the canonical
# ``observed_text`` (Prompt contract + quality gates + calibration all use
# it); ``visible_text`` remains only as a legacy alias accepted through the
# explicit resolver in tools/vision_policy. Byte-identical semantics to the
# harness ``schema_obj()`` except the required field name.
NATIVE_GENERATE_STRICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible_text": {"type": "array", "items": {"type": "string"}},
        "observed_text": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
        "uncertainty": {"type": "string"},
        "inference": {"type": "string"},
    },
    "required": ["observed_text"],
}

# Default candidate runtime profile (task §9) — supported, never activated
# as a permanent default. Mirrors the validated canonical request
# (temperature 0.1, num_predict 4000, seed 42, JSON Schema output).
NATIVE_GENERATE_DEFAULT_PROFILE: Dict[str, Any] = {
    "num_ctx": 32768,
    "num_predict": 4000,
    "temperature": 0.1,
    "seed": 42,
    "format": NATIVE_GENERATE_STRICT_SCHEMA,
}

# Task-specific strict JSON Schemas for the native generate transport.
# UI_READ uses the canonical NATIVE_GENERATE_STRICT_SCHEMA (required
# observed_text); SCENE_DESCRIBE / EVIDENCE_VERIFY carry their own
# canonical task fields (task HERMES_VISION_PRECISION_NATIVE_GENERATE_
# PROFILE_BINDING_V0_1 — profile output contract is task-specific).
TASK_STRICT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "UI_READ": NATIVE_GENERATE_STRICT_SCHEMA,
    "SCENE_DESCRIBE": {
        "type": "object",
        "properties": {
            "observation": {"type": "string"},
            "inference": {"type": "string"},
            "uncertainty": {"type": "string"},
        },
        "required": ["observation", "inference"],
    },
    "EVIDENCE_VERIFY": {
        "type": "object",
        "properties": {
            "evidence": {"type": "string"},
            "observation": {"type": "string"},
            "inference": {"type": "string"},
            "uncertainty": {"type": "string"},
            "contradiction": {"type": "string"},
        },
        "required": ["evidence"],
    },
}

_TIMING_NS_FIELDS = (
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
    "total_duration",
)
_COUNT_FIELDS = ("prompt_eval_count", "eval_count")
_META_FIELDS = ("model", "created_at", "done", "done_reason", "context")


# ---------------------------------------------------------------------------
# HTTP transport (single seam for offline mocks).
# ---------------------------------------------------------------------------


async def _http_post_json(
    base_url: str,
    path: str,
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> Tuple[int, Any]:
    """POST ``payload`` as JSON to ``base_url + path``.

    Returns ``(status_code, body)`` where ``body`` is the parsed JSON value
    (typically a dict) or ``None`` when the response had no body or was not
    JSON. Raises ``httpx`` transport exceptions (timeout / connect) which the
    caller classifies — this function performs no classification.
    """
    import httpx

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(f"{base_url.rstrip('/')}{path}", json=payload)
    if not resp.content:
        return resp.status_code, None
    try:
        return resp.status_code, resp.json()
    except Exception:  # noqa: BLE001 — non-JSON body handled by caller
        return resp.status_code, None


# ---------------------------------------------------------------------------
# Deterministic response/thinking extraction (task §11).
# ---------------------------------------------------------------------------


def _extract_final_content(
    envelope: Dict[str, Any],
) -> Tuple[Optional[str], str, bool, Optional[str]]:
    """Extract the usable final content from a native generate envelope.

    Returns ``(content, content_source, thinking_fallback_used, error_code)``:

    - ``content`` is non-None when usable content exists;
    - ``content_source`` is ``"response"`` or ``"thinking_fallback"``;
    - ``error_code`` is set (and ``content`` is None) for the invalid
      classes: ``non_string_response``, ``non_string_thinking``,
      ``missing_response_and_thinking``.

    Rules (never merge, never silently prefer one):
    - non-empty ``response`` is primary;
    - empty ``response`` + non-empty ``thinking`` → explicit thinking
      fallback (qwen3.6 thinking model);
    - both non-empty → ``response`` wins (thinking preserved only as
      non-sensitive metadata);
    - non-string ``response`` / ``thinking`` → rejected safely;
    - both empty → INVALID_RESPONSE class.
    """
    if "response" in envelope and not isinstance(envelope.get("response"), str):
        return None, "", False, "non_string_response"
    if "thinking" in envelope and not isinstance(envelope.get("thinking"), str):
        return None, "", False, "non_string_thinking"

    response = envelope.get("response") or ""
    thinking = envelope.get("thinking") or ""
    if response.strip():
        return response, "response", False, None
    if thinking.strip():
        return thinking, "thinking_fallback", True, None
    return None, "", False, "missing_response_and_thinking"


# ---------------------------------------------------------------------------
# Safe metadata normalization (task §12).
# ---------------------------------------------------------------------------


def _normalize_timing(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Ollama nanosecond durations to integer milliseconds.

    Non-numeric values become ``None``. Count fields pass through as-is.
    """
    out: Dict[str, Any] = {}
    for field in _TIMING_NS_FIELDS:
        value = envelope.get(field)
        key = field.replace("_duration", "_ms")
        out[key] = int(value / 1_000_000) if isinstance(value, (int, float)) else None
    for field in _COUNT_FIELDS:
        out[field] = envelope.get(field)
    return out


def _safe_envelope_meta(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Non-sensitive envelope metadata only (no response/thinking/base64)."""
    return {key: envelope.get(key) for key in _META_FIELDS if key in envelope}


def _classify_transport_error(
    exc: Exception,
    *,
    status: Optional[int] = None,
    body_error: Optional[str] = None,
) -> str:
    """Map transport failures to an ``ExecutionStatus`` value."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg:
        return ExecutionStatus.TIMEOUT.value
    if status is not None:
        if status == 404 or (body_error and "not found" in body_error.lower()):
            return ExecutionStatus.MODEL_NOT_FOUND.value
        if 500 <= status <= 599:
            return ExecutionStatus.ENDPOINT_UNAVAILABLE.value
        if 400 <= status <= 499:
            return ExecutionStatus.INVALID_RESPONSE.value
    if "connect" in name or "connection" in msg or "refused" in msg:
        return ExecutionStatus.ENDPOINT_UNAVAILABLE.value
    return ExecutionStatus.ENDPOINT_UNAVAILABLE.value


def _is_transient(exc: Exception) -> bool:
    """True for transport-level blips worth a bounded retry (when enabled)."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg:
        return True
    if "connect" in name or "refused" in msg:
        return False  # endpoint unreachable — retrying is pointless
    return True


# ---------------------------------------------------------------------------
# Native generate invocation (task §8).
# ---------------------------------------------------------------------------


async def invoke_native_generate(
    *,
    model: str,
    prompt: str,
    image_raw_base64: str,
    num_ctx: int,
    num_predict: int,
    temperature: float,
    seed: Optional[int],
    format_spec: Any,
    base_url: str,
    timeout_seconds: float,
    transport_retries: int = 0,
) -> Dict[str, Any]:
    """Invoke exactly one native Ollama ``/api/generate`` call (bounded
    transport retries only when ``transport_retries > 0``).

    Returns the provider-result contract (safe metadata only):

    ``execution_status``, ``extracted_content``, ``content_source``,
    ``thinking_fallback_used``, ``response_character_count``,
    ``thinking_character_count``, ``response_envelope`` (safe meta),
    ``model``, ``done``, ``done_reason``, ``created_at``,
    ``total_ms``/``load_ms``/``prompt_eval_ms``/``eval_ms``,
    ``prompt_eval_count``/``eval_count``, ``transport_attempts``, ``error``.

    Never falls back to another transport, never calls a second model, never
    escalates. Raw response text never enters the returned contract.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "images": [image_raw_base64],
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    if format_spec is not None:
        payload["format"] = format_spec

    attempts = max(1, int(transport_retries or 0) + 1)
    last_exc: Optional[Exception] = None
    last_status: Optional[int] = None
    last_body: Any = None

    for attempt in range(attempts):
        last_exc = None
        try:
            last_status, last_body = await _http_post_json(
                base_url, "/api/generate", payload, timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 — classified below
            last_exc = exc
            if _is_transient(exc) and attempt + 1 < attempts:
                await asyncio.sleep(0.5)
                continue
            status = _classify_transport_error(exc)
            return {
                "execution_status": status,
                "extracted_content": "",
                "content_source": "",
                "thinking_fallback_used": False,
                "response_character_count": 0,
                "thinking_character_count": 0,
                "response_envelope": {},
                "model": model,
                "done": None,
                "done_reason": None,
                "created_at": None,
                "total_ms": None,
                "load_ms": None,
                "prompt_eval_ms": None,
                "eval_ms": None,
                "prompt_eval_count": None,
                "eval_count": None,
                "transport_attempts": attempt + 1,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }

        if not isinstance(last_status, int) or last_status < 200 or last_status >= 300:
            body_error = ""
            if isinstance(last_body, dict) and isinstance(last_body.get("error"), str):
                body_error = last_body["error"]
            if last_status is not None and 500 <= last_status <= 599 and attempt + 1 < attempts:
                await asyncio.sleep(0.5)
                continue
            status = _classify_transport_error(
                Exception(body_error or f"HTTP {last_status}"), status=last_status
            )
            return {
                "execution_status": status,
                "extracted_content": "",
                "content_source": "",
                "thinking_fallback_used": False,
                "response_character_count": 0,
                "thinking_character_count": 0,
                "response_envelope": {},
                "model": model,
                "done": None,
                "done_reason": None,
                "created_at": None,
                "total_ms": None,
                "load_ms": None,
                "prompt_eval_ms": None,
                "eval_ms": None,
                "prompt_eval_count": None,
                "eval_count": None,
                "transport_attempts": attempt + 1,
                "error": body_error or f"http_{last_status}",
            }
        break

    # 2xx with a non-dict body → malformed JSON envelope.
    if not isinstance(last_body, dict):
        return {
            "execution_status": ExecutionStatus.INVALID_RESPONSE.value,
            "extracted_content": "",
            "content_source": "",
            "thinking_fallback_used": False,
            "response_character_count": 0,
            "thinking_character_count": 0,
            "response_envelope": {},
            "model": model,
            "done": None,
            "done_reason": None,
            "created_at": None,
            "total_ms": None,
            "load_ms": None,
            "prompt_eval_ms": None,
            "eval_ms": None,
            "prompt_eval_count": None,
            "eval_count": None,
            "transport_attempts": attempts,
            "error": "malformed_json_envelope",
        }

    content, source, fallback_used, extract_error = _extract_final_content(last_body)
    if extract_error is not None:
        return {
            "execution_status": ExecutionStatus.INVALID_RESPONSE.value,
            "extracted_content": "",
            "content_source": "",
            "thinking_fallback_used": False,
            "response_character_count": 0,
            "thinking_character_count": 0,
            "response_envelope": _safe_envelope_meta(last_body),
            "model": last_body.get("model") or model,
            "done": last_body.get("done"),
            "done_reason": last_body.get("done_reason"),
            "created_at": last_body.get("created_at"),
            "total_ms": _normalize_timing(last_body).get("total_ms"),
            "load_ms": _normalize_timing(last_body).get("load_ms"),
            "prompt_eval_ms": _normalize_timing(last_body).get("prompt_eval_ms"),
            "eval_ms": _normalize_timing(last_body).get("eval_ms"),
            "prompt_eval_count": _normalize_timing(last_body).get("prompt_eval_count"),
            "eval_count": _normalize_timing(last_body).get("eval_count"),
            "transport_attempts": attempts,
            "error": extract_error,
        }

    timing = _normalize_timing(last_body)
    meta = _safe_envelope_meta(last_body)
    return {
        "execution_status": ExecutionStatus.SUCCESS.value,
        "extracted_content": content,
        "content_source": source,
        "thinking_fallback_used": fallback_used,
        "response_character_count": len(str(last_body.get("response") or "")),
        "thinking_character_count": len(str(last_body.get("thinking") or "")),
        "response_envelope": meta,
        "model": meta.get("model") or model,
        "done": meta.get("done"),
        "done_reason": meta.get("done_reason"),
        "created_at": meta.get("created_at"),
        "total_ms": timing.get("total_ms"),
        "load_ms": timing.get("load_ms"),
        "prompt_eval_ms": timing.get("prompt_eval_ms"),
        "eval_ms": timing.get("eval_ms"),
        "prompt_eval_count": timing.get("prompt_eval_count"),
        "eval_count": timing.get("eval_count"),
        "transport_attempts": attempts,
        "error": None,
    }
