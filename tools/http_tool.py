#!/usr/bin/env python3
"""HTTP tool — structured HTTP requests without going through a shell.

The `terminal` tool is general-purpose but routes every command through
`bash -c`, which forces the model to escape JSON bodies into shell-safe
strings. A single apostrophe in a JSON value (`"It's done"`) breaks the
outer single-quoting and the request fails with `unexpected EOF while
looking for matching '` — costing a wasted model turn.

This tool takes structured args (`{method, url, headers, json, body, ...}`)
and dispatches via httpx directly. No shell, no quoting, no per-call bash
fork+exec overhead (~30-80 ms saved per call).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from agent.redact import redact_sensitive_text
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults & caps
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_TIMEOUT_SECONDS = 300.0
_MAX_RESPONSE_BYTES = 1_000_000  # 1 MB; matches registry max_result_size_chars

# Headers that should never be echoed back to the model (request-side leakage
# protection — the response body is separately run through redact_sensitive_text).
_SENSITIVE_HEADER_NAMES = frozenset({
    "authorization",
    "cookie",
    "set-cookie",
    "proxy-authorization",
})

_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})

# ---------------------------------------------------------------------------
# Module-level httpx client (connection pool reused across calls).
# Module-level so unit tests can monkeypatch tools.http_tool._http_client.
# ---------------------------------------------------------------------------
_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    """Return the singleton httpx.Client, constructing it on first use."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT_SECONDS, connect=10.0),
            follow_redirects=False,
        )
    return _http_client


def _redact_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop sensitive header names from the echoed response."""
    return {
        k: ("[REDACTED]" if k.lower() in _SENSITIVE_HEADER_NAMES or k.lower().endswith("-token") else v)
        for k, v in headers.items()
    }


def http_tool(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    body: str | None = None,
    params: dict[str, str] | None = None,
    timeout: int | float | None = None,
    task_id: str = "default",  # noqa: ARG001 — reserved for future per-task scoping
) -> str:
    """Issue a single HTTP request and return a JSON-encoded result string."""
    # ── Validate method ──────────────────────────────────────────────
    if not isinstance(method, str):
        return tool_error("method must be a string (one of GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS)")
    method_upper = method.upper().strip()
    if method_upper not in _ALLOWED_METHODS:
        return tool_error(
            f"method '{method}' not allowed. Use one of: {sorted(_ALLOWED_METHODS)}"
        )

    # ── Validate url ─────────────────────────────────────────────────
    if not isinstance(url, str) or not url:
        return tool_error("url must be a non-empty string")

    # ── Mutually exclusive body fields ───────────────────────────────
    if json_body is not None and body is not None:
        return tool_error(
            "Provide either 'json' (auto-serialised object) or 'body' (raw string), not both."
        )

    # ── Resolve timeout ──────────────────────────────────────────────
    eff_timeout: float = _DEFAULT_TIMEOUT_SECONDS
    if timeout is not None:
        try:
            eff_timeout = float(timeout)
        except (TypeError, ValueError):
            return tool_error(f"timeout must be a number, got {type(timeout).__name__}")
        if eff_timeout <= 0:
            return tool_error("timeout must be positive")
        if eff_timeout > _MAX_TIMEOUT_SECONDS:
            return tool_error(
                f"timeout {eff_timeout}s exceeds maximum {_MAX_TIMEOUT_SECONDS}s"
            )

    # ── Build request kwargs ─────────────────────────────────────────
    request_kwargs: dict[str, Any] = {
        "method": method_upper,
        "url": url,
        "timeout": eff_timeout,
    }
    if headers:
        if not isinstance(headers, dict):
            return tool_error("headers must be an object mapping header name to value")
        request_kwargs["headers"] = headers
    if params:
        if not isinstance(params, dict):
            return tool_error("params must be an object mapping query-param name to value")
        request_kwargs["params"] = params
    if json_body is not None:
        # httpx auto-serialises and sets Content-Type: application/json
        request_kwargs["json"] = json_body
    elif body is not None:
        if not isinstance(body, str):
            return tool_error("body must be a string; use 'json' for objects")
        request_kwargs["content"] = body

    # ── Dispatch ─────────────────────────────────────────────────────
    client = _get_http_client()
    started = time.monotonic()
    try:
        response = client.request(**request_kwargs)
    except httpx.TimeoutException as e:
        return tool_error(
            f"Request timed out after {eff_timeout}s: {e}",
            timeout=True,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    except httpx.RequestError as e:
        return tool_error(
            f"Transport error: {type(e).__name__}: {e}",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)

    # ── Body cap + decode ────────────────────────────────────────────
    raw_body = response.content or b""
    truncated = False
    if len(raw_body) > _MAX_RESPONSE_BYTES:
        raw_body = raw_body[:_MAX_RESPONSE_BYTES]
        truncated = True
    try:
        decoded_body = raw_body.decode(response.encoding or "utf-8", errors="replace")
    except (LookupError, TypeError):
        decoded_body = raw_body.decode("utf-8", errors="replace")

    # Redact secrets that may appear in echoed bodies (api keys, tokens, etc.).
    redacted_body = redact_sensitive_text(decoded_body, code_file=False)

    result: dict[str, Any] = {
        "status": response.status_code,
        "headers": _redact_response_headers(dict(response.headers)),
        "body": redacted_body,
        "elapsed_ms": elapsed_ms,
    }
    if truncated:
        result["truncated"] = True
        result["_hint"] = (
            f"Response body exceeded {_MAX_RESPONSE_BYTES:,} bytes and was truncated. "
            "If you need more, request a smaller slice (e.g. via Range header or pagination)."
        )
    return tool_result(result)


# ---------------------------------------------------------------------------
# Schema + Registration
# ---------------------------------------------------------------------------

HTTP_TOOL_DESCRIPTION = (
    "Issue a structured HTTP request without going through a shell. "
    "Prefer this over `terminal` curl for ALL API calls — it eliminates "
    "the bash-quoting bug class (e.g. apostrophes in JSON bodies) and "
    "saves ~30-80 ms per call. "
    "Body: pass `json` (object → auto-serialised + Content-Type set) OR "
    "`body` (raw string), not both. "
    "GET/HEAD requests are safe to retry; POST/PUT/PATCH/DELETE may have "
    "side effects — do not retry blindly. "
    "Returns: {status, headers, body, elapsed_ms} on success; "
    "{error, ...} on transport failure (timeout, DNS, connection refused). "
    "Note: body cap is 1 MB; egress is unrestricted (parity with terminal)."
)

HTTP_SCHEMA = {
    "name": "http",
    "description": HTTP_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, or OPTIONS.",
            },
            "url": {
                "type": "string",
                "description": "Absolute URL (http:// or https://).",
            },
            "headers": {
                "type": "object",
                "description": "Request headers as a flat name→value map. Authorization / Cookie / Set-Cookie are redacted from the echoed response.",
                "additionalProperties": {"type": "string"},
            },
            "json": {
                "type": "object",
                "description": "Request body as a JSON object — httpx serialises it and sets Content-Type: application/json. Use this for almost every API write; mutually exclusive with `body`.",
                "additionalProperties": True,
            },
            "body": {
                "type": "string",
                "description": "Raw request body as a string. Use only when sending non-JSON (e.g. form-encoded, plain text). Mutually exclusive with `json`.",
            },
            "params": {
                "type": "object",
                "description": "Query-string parameters as a flat name→value map. Appended to the URL.",
                "additionalProperties": {"type": "string"},
            },
            "timeout": {
                "type": "integer",
                "description": f"Request timeout in seconds (default {int(_DEFAULT_TIMEOUT_SECONDS)}, max {int(_MAX_TIMEOUT_SECONDS)}).",
                "minimum": 1,
                "maximum": int(_MAX_TIMEOUT_SECONDS),
            },
        },
        "required": ["method", "url"],
    },
}


def _handle_http(args, **kw):
    return http_tool(
        method=args.get("method"),
        url=args.get("url"),
        headers=args.get("headers"),
        json_body=args.get("json"),
        body=args.get("body"),
        params=args.get("params"),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id", "default"),
    )


registry.register(
    name="http",
    toolset="http",
    schema=HTTP_SCHEMA,
    handler=_handle_http,
    emoji="🌐",
    max_result_size_chars=_MAX_RESPONSE_BYTES,
)
