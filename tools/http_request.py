#!/usr/bin/env python3
"""HTTP Request core tool for performing web requests safely.

SSRF protection is enforced via `tools.url_safety.is_safe_url` and
event-hook redirects are validated to prevent bypasses.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from tools.registry import registry, tool_result, tool_error
from tools.url_safety import is_safe_url, normalize_url_for_request

logger = logging.getLogger(__name__)


def _get_env_value(name: str) -> str:
    """Resolve environment variable from Hermes config or process env."""
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value(name)
    except Exception:
        val = None
    if val is None:
        val = os.getenv(name, "")
    return (val or "").strip()


def redact_url_secrets(url: str) -> str:
    """Redact keys containing key/token/secret/auth from query params."""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        qsl = parse_qsl(parsed.query, keep_blank_values=True)
        redacted_qsl = []
        for k, v in qsl:
            k_lower = k.lower()
            if any(secret_word in k_lower for secret_word in ("key", "token", "secret", "auth")):
                redacted_qsl.append((k, "[REDACTED]"))
            else:
                redacted_qsl.append((k, v))
        new_query = urlencode(redacted_qsl)
        return urlunparse(parsed._replace(query=new_query))
    except Exception:
        return url


def redact_sensitive_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Redact sensitive header values to prevent credential leaks."""
    if not headers:
        return {}
    redacted = {}
    for k, v in headers.items():
        k_lower = k.lower()
        if any(secret_word in k_lower for secret_word in ("auth", "key", "token", "secret", "signature", "cookie", "cert")):
            redacted[k] = "[REDACTED]"
        else:
            redacted[k] = v
    return redacted


def http_request_tool(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    query: Optional[Dict[str, Any]] = None,
    json_body: Optional[Any] = None,
    form_body: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = 30.0,
    expected_content_type: Optional[str] = None,
    auth_mode: str = "none",
    auth_token_env: Optional[str] = None,
) -> str:
    """Perform a secure HTTP request with redirection protection and output sanitization."""
    # Redacted URL placeholder in case we error before full URL parsing
    redacted_url = redact_url_secrets(url)
    
    try:
        # 1. Method Validation
        method = str(method).strip().upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"Unsupported HTTP method: {method}")

        # 2. URL Normalization & Prep
        url_to_check = url.strip()
        url_lower = url_to_check.lower()
        if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
            if not re.match(r"^[a-z]+://", url_lower):
                url_to_check = "https://" + url_to_check

        normalized_url = normalize_url_for_request(url_to_check)
        redacted_url = redact_url_secrets(normalized_url)

        # 3. Safety/SSRF Checks
        if not is_safe_url(normalized_url):
            raise PermissionError(f"URL is not safe to access: {redacted_url}")

        # 4. Auth & Header Prep
        req_headers = dict(headers) if headers else {}
        auth_mode = str(auth_mode).strip().lower()
        if auth_mode not in {"none", "bearer_env"}:
            raise ValueError(f"Unsupported auth_mode: {auth_mode}")

        if auth_mode == "bearer_env":
            if not auth_token_env:
                raise ValueError("auth_token_env must be provided when auth_mode is bearer_env")
            token = _get_env_value(auth_token_env)
            if not token:
                raise ValueError(f"Environment variable '{auth_token_env}' is empty or not set")
            req_headers["Authorization"] = f"Bearer {token}"

        # 5. Body Validation
        if json_body is not None and form_body is not None:
            raise ValueError("Cannot specify both json_body and form_body")

        # 6. Timeout Setup
        if timeout_seconds is not None:
            try:
                timeout = float(timeout_seconds)
                if timeout <= 0:
                    raise ValueError("Timeout must be a positive number")
            except (ValueError, TypeError):
                raise ValueError(f"Invalid timeout value: {timeout_seconds}")
        else:
            timeout = 30.0

        # 7. Redirect safety guard
        def _ssrf_redirect_guard(response):
            if response.is_redirect and response.next_request:
                redirect_url = str(response.next_request.url)
                if not is_safe_url(redirect_url):
                    raise ValueError(
                        f"Blocked redirect to private/internal address: {redact_url_secrets(redirect_url)}"
                    )

        # 8. Request Execution
        with httpx.Client(
            follow_redirects=True,
            event_hooks={"response": [_ssrf_redirect_guard]},
        ) as client:
            response = client.request(
                method=method,
                url=normalized_url,
                headers=req_headers,
                params=query,
                json=json_body,
                data=form_body,
                timeout=timeout,
            )

        # 9. Expected Content Type validation
        content_type = response.headers.get("content-type", "")
        if expected_content_type:
            ect = expected_content_type.strip().lower()
            if ect not in content_type.lower():
                raise ValueError(
                    f"Expected content type '{expected_content_type}', but got '{content_type}'"
                )

        # 10. Truncation and JSON Parsing
        max_len = registry.get_max_result_size("http_request", 100_000)
        
        truncated = False
        text_preview = response.text
        if len(text_preview) > max_len:
            text_preview = text_preview[:max_len] + "\n... [TRUNCATED] ..."
            truncated = True

        json_data = None
        if not truncated and "application/json" in content_type.lower():
            try:
                json_data = response.json()
            except Exception:
                pass

        return tool_result(
            success=True,
            tool="http_request",
            method=method,
            url=redacted_url,
            status=response.status_code,
            ok=response.is_success,
            content_type=content_type,
            headers=redact_sensitive_headers(dict(response.headers)),
            json=json_data,
            text_preview=text_preview,
            truncated=truncated,
        )

    except Exception as e:
        logger.exception("http_request tool execution failed")
        return tool_result(
            success=False,
            tool="http_request",
            method=method if 'method' in locals() else str(method),
            url=redacted_url,
            error=str(e),
            error_type=type(e).__name__,
        )


HTTP_REQUEST_SCHEMA = {
    "name": "http_request",
    "description": (
        "Perform an HTTP request (GET, POST, PUT, PATCH, DELETE) to a public URL. "
        "Supports custom headers, query parameters, JSON/form bodies, timeout configuration, "
        "and bearer token authentication via environment variables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "The HTTP method to use (GET, POST, PUT, PATCH, DELETE).",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            },
            "url": {
                "type": "string",
                "description": "The target HTTP or HTTPS URL.",
            },
            "headers": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                },
                "description": "Optional HTTP headers to include in the request.",
            },
            "query": {
                "type": "object",
                "description": "Optional query parameters to append to the URL.",
            },
            "json_body": {
                "type": "object",
                "description": "Optional JSON body to send with the request.",
            },
            "form_body": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                },
                "description": "Optional form body (URL-encoded) to send with the request.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Optional timeout for the request in seconds. Default is 30.",
                "default": 30.0,
            },
            "expected_content_type": {
                "type": "string",
                "description": "Optional expected content-type pattern to check in response headers (e.g., 'application/json').",
            },
            "auth_mode": {
                "type": "string",
                "description": "Authentication mode. Use 'none' or 'bearer_env'. Default is 'none'.",
                "enum": ["none", "bearer_env"],
                "default": "none",
            },
            "auth_token_env": {
                "type": "string",
                "description": "The environment variable name containing the bearer token (only used if auth_mode is 'bearer_env').",
            },
        },
        "required": ["method", "url"],
    },
}


def _handle_http_request(args: dict, **kwargs) -> str:
    return http_request_tool(
        method=args.get("method", "GET"),
        url=args.get("url", ""),
        headers=args.get("headers"),
        query=args.get("query"),
        json_body=args.get("json_body"),
        form_body=args.get("form_body"),
        timeout_seconds=args.get("timeout_seconds"),
        expected_content_type=args.get("expected_content_type"),
        auth_mode=args.get("auth_mode", "none"),
        auth_token_env=args.get("auth_token_env"),
    )


registry.register(
    name="http_request",
    toolset="api",
    schema=HTTP_REQUEST_SCHEMA,
    handler=_handle_http_request,
    emoji="🌐",
    max_result_size_chars=100_000,
)
