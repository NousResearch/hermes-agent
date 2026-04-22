#!/usr/bin/env python3
"""
HTTP Request Tool

Make HTTP requests to any URL. Supports GET, POST, PUT, PATCH, DELETE
with custom headers, JSON body, form data, and query parameters.

Useful for:
- API testing and integration
- Webhook testing
- REST API consumption
- File downloads
- Health checks
"""

import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx

from tools.registry import registry

logger = logging.getLogger(__name__)

# Max response body size to return (1MB)
MAX_RESPONSE_SIZE = 1_048_576

# Allowed methods
_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


def _handle_http_request(args: Dict[str, Any], **kw) -> str:
    """Handle http_request tool calls."""
    url = args.get("url", "")
    if not url:
        return "Error: 'url' is required"

    method = args.get("method", "GET").upper()
    if method not in _ALLOWED_METHODS:
        return f"Error: method must be one of {sorted(_ALLOWED_METHODS)}"

    headers = args.get("headers") or {}
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except json.JSONDecodeError:
            return "Error: 'headers' must be a valid JSON object"

    params = args.get("params") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return "Error: 'params' must be a valid JSON object"

    body = args.get("body")
    json_body = args.get("json")
    form_data = args.get("form_data")

    timeout = min(args.get("timeout", 30), 120)  # Max 120s
    follow_redirects = args.get("follow_redirects", True)

    # Security: block internal networks (SSRF protection)
    from tools.url_safety import is_safe_url
    if not is_safe_url(url):
        return "Error: URL targets a private/internal network address (SSRF protection)"

    try:
        start = time.time()

        with httpx.Client(
            timeout=timeout,
            follow_redirects=follow_redirects,
            verify=True,
        ) as client:
            request_kwargs: Dict[str, Any] = {
                "method": method,
                "url": url,
            }

            if headers:
                request_kwargs["headers"] = headers
            if params:
                request_kwargs["params"] = params
            if json_body is not None:
                request_kwargs["json"] = json_body
            elif body is not None:
                if isinstance(body, dict):
                    request_kwargs["content"] = json.dumps(body)
                    if "content-type" not in {k.lower() for k in headers}:
                        headers["Content-Type"] = "application/json"
                        request_kwargs["headers"] = headers
                else:
                    request_kwargs["content"] = str(body)
            elif form_data is not None:
                if isinstance(form_data, str):
                    try:
                        form_data = json.loads(form_data)
                    except json.JSONDecodeError:
                        return "Error: 'form_data' must be a valid JSON object"
                request_kwargs["data"] = form_data

            response = client.request(**request_kwargs)

        elapsed = time.time() - start

        # Build response
        result = {
            "status_code": response.status_code,
            "status_text": response.reason_phrase,
            "elapsed_seconds": round(elapsed, 3),
            "url": str(response.url),
            "headers": dict(response.headers),
        }

        # Try to parse response body
        content_type = response.headers.get("content-type", "")
        body_text = response.text

        if len(body_text) > MAX_RESPONSE_SIZE:
            result["body"] = body_text[:MAX_RESPONSE_SIZE]
            result["body_truncated"] = True
            result["body_original_size"] = len(body_text)
        else:
            result["body"] = body_text

        # If JSON, also include parsed version
        if "application/json" in content_type:
            try:
                result["json"] = response.json()
            except Exception:
                pass

        # Format output
        status_icon = "✓" if response.is_success else "✗"
        output_lines = [
            f"{status_icon} HTTP {response.status_code} {response.reason_phrase} ({elapsed:.2f}s)",
            f"URL: {response.url}",
            "",
        ]

        # Show response headers (compact)
        output_lines.append("Response Headers:")
        for k, v in list(response.headers.items())[:20]:
            output_lines.append(f"  {k}: {v}")

        output_lines.append("")

        # Show body
        if "application/json" in content_type:
            try:
                parsed = response.json()
                output_lines.append("Response Body (JSON):")
                output_lines.append(json.dumps(parsed, indent=2, ensure_ascii=False)[:MAX_RESPONSE_SIZE])
            except Exception:
                output_lines.append("Response Body:")
                output_lines.append(body_text[:MAX_RESPONSE_SIZE])
        else:
            output_lines.append("Response Body:")
            output_lines.append(body_text[:MAX_RESPONSE_SIZE])

        return "\n".join(output_lines)

    except httpx.TimeoutException:
        return f"Error: Request timed out after {timeout}s"
    except httpx.ConnectError as e:
        return f"Error: Connection failed — {e}"
    except httpx.UnsupportedProtocol as e:
        return f"Error: Unsupported protocol — {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


HTTP_REQUEST_SCHEMA = {
    "name": "http_request",
    "description": (
        "Make HTTP requests to any URL. Supports GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS "
        "with custom headers, JSON body, form data, and query parameters. "
        "Returns status code, headers, and response body. "
        "Useful for API testing, webhook debugging, REST API calls, and health checks. "
        "SSRF protection blocks internal/private network targets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to send the request to (must be a public URL, not localhost/private IPs)",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                "description": "HTTP method (default: GET)",
                "default": "GET",
            },
            "headers": {
                "type": "object",
                "description": "Request headers as key-value pairs (e.g., {\"Authorization\": \"Bearer token\"})",
            },
            "params": {
                "type": "object",
                "description": "URL query parameters as key-value pairs (e.g., {\"page\": \"1\", \"limit\": \"10\"})",
            },
            "json": {
                "type": "object",
                "description": "JSON request body (sets Content-Type: application/json automatically)",
            },
            "body": {
                "type": "string",
                "description": "Raw request body as string (for non-JSON payloads)",
            },
            "form_data": {
                "type": "object",
                "description": "Form data as key-value pairs (sets Content-Type: application/x-www-form-urlencoded)",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default: 30, max: 120)",
                "default": 30,
            },
            "follow_redirects": {
                "type": "boolean",
                "description": "Whether to follow redirects (default: true)",
                "default": True,
            },
        },
        "required": ["url"],
    },
}

registry.register(
    name="http_request",
    toolset="http",
    schema=HTTP_REQUEST_SCHEMA,
    handler=_handle_http_request,
    emoji="🌐",
    max_result_size_chars=100_000,
)
