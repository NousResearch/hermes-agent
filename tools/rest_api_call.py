#!/usr/bin/env python3
"""
REST API Call Tool - Make HTTP requests to external services

Provides the ability to make REST API calls to external HTTP services.
Supports GET, POST, PUT, DELETE, PATCH methods with headers, query parameters,
and JSON body. Returns response as JSON.
"""

import json
import ssl
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, Optional


def rest_api_call(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    verify_ssl: bool = True,
    task_id: Optional[str] = None,
) -> str:
    """
    Make a REST API call to an external HTTP service.

    Args:
        url: Full URL to call
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        headers: HTTP headers as key-value pairs
        body: Request body (for POST, PUT, PATCH)
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates (default: True)
        task_id: Optional task ID for context tracking

    Returns:
        JSON string with response data or error information
    """
    if headers is None:
        headers = {}

    if body is not None and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    try:
        req = urllib.request.Request(
            url,
            method=method.upper(),
            headers=headers,
        )

        if body is not None:
            req.data = json.dumps(body).encode("utf-8")

        context = ssl.create_default_context()
        if not verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            response_body = response.read()
            content_type = response.headers.get("Content-Type", "")

            if "application/json" in content_type:
                try:
                    result = json.loads(response_body.decode("utf-8"))
                except json.JSONDecodeError:
                    result = {"raw": response_body.decode("utf-8", errors="replace")}
            else:
                result = {"raw": response_body.decode("utf-8", errors="replace")}

            return json.dumps({
                "success": True,
                "status_code": response.status,
                "headers": dict(response.headers),
                "body": result,
            }, ensure_ascii=False)

    except urllib.error.HTTPError as e:
        error_body = e.read()
        try:
            error_json = json.loads(error_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_json = {"raw": error_body.decode("utf-8", errors="replace")}

        return json.dumps({
            "success": False,
            "error": f"HTTP {e.code}: {e.reason}",
            "status_code": e.code,
            "body": error_json,
        }, ensure_ascii=False)

    except urllib.error.URLError as e:
        return json.dumps({
            "success": False,
            "error": f"URL error: {str(e.reason)}",
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}",
        }, ensure_ascii=False)


def check_rest_api_call_requirements() -> bool:
    """REST API call tool has no external requirements -- always available."""
    return True


REST_API_CALL_SCHEMA = {
    "name": "rest_api_call",
    "description": (
        "Make REST API calls to external HTTP services. Supports GET, POST, PUT, DELETE, "
        "PATCH methods with headers and JSON body. Use for interacting with REST APIs, "
        "webhooks, and HTTP services.\n\n"
        "Returns:\n"
        "- success: boolean indicating if request succeeded\n"
        "- status_code: HTTP status code\n"
        "- headers: response headers\n"
        "- body: response body (parsed as JSON if possible)\n\n"
        "For errors, returns success: false with error details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to call"
            },
            "method": {
                "type": "string",
                "description": "HTTP method",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers as key-value pairs (e.g., {\"Authorization\": \"Bearer token\"})"
            },
            "body": {
                "type": "object",
                "description": "Request body for POST, PUT, PATCH methods (will be sent as JSON)"
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds"
            },
            "verify_ssl": {
                "type": "boolean",
                "description": "Whether to verify SSL certificates (set to false for self-signed certs)"
            }
        },
        "required": ["url"]
    }
}


from tools.registry import registry

registry.register(
    name="rest_api_call",
    toolset="api",
    schema=REST_API_CALL_SCHEMA,
    handler=lambda args, **kw: rest_api_call(
        url=args.get("url", ""),
        method=args.get("method", "GET"),
        headers=args.get("headers"),
        body=args.get("body"),
        timeout=args.get("timeout", 30),
        verify_ssl=args.get("verify_ssl", True),
    ),
    check_fn=check_rest_api_call_requirements,
    emoji="🌐",
)
