"""App integration tools — 500+ external apps via the Nous tool gateway.

Four meta tools that let the LLM discover, authenticate, and execute
real app tools at runtime through the Nous managed tool gateway.

Architecture:
  Hermes → POST JSON → tools-gateway.nousresearch.com/v1/* → External APIs
  Auth:   Bearer <nous_user_token> (subscription-gated)
  Vendor: "tools" in the managed gateway infra (build_vendor_gateway_url)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

from tools.registry import registry
from tools.managed_tool_gateway import (
    is_managed_tool_gateway_ready,
    resolve_managed_tool_gateway,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeouts per endpoint (connect, read)
# ---------------------------------------------------------------------------
_TIMEOUT_SEARCH = httpx.Timeout(30.0, connect=5.0)
_TIMEOUT_SCHEMAS = httpx.Timeout(15.0, connect=5.0)
_TIMEOUT_EXECUTE = httpx.Timeout(120.0, connect=5.0)
_TIMEOUT_CONNECTIONS = httpx.Timeout(30.0, connect=5.0)

# ---------------------------------------------------------------------------
# Module-level cached httpx client — avoids TCP+TLS setup per tool call.
# Follows the same thread-safe staleness pattern as image_generation_tool.py.
# ---------------------------------------------------------------------------
import threading

_http_client: Optional[httpx.Client] = None
_http_client_origin: Optional[str] = None
_http_client_lock = threading.Lock()


def _get_http_client(origin: str, verify: bool = True) -> httpx.Client:
    """Return a reusable httpx.Client, recreated when the origin changes."""
    global _http_client, _http_client_origin
    with _http_client_lock:
        if _http_client is not None and _http_client_origin == origin:
            return _http_client
        if _http_client is not None:
            try:
                _http_client.close()
            except Exception:
                pass
        _http_client = httpx.Client(verify=verify)
        _http_client_origin = origin
        return _http_client


# ---------------------------------------------------------------------------
# Config / availability helpers
# ---------------------------------------------------------------------------

def _read_portal_app_tools_enabled() -> bool:
    """Return True when the portal.app_tools config flag is on."""
    from tools.tool_backend_helpers import portal_app_tools_enabled
    return portal_app_tools_enabled()


def _app_tools_available() -> bool:
    """check_fn: True when subscription is active, gateway reachable, config on."""
    if not _read_portal_app_tools_enabled():
        return False
    return is_managed_tool_gateway_ready("tools")


def _get_current_model_name() -> Optional[str]:
    """Best-effort read of the current model name from config.

    Handles both ``"model": "name"`` and ``"model": {"default": "name"}``
    config shapes.  Returns None if unresolvable (caller should omit the
    field rather than sending garbage).
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        model_cfg = config.get("model")
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default")
            if isinstance(default, str) and default.strip():
                return default.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Gateway HTTP client
# ---------------------------------------------------------------------------

def _gateway_post(
    path: str,
    payload: Dict[str, Any],
    timeout: httpx.Timeout,
) -> Dict[str, Any]:
    """POST JSON to the tool gateway and return the parsed response.

    Never raises — HTTP errors and network failures are returned as dicts
    so the LLM can see them and communicate with the user.
    """
    gateway = resolve_managed_tool_gateway("tools")
    if gateway is None:
        return {
            "error": {
                "code": "GATEWAY_UNAVAILABLE",
                "message": "Nous tool gateway is not available. Check your subscription status.",
            }
        }

    url = f"{gateway.gateway_origin.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {gateway.nous_user_token}",
        "Content-Type": "application/json",
    }

    try:
        client = _get_http_client(url.split("/v1/")[0])
        response = client.post(url, json=payload, headers=headers, timeout=timeout)

        # Return parsed body regardless of status code — the LLM handles errors
        try:
            return response.json()
        except Exception:
            return {
                "error": {
                    "code": f"HTTP_{response.status_code}",
                    "message": response.text[:2000],
                }
            }

    except httpx.TimeoutException as exc:
        return {
            "error": {
                "code": "GATEWAY_TIMEOUT",
                "message": f"Request to {path} timed out: {exc}",
            }
        }
    except Exception as exc:
        return {
            "error": {
                "code": "GATEWAY_UNREACHABLE",
                "message": f"Failed to reach tool gateway: {exc}",
            }
        }


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_app_search_tools(args: dict, **kw) -> str:
    """Search 500+ app integrations for tools matching a use case."""
    payload: Dict[str, Any] = {}

    queries = args.get("queries")
    if queries:
        payload["queries"] = queries

    # session is an OBJECT {id, generate_id} — NOT a string
    session = args.get("session")
    if session is not None:
        payload["session"] = session

    # Auto-inject model name from config (omit if unresolvable)
    model = args.get("model") or _get_current_model_name()
    if model:
        payload["model"] = model

    return json.dumps(_gateway_post("/v1/search", payload, _TIMEOUT_SEARCH),
                      ensure_ascii=False, default=str)


def handle_app_tool_schemas(args: dict, **kw) -> str:
    """Get full input schemas for tools discovered via app_search_tools."""
    payload: Dict[str, Any] = {}

    tool_slugs = args.get("tool_slugs")
    if tool_slugs:
        payload["tool_slugs"] = tool_slugs

    include = args.get("include")
    if include:
        payload["include"] = include

    # session_id is a STRING — not an object
    session_id = args.get("session_id")
    if session_id is not None:
        payload["session_id"] = session_id

    return json.dumps(_gateway_post("/v1/schemas", payload, _TIMEOUT_SCHEMAS),
                      ensure_ascii=False, default=str)


def handle_app_execute_tools(args: dict, **kw) -> str:
    """Execute one or more app tools in parallel."""
    payload: Dict[str, Any] = {}

    tools = args.get("tools")
    if tools:
        payload["tools"] = tools

    # session_id is a STRING
    session_id = args.get("session_id")
    if session_id is not None:
        payload["session_id"] = session_id

    # Strip gateway-internal params that are meaningless in Hermes
    # (sync_response_to_workbench, thought, current_step, current_step_metric)
    # They never enter the payload — we only pick the fields we need.

    return json.dumps(_gateway_post("/v1/execute", payload, _TIMEOUT_EXECUTE),
                      ensure_ascii=False, default=str)


def handle_app_manage_connections(args: dict, **kw) -> str:
    """Check or initiate OAuth/API key connections for app toolkits."""
    payload: Dict[str, Any] = {}

    toolkits = args.get("toolkits")
    if toolkits:
        payload["toolkits"] = toolkits

    reinitiate_all = args.get("reinitiate_all")
    if reinitiate_all is not None:
        payload["reinitiate_all"] = reinitiate_all

    # session_id is a STRING
    session_id = args.get("session_id")
    if session_id is not None:
        payload["session_id"] = session_id

    return json.dumps(_gateway_post("/v1/connections", payload, _TIMEOUT_CONNECTIONS),
                      ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(
    name="app_search_tools",
    toolset="app_tools",
    schema={
        "name": "app_search_tools",
        "description": (
            "Search 500+ app integrations (Gmail, Slack, GitHub, Notion, Google Sheets, "
            "Jira, Linear, Figma, and more) to find tools for a task. Returns tool slugs, "
            "execution plans, pitfalls, and connection status."
        ),
        "parameters": {
            "type": "object",
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "minItems": 1,
                    "description": (
                        "Structured search queries. Split independent app actions "
                        "into separate queries. Each returns 4-6 tools."
                    ),
                    "items": {
                        "type": "object",
                        "required": ["use_case"],
                        "properties": {
                            "use_case": {
                                "type": "string",
                                "maxLength": 1024,
                                "description": (
                                    "Normalized description of the task. Include app "
                                    "names if mentioned. Do NOT include personal "
                                    "identifiers — put those in known_fields."
                                ),
                            },
                            "known_fields": {
                                "type": "string",
                                "description": (
                                    "Known inputs as comma-separated key:value pairs "
                                    "(e.g. 'channel_name:general'). Omit if not relevant."
                                ),
                            },
                        },
                    },
                },
                "session": {
                    "type": "object",
                    "description": "Session context. Pass {generate_id: true} for new workflows, {id: \"EXISTING\"} to continue.",
                    "properties": {
                        "id": {"type": "string", "description": "Existing session ID to reuse."},
                        "generate_id": {"type": "boolean", "description": "Set true for first call of a new workflow."},
                    },
                },
            },
        },
    },
    handler=lambda args, **kw: handle_app_search_tools(args, **kw),
    check_fn=_app_tools_available,
    description="Search 500+ app integrations",
    emoji="🔍",
)

registry.register(
    name="app_tool_schemas",
    toolset="app_tools",
    schema={
        "name": "app_tool_schemas",
        "description": (
            "Get full input parameter schemas for tools discovered via "
            "app_search_tools. Only use slugs from search results — never invent."
        ),
        "parameters": {
            "type": "object",
            "required": ["tool_slugs"],
            "properties": {
                "tool_slugs": {
                    "type": "array",
                    "description": "Tool slugs to retrieve schemas for.",
                    "items": {"type": "string", "minLength": 1},
                },
                "include": {
                    "type": "array",
                    "default": ["input_schema"],
                    "description": "Schema fields to include. Add 'output_schema' for response validation.",
                    "items": {"type": "string", "enum": ["input_schema", "output_schema"]},
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID from a prior app_search_tools call.",
                },
            },
        },
    },
    handler=lambda args, **kw: handle_app_tool_schemas(args, **kw),
    check_fn=_app_tools_available,
    description="Get tool input schemas",
    emoji="📋",
)

registry.register(
    name="app_execute_tools",
    toolset="app_tools",
    schema={
        "name": "app_execute_tools",
        "description": (
            "Execute one or more app tools in parallel (up to 50). "
            "Requires active connection per toolkit. Use schema-compliant arguments only."
        ),
        "parameters": {
            "type": "object",
            "required": ["tools"],
            "properties": {
                "tools": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "description": "Logically independent tools to execute in parallel.",
                    "items": {
                        "type": "object",
                        "required": ["tool_slug", "arguments"],
                        "additionalProperties": False,
                        "properties": {
                            "tool_slug": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Tool slug from search results — never invent.",
                            },
                            "arguments": {
                                "type": "object",
                                "additionalProperties": True,
                                "description": "Arguments matching the tool's input schema exactly.",
                            },
                        },
                    },
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID from a prior app_search_tools call.",
                },
            },
        },
    },
    handler=lambda args, **kw: handle_app_execute_tools(args, **kw),
    check_fn=_app_tools_available,
    max_result_size_chars=50_000,
    description="Execute app tools",
    emoji="⚡",
)

registry.register(
    name="app_manage_connections",
    toolset="app_tools",
    schema={
        "name": "app_manage_connections",
        "description": (
            "Check or initiate OAuth/API key connections for app toolkits. "
            "Returns auth links for inactive connections."
        ),
        "parameters": {
            "type": "object",
            "required": ["toolkits"],
            "properties": {
                "toolkits": {
                    "type": "array",
                    "description": "Toolkit slugs to check or connect (e.g. ['gmail', 'slack']).",
                    "items": {"type": "string"},
                },
                "reinitiate_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force reconnection even for active connections.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID from a prior app_search_tools call.",
                },
            },
        },
    },
    handler=lambda args, **kw: handle_app_manage_connections(args, **kw),
    check_fn=_app_tools_available,
    description="Manage app connections",
    emoji="🔗",
)
