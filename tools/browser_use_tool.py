#!/usr/bin/env python3
"""browser-use sidecar tool.

Calls an external browser-use-agent HTTP sidecar for browser tasks that need a
separate long-lived Playwright runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import requests

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT_SECONDS = 240
_HEALTH_TIMEOUT_SECONDS = 5
_MAX_STEPS_LIMIT = 50
_LOCAL_HOSTS = frozenset({
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "browser-use-agent",
    "host.docker.internal",
})
_DEFAULT_ENDPOINTS = (
    "http://browser-use-agent:5000",
    "http://127.0.0.1:5056",
    "http://host.docker.internal:5056",
)


BROWSER_USE_SCHEMA: Dict[str, Any] = {
    "name": "browser_use",
    "description": (
        "Run a browser-use sidecar task through the external browser-use-agent service. "
        "Best for separate long-lived Playwright work when the built-in browser tool is not enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Browser task to run, e.g. 'Open example.com and return the page title.'",
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum browser steps. Clamped to 50. Defaults to 25.",
                "minimum": 1,
                "maximum": _MAX_STEPS_LIMIT,
                "default": 25,
            },
            "headless": {
                "type": "boolean",
                "description": "Whether the sidecar browser should run headless. Defaults to true.",
                "default": True,
            },
            "endpoint": {
                "type": "string",
                "description": "Optional browser-use-agent base URL override.",
            },
        },
        "required": ["task"],
    },
}


def _normalize_endpoint(endpoint: Optional[str]) -> str:
    value = str(endpoint or "").strip()
    if not value:
        return ""
    value = value.rstrip("/")
    for suffix in ("/v1/query", "/health"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value.rstrip("/")


def _candidate_endpoints(explicit: Optional[str] = None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in (
        explicit,
        os.getenv("BROWSER_USE_AGENT_URL", ""),
        *_DEFAULT_ENDPOINTS,
    ):
        endpoint = _normalize_endpoint(raw)
        if not endpoint or endpoint in seen:
            continue
        seen.add(endpoint)
        ordered.append(endpoint)
    return ordered


def _is_local_endpoint(endpoint: str) -> bool:
    try:
        host = (urlparse(endpoint).hostname or "").strip().lower()
    except Exception:
        return False
    return host in _LOCAL_HOSTS


def _request(method: str, url: str, *, timeout: float, json_body: Optional[dict] = None) -> requests.Response:
    session = requests.Session()
    if _is_local_endpoint(url):
        session.trust_env = False
        session.proxies = {}
    try:
        kwargs: Dict[str, Any] = {"timeout": timeout}
        if json_body is not None:
            kwargs["json"] = json_body
        return session.request(method, url, **kwargs)
    finally:
        session.close()


def _health_ok(endpoint: str) -> bool:
    try:
        response = _request("GET", f"{endpoint}/health", timeout=_HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        return bool(isinstance(payload, dict) and payload.get("ok") is True)
    except Exception:
        return False


def check_browser_use_requirements() -> bool:
    return any(_health_ok(endpoint) for endpoint in _candidate_endpoints())


def _coerce_max_steps(value: Any) -> int:
    try:
        steps = int(value)
    except (TypeError, ValueError):
        steps = 25
    return max(1, min(steps, _MAX_STEPS_LIMIT))


def _query_endpoint(endpoint: str, payload: dict) -> str:
    try:
        response = _request(
            "POST",
            f"{endpoint}/v1/query",
            timeout=_QUERY_TIMEOUT_SECONDS,
            json_body=payload,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = ""
        try:
            data = response.json()
            if isinstance(data, dict):
                detail = str(data.get("detail") or data.get("error") or "").strip()
        except ValueError:
            detail = ""
        if not detail:
            detail = (response.text or "").strip()
        raise RuntimeError(f"HTTP {response.status_code}: {detail or 'request failed'}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("invalid JSON response from sidecar") from exc
    return tool_result(
        success=True,
        endpoint=endpoint,
        result=str(data.get("result", "")) if isinstance(data, dict) else str(data),
    )


def browser_use_tool(args: dict, **_kwargs) -> str:
    task = str(args.get("task") or "").strip()
    if not task:
        return tool_error("task is required")

    payload = {
        "task": task,
        "max_steps": _coerce_max_steps(args.get("max_steps", 25)),
        "headless": bool(args.get("headless", True)),
    }

    last_network_error = ""
    for endpoint in _candidate_endpoints(args.get("endpoint")):
        try:
            return _query_endpoint(endpoint, payload)
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("HTTP "):
                return tool_error(
                    f"browser_use endpoint {endpoint} returned {message}",
                    endpoint=endpoint,
                )
            last_network_error = f"{endpoint}: {message}"
            logger.debug("browser_use endpoint failed: %s", last_network_error)

    if not last_network_error:
        last_network_error = "no endpoint configured"
    return tool_error(
        f"browser_use sidecar unavailable ({last_network_error})",
        endpoints=_candidate_endpoints(args.get("endpoint")),
    )


registry.register(
    name="browser_use",
    toolset="browser_use",
    schema=BROWSER_USE_SCHEMA,
    handler=browser_use_tool,
    check_fn=check_browser_use_requirements,
    requires_env=[],
    emoji="🧭",
)
