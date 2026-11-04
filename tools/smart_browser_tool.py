#!/usr/bin/env python3
"""
Smart Browser Tool — calls the smart-browser sidecar over HTTP.

The heavy lifting (browser-use, playwright, Chromium) runs in a separate
Docker container (smart-browser service). This tool is a thin HTTP client
that forwards the request and passes the active session's LLM config so
the sidecar uses the same provider/model as the Hermes agent.

Registered as: smart_browser
Toolset: browser

Args:
  instruction      (str, required)  Natural-language task
  session_id       (str, optional)  ID for log isolation; auto-generated if absent
  timeout_seconds  (int, optional)  Per-attempt timeout; default 120
  max_steps        (int, optional)  Max agent steps per attempt; default 20

Returns JSON:
  {
    "success": bool,
    "result":  str | null,
    "steps_taken": int,
    "attempts": int,
    "error": str | null
  }
"""

import json
import logging
import os
import uuid

from tools.registry import registry

logger = logging.getLogger(__name__)

_SIDECAR_URL = os.environ.get("SMART_BROWSER_URL", "http://smart-browser:9181/run")


def _handle_smart_browser(args: dict, **_) -> str:
    import urllib.request
    import urllib.error

    instruction     = args["instruction"]
    session_id      = args.get("session_id") or uuid.uuid4().hex[:12]
    timeout_seconds = int(args.get("timeout_seconds", 120))
    max_steps       = int(args.get("max_steps", 20))

    payload: dict = {
        "instruction":     instruction,
        "session_id":      session_id,
        "timeout_seconds": timeout_seconds,
        "max_steps":       max_steps,
    }

    # Forward the active session's LLM config so the sidecar uses the same model
    try:
        from tools._agent_context import get_llm_config
        ctx = get_llm_config()
        if ctx["base_url"] and ctx["api_key"] and ctx["model"]:
            payload["llm_base_url"] = ctx["base_url"]
            payload["llm_api_key"]  = ctx["api_key"]
            payload["llm_model"]    = ctx["model"]
            logger.debug(
                "smart_browser: forwarding session LLM %s @ %s",
                ctx["model"], ctx["base_url"],
            )
    except Exception:
        pass

    # Forward the session identity so the sidecar can send X-Hermes-* headers
    # to the LLM gateway. Cron jobs use their owner as the attribution key.
    try:
        from gateway.session_context import get_session_env
        user_id = (
            get_session_env("HERMES_SESSION_USER_ID", "").strip()
            or get_session_env("HERMES_CRON_JOB_OWNER_ID", "").strip()
        )
        user_label = (
            get_session_env("HERMES_SESSION_USER_NAME", "").strip()
            or user_id
        )
        platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip() or "browser"
        if user_id:
            payload["user_id"] = user_id
            payload["user_label"] = user_label
        payload["platform"] = platform
    except Exception:
        pass

    logger.info(
        "smart_browser → sidecar session=%s instruction=%r",
        session_id, instruction[:120],
    )

    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            _SIDECAR_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Total timeout = per-attempt timeout × retries + buffer
        http_timeout = timeout_seconds * 3 + 60
        with urllib.request.urlopen(req, timeout=http_timeout) as resp:
            result = json.loads(resp.read().decode())
        return json.dumps(result, ensure_ascii=False)

    except urllib.error.URLError as exc:
        error = f"Sidecar unreachable ({_SIDECAR_URL}): {exc.reason}"
        logger.error("smart_browser: %s", error)
        return json.dumps({
            "success": False, "result": None,
            "steps_taken": 0, "attempts": 0, "error": error,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("smart_browser sidecar call failed: %s", error)
        return json.dumps({
            "success": False, "result": None,
            "steps_taken": 0, "attempts": 0, "error": error,
        })


def _check_available() -> bool:
    # Ping the sidecar health endpoint; fail gracefully if not up yet
    import urllib.request
    health_url = _SIDECAR_URL.replace("/run", "/health")
    try:
        urllib.request.urlopen(health_url, timeout=2)
        return True
    except Exception:
        return False


_SMART_BROWSER_SCHEMA = {
    "name": "smart_browser",
    "description": (
        "AI-driven browser agent. Accepts a natural-language instruction, "
        "autonomously navigates websites, clicks, types, and extracts data, "
        "then returns a structured result. Uses stealth Chromium. "
        "Good for: scraping, form submission, multi-step web workflows, "
        "login-gated pages, research tasks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "Natural-language task for the browser agent to execute.",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Optional session identifier for log isolation. "
                    "Auto-generated if not provided."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Per-attempt timeout in seconds. Default 120.",
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum agent steps per attempt. Default 20.",
            },
        },
        "required": ["instruction"],
    },
}


# NOTE: registration must be a BARE top-level call — the auto-discovery AST
# scan (tools/registry.py::_module_registers_tools) only picks up modules with
# a top-level ``registry.register(...)`` expression statement. Wrapping it in
# a function or try-block silently disables the tool (regression fixed
# 2026-08-02).
registry.register(
    name="smart_browser",
    toolset="browser",
    schema=_SMART_BROWSER_SCHEMA,
    handler=_handle_smart_browser,
    check_fn=_check_available,
    is_async=False,
    description=_SMART_BROWSER_SCHEMA["description"],
    emoji="🌐",
)
logger.info("smart_browser tool registered (HTTP sidecar mode)")
