#!/usr/bin/env python3
"""
Composio OAuth Integration

Composio manages OAuth for 1000+ services (Twitter, YouTube, TikTok, Gmail, Slack, etc.)
and exposes them as agent-callable tools in OpenAI function-calling format.

This module:
  1. Discovers which Composio apps are configured in config.yaml
  2. Converts Composio tool schemas → Hermes registry entries
  3. Provides composio_connect / composio_list_connections management tools
  4. Handles OAuth initiation and token storage transparently via Composio's cloud

Configuration (config.yaml):
    composio:
      api_key: "your-composio-api-key"   # or set COMPOSIO_API_KEY env var
      apps:                               # apps to expose as tools
        - TWITTER
        - YOUTUBE
        - TIKTOK
        - GMAIL
      entity_id: "default"               # per-user entity (default: "default")

Usage:
    The agent can call composio_connect to initiate OAuth flows and
    composio_list_connections to see what's already authenticated.
    Once connected, Twitter/YouTube/etc. tools appear automatically.

OAuth flow:
    1. Agent calls composio_connect(app="TWITTER")
    2. Returns an auth URL for the user to visit
    3. User completes OAuth in browser
    4. Composio stores the token — subsequent tool calls work automatically
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    key = os.getenv("COMPOSIO_API_KEY", "")
    if key:
        return key
    try:
        import yaml
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        cfg_path = hermes_home / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            return cfg.get("composio", {}).get("api_key", "")
    except Exception:
        pass
    return None


def _get_config() -> Dict[str, Any]:
    """Load composio section from config.yaml."""
    try:
        import yaml
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        cfg_path = hermes_home / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            return cfg.get("composio", {})
    except Exception:
        pass
    return {}


def _get_toolset():
    """Return an authenticated ComposioToolSet, or raise if unavailable."""
    try:
        from composio_openai import ComposioToolSet as OpenAIToolSet
        api_key = _get_api_key()
        cfg = _get_config()
        entity_id = cfg.get("entity_id", "default")
        return OpenAIToolSet(api_key=api_key or None, entity_id=entity_id)
    except ImportError:
        pass

    try:
        from composio import ComposioToolSet
        api_key = _get_api_key()
        cfg = _get_config()
        entity_id = cfg.get("entity_id", "default")
        return ComposioToolSet(api_key=api_key or None, entity_id=entity_id)
    except ImportError:
        raise RuntimeError(
            "composio not installed. Run: pip install composio-openai"
        )


def _check_requirements() -> bool:
    if not _get_api_key():
        return False
    try:
        import composio  # noqa: F401
        return True
    except ImportError:
        try:
            import composio_openai  # noqa: F401
            return True
        except ImportError:
            return False


# ── Management tools ───────────────────────────────────────────────────────────

def composio_connect(app: str, entity_id: str = "default", task_id: str = None) -> str:
    """Initiate an OAuth connection for a Composio app. Returns the auth URL."""
    try:
        from composio import ComposioClient, App as ComposioApp
        api_key = _get_api_key()
        if not api_key:
            return json.dumps({
                "error": "COMPOSIO_API_KEY not set. Add it to config.yaml under composio.api_key "
                         "or set the COMPOSIO_API_KEY environment variable.",
                "docs": "https://docs.composio.dev/getting-started/quickstart",
            })

        client = ComposioClient(api_key=api_key)
        cfg = _get_config()
        resolved_entity = entity_id or cfg.get("entity_id", "default")

        # Normalize app name
        app_upper = app.strip().upper()

        entity = client.get_entity(id=resolved_entity)
        connection_request = entity.initiate_connection(app_name=app_upper)

        return json.dumps({
            "ok": True,
            "app": app_upper,
            "entity_id": resolved_entity,
            "auth_url": connection_request.redirectUrl,
            "instructions": (
                f"Visit the URL above to authorize {app_upper}. "
                "Once complete, Composio will store the token and tools will be available."
            ),
        })
    except Exception as e:
        logger.error("[composio] connect error: %s", e)
        return json.dumps({"error": str(e)})


def composio_list_connections(entity_id: str = "default", task_id: str = None) -> str:
    """List active Composio OAuth connections for the entity."""
    try:
        from composio import ComposioClient
        api_key = _get_api_key()
        if not api_key:
            return json.dumps({"error": "COMPOSIO_API_KEY not configured."})

        client = ComposioClient(api_key=api_key)
        cfg = _get_config()
        resolved_entity = entity_id or cfg.get("entity_id", "default")

        entity = client.get_entity(id=resolved_entity)
        connections = entity.get_connections()

        items = []
        for conn in connections:
            items.append({
                "app": getattr(conn, "appName", None) or getattr(conn, "app_name", str(conn)),
                "status": getattr(conn, "status", "unknown"),
                "id": getattr(conn, "id", None),
            })

        return json.dumps({
            "ok": True,
            "entity_id": resolved_entity,
            "connections": items,
            "count": len(items),
        })
    except Exception as e:
        logger.error("[composio] list_connections error: %s", e)
        return json.dumps({"error": str(e)})


def composio_execute_action(
    app: str,
    action: str,
    params: Dict[str, Any] = None,
    entity_id: str = "default",
    task_id: str = None,
) -> str:
    """Execute a specific Composio action directly (for internal use by discovered tools)."""
    try:
        toolset = _get_toolset()
        cfg = _get_config()
        resolved_entity = entity_id or cfg.get("entity_id", "default")

        result = toolset.execute_action(
            action=action,
            params=params or {},
            entity_id=resolved_entity,
        )
        return json.dumps(result if isinstance(result, dict) else {"result": str(result)})
    except Exception as e:
        logger.error("[composio] execute_action %s.%s error: %s", app, action, e)
        return json.dumps({"error": str(e)})


# ── Dynamic tool discovery from Composio ──────────────────────────────────────

def discover_composio_tools() -> int:
    """
    Load configured apps from config.yaml, fetch their tool schemas from
    Composio, and register each as a Hermes tool.

    Returns count of tools registered.
    """
    if not _check_requirements():
        logger.debug("[composio] Skipping discovery — API key or package not available.")
        return 0

    cfg = _get_config()
    apps_raw = cfg.get("apps", [])
    if not apps_raw:
        logger.debug("[composio] No apps configured under composio.apps — skipping discovery.")
        return 0

    try:
        toolset = _get_toolset()
    except RuntimeError as e:
        logger.warning("[composio] %s", e)
        return 0

    # Normalize app names and import App enum
    try:
        from composio import App as ComposioApp
        app_objs = []
        for name in apps_raw:
            name_upper = str(name).strip().upper()
            try:
                app_objs.append(getattr(ComposioApp, name_upper))
            except AttributeError:
                logger.warning("[composio] Unknown app '%s' — skipping.", name_upper)
    except ImportError:
        # Fall back to raw string list
        app_objs = [str(a).strip().upper() for a in apps_raw]

    try:
        schemas = toolset.get_tools(apps=app_objs)
    except Exception as e:
        logger.warning("[composio] Failed to fetch tool schemas: %s", e)
        return 0

    count = 0
    for schema in schemas:
        # Composio returns OpenAI-format tool dicts: {"type": "function", "function": {...}}
        if isinstance(schema, dict) and "function" in schema:
            fn = schema["function"]
            tool_name = fn.get("name", "")
        elif isinstance(schema, dict) and "name" in schema:
            # Raw format without wrapper
            fn = schema
            tool_name = schema.get("name", "")
        else:
            continue

        if not tool_name:
            continue

        # Build Hermes-compatible schema
        hermes_schema = {
            "name": tool_name,
            "description": fn.get("description", f"Composio tool: {tool_name}"),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }

        # Create a closure to capture tool_name
        def _make_handler(tn: str):
            def _handler(args: Dict[str, Any], **kw) -> str:
                return composio_execute_action(
                    app="",
                    action=tn,
                    params=args,
                    task_id=kw.get("task_id"),
                )
            return _handler

        try:
            registry.register(
                name=tool_name,
                toolset="composio",
                schema=hermes_schema,
                handler=_make_handler(tool_name),
                check_fn=_check_requirements,
                requires_env=["COMPOSIO_API_KEY"],
            )
            count += 1
            logger.debug("[composio] Registered tool: %s", tool_name)
        except Exception as e:
            logger.warning("[composio] Failed to register tool %s: %s", tool_name, e)

    logger.info("[composio] Discovered %d tools from %d app(s).", count, len(app_objs))
    return count


# ── Register management tools unconditionally ─────────────────────────────────

registry.register(
    name="composio_connect",
    toolset="composio",
    schema={
        "name": "composio_connect",
        "description": (
            "Initiate an OAuth connection for a third-party service via Composio. "
            "Returns an auth URL the user must visit to authorize access. "
            "Supported apps: TWITTER, YOUTUBE, TIKTOK, GMAIL, SLACK, DISCORD, GITHUB, NOTION, and 1000+ more."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "app": {
                    "type": "string",
                    "description": "App to connect, e.g. TWITTER, YOUTUBE, TIKTOK, GMAIL, NOTION.",
                },
                "entity_id": {
                    "type": "string",
                    "description": "Entity (user) ID. Defaults to 'default'.",
                },
            },
            "required": ["app"],
        },
    },
    handler=lambda args, **kw: composio_connect(
        app=args.get("app", ""),
        entity_id=args.get("entity_id", "default"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_requirements,
    requires_env=["COMPOSIO_API_KEY"],
)

registry.register(
    name="composio_list_connections",
    toolset="composio",
    schema={
        "name": "composio_list_connections",
        "description": "List all active Composio OAuth connections. Shows which apps are authorized.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity (user) ID. Defaults to 'default'.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: composio_list_connections(
        entity_id=args.get("entity_id", "default"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_requirements,
    requires_env=["COMPOSIO_API_KEY"],
)
