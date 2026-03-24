"""Generic local-service exposure tool.

Provides a small abstraction layer for publishing a local HTTP service via
localhost URLs, operator-provided cloud routers, or command-template wrappers
for tools like tailscale serve/funnel.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tools.exposure_helpers import build_local_url, normalize_path_fragment, run_command_template
from tools.registry import registry

logger = logging.getLogger(__name__)


_ENV_TEMPLATE_BY_STRATEGY = {
    "cloud77": "HERMES_SERVICE_EXPOSE_CLOUD77_TEMPLATE",
    "tailscale-serve": "HERMES_SERVICE_EXPOSE_TAILSCALE_SERVE_TEMPLATE",
    "tailscale-funnel": "HERMES_SERVICE_EXPOSE_TAILSCALE_FUNNEL_TEMPLATE",
}

_SERVICE_EXPOSE_SCHEMA = {
    "name": "service_expose",
    "description": (
        "Expose or describe access to a local HTTP service. Supports direct localhost URLs, "
        "or configurable command-template strategies for routers/tunnels like cloud77 or "
        "tailscale serve/funnel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["describe", "expose"],
                "description": "'describe' lists supported strategies and configuration. 'expose' returns a usable URL for a local service."
            },
            "strategy": {
                "type": "string",
                "enum": ["localhost", "cloud77", "tailscale-serve", "tailscale-funnel", "command"],
                "description": "Exposure strategy to use. 'command' requires command_template."
            },
            "local_port": {
                "type": "integer",
                "description": "Local port where the HTTP service is listening. Required for action='expose'."
            },
            "local_host": {
                "type": "string",
                "description": "Local host/interface for the service. Defaults to 127.0.0.1."
            },
            "path": {
                "type": "string",
                "description": "Optional URL path suffix, such as '/review' or '/healthz'."
            },
            "service_name": {
                "type": "string",
                "description": "Optional human-friendly service label for templates and logs."
            },
            "requested_host": {
                "type": "string",
                "description": "Optional hostname hint for router/tunnel templates, such as a desired wildcard host."
            },
            "command_template": {
                "type": "string",
                "description": "Optional shell template override. Available placeholders: {local_host}, {local_port}, {local_url}, {path}, {service_name}, {requested_host}, {strategy}."
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory when running a command-template strategy."
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Command timeout for command-template strategies. Default 120 seconds."
            }
        },
        "required": []
    }
}


def service_expose_tool(args: dict[str, Any], **_kw) -> str:
    action = (args.get("action") or "describe").strip().lower()
    if action == "describe":
        return json.dumps(_describe_strategies())
    if action != "expose":
        return json.dumps({"error": f"Unsupported action: {action}"})
    return json.dumps(_handle_expose(args))


def _describe_strategies() -> dict[str, Any]:
    return {
        "strategies": [
            {
                "name": "localhost",
                "works_without_extra_setup": True,
                "description": "Returns a direct local URL such as http://127.0.0.1:19432/."
            },
            {
                "name": "cloud77",
                "env_template": _ENV_TEMPLATE_BY_STRATEGY["cloud77"],
                "description": "Runs an operator-supplied command template that should provision a cloud77-style routed URL and print URL=..."
            },
            {
                "name": "tailscale-serve",
                "env_template": _ENV_TEMPLATE_BY_STRATEGY["tailscale-serve"],
                "description": "Runs an operator-supplied command template for tailnet-only publishing, e.g. tailscale serve."
            },
            {
                "name": "tailscale-funnel",
                "env_template": _ENV_TEMPLATE_BY_STRATEGY["tailscale-funnel"],
                "description": "Runs an operator-supplied command template for public publishing, e.g. tailscale funnel."
            },
            {
                "name": "command",
                "description": "Runs a one-off command template supplied directly in the tool call."
            },
        ],
        "template_placeholders": [
            "{local_host}",
            "{local_port}",
            "{local_url}",
            "{path}",
            "{service_name}",
            "{requested_host}",
            "{strategy}",
        ],
        "expected_command_output": ["URL=...", "PID=...", "LOG=..."],
    }


def _handle_expose(args: dict[str, Any]) -> dict[str, Any]:
    strategy = (args.get("strategy") or "localhost").strip().lower()
    local_port = args.get("local_port")
    if not local_port:
        return {"error": "'local_port' is required when action='expose'"}
    try:
        port = int(local_port)
    except (TypeError, ValueError):
        return {"error": "'local_port' must be an integer"}

    local_host = (args.get("local_host") or "127.0.0.1").strip() or "127.0.0.1"
    path = normalize_path_fragment(args.get("path"))
    service_name = (args.get("service_name") or "service").strip() or "service"
    requested_host = (args.get("requested_host") or "").strip()
    local_url = build_local_url(local_host, port, path)

    if strategy == "localhost":
        return {
            "success": True,
            "strategy": strategy,
            "url": local_url,
            "public": False,
            "service_name": service_name,
        }

    template = (args.get("command_template") or "").strip()
    if not template and strategy != "command":
        import os

        env_name = _ENV_TEMPLATE_BY_STRATEGY.get(strategy)
        if env_name:
            template = os.getenv(env_name, "").strip()

    if not template:
        if strategy == "command":
            return {"error": "'command_template' is required when strategy='command'"}
        env_name = _ENV_TEMPLATE_BY_STRATEGY.get(strategy, "<unknown>")
        return {
            "error": (
                f"No command template configured for strategy '{strategy}'. "
                f"Set {env_name} or pass command_template directly."
            )
        }

    try:
        execution = run_command_template(
            template,
            variables={
                "strategy": strategy,
                "local_host": local_host,
                "local_port": port,
                "path": path,
                "service_name": service_name,
                "requested_host": requested_host,
                "local_url": local_url,
            },
            cwd=(args.get("cwd") or None),
            timeout=int(args.get("timeout_seconds") or 120),
        )
    except KeyError as exc:
        return {"error": f"Command template references unknown placeholder: {exc}"}
    except Exception as exc:
        logger.exception("service_expose failed")
        return {"error": f"Failed to run strategy '{strategy}': {type(exc).__name__}: {exc}"}

    if execution["exit_code"] != 0:
        return {
            "error": f"Exposure command failed with exit code {execution['exit_code']}",
            "strategy": strategy,
            "command": execution["command"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
        }

    if not execution["url"]:
        return {
            "error": "Exposure command succeeded but did not report a URL.",
            "strategy": strategy,
            "command": execution["command"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
        }

    return {
        "success": True,
        "strategy": strategy,
        "service_name": service_name,
        "url": execution["url"],
        "pid": execution["pid"],
        "log": execution["log"],
        "command": execution["command"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
        "local_url": local_url,
        "public": strategy != "tailscale-serve",
    }


registry.register(
    name="service_expose",
    toolset="service_exposure",
    schema=_SERVICE_EXPOSE_SCHEMA,
    handler=service_expose_tool,
    emoji="🌐",
)
