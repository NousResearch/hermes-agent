"""Generic local-service exposure tool.

Provides a small abstraction layer for publishing a local HTTP service via
localhost URLs, reverse proxies, private tunnels, or public tunnels through
operator-configured command templates.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from tools.exposure_helpers import build_local_url, normalize_path_fragment, run_command_template
from tools.registry import registry

logger = logging.getLogger(__name__)

_LOCALHOST = "localhost"
_COMMAND = "command"
_REVERSE_PROXY = "reverse-proxy"
_TAILSCALE_SERVE = "tailscale-serve"
_TAILSCALE_FUNNEL = "tailscale-funnel"

_STRATEGY_SPECS = {
    _LOCALHOST: {
        "description": "Returns a direct local URL such as http://127.0.0.1:19432/.",
        "works_without_extra_setup": True,
        "public": False,
        "env_template": None,
    },
    _REVERSE_PROXY: {
        "description": "Runs an operator-supplied command template that should provision a routed URL through a configured reverse proxy and print URL=...",
        "works_without_extra_setup": False,
        "public": True,
        "env_template": "HERMES_SERVICE_EXPOSE_REVERSE_PROXY_TEMPLATE",
    },
    _TAILSCALE_SERVE: {
        "description": "Runs an operator-supplied command template for tailnet-only publishing, e.g. tailscale serve.",
        "works_without_extra_setup": False,
        "public": False,
        "env_template": "HERMES_SERVICE_EXPOSE_TAILSCALE_SERVE_TEMPLATE",
    },
    _TAILSCALE_FUNNEL: {
        "description": "Runs an operator-supplied command template for public publishing, e.g. tailscale funnel.",
        "works_without_extra_setup": False,
        "public": True,
        "env_template": "HERMES_SERVICE_EXPOSE_TAILSCALE_FUNNEL_TEMPLATE",
    },
    _COMMAND: {
        "description": "Runs a one-off command template supplied directly in the tool call.",
        "works_without_extra_setup": False,
        "public": True,
        "env_template": None,
    },
}

_SERVICE_EXPOSE_SCHEMA = {
    "name": "service_expose",
    "description": (
        "Expose or describe access to a local HTTP service. Supports direct localhost URLs, "
        "or configurable command-template strategies for reverse proxies and tunnels."
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
                "enum": [_LOCALHOST, _REVERSE_PROXY, _TAILSCALE_SERVE, _TAILSCALE_FUNNEL, _COMMAND],
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
                "description": "Optional hostname hint for reverse-proxy or tunnel templates."
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
                "name": name,
                "description": spec["description"],
                "works_without_extra_setup": spec["works_without_extra_setup"],
                **({"env_template": spec["env_template"]} if spec.get("env_template") else {}),
            }
            for name, spec in _STRATEGY_SPECS.items()
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
    normalized = _normalize_expose_args(args)
    error = _validate_expose_args(normalized)
    if error:
        return {"error": error}

    if normalized["strategy"] == _LOCALHOST:
        return _build_localhost_result(normalized)

    template, template_error = _resolve_strategy_template(normalized)
    if template_error:
        return {"error": template_error}

    return _run_template_strategy(normalized, template)


def _normalize_expose_args(args: dict[str, Any]) -> dict[str, Any]:
    local_host = (args.get("local_host") or "127.0.0.1").strip() or "127.0.0.1"
    path = normalize_path_fragment(args.get("path"))
    service_name = (args.get("service_name") or "service").strip() or "service"
    requested_host = (args.get("requested_host") or "").strip()
    local_port = args.get("local_port")
    port = None
    if local_port not in (None, ""):
        try:
            port = int(local_port)
        except (TypeError, ValueError):
            port = local_port

    return {
        "strategy": (args.get("strategy") or _LOCALHOST).strip().lower(),
        "local_port": port,
        "local_host": local_host,
        "path": path,
        "service_name": service_name,
        "requested_host": requested_host,
        "local_url": build_local_url(local_host, int(port), path) if isinstance(port, int) else None,
        "command_template": (args.get("command_template") or "").strip(),
        "cwd": args.get("cwd") or None,
        "timeout_seconds": int(args.get("timeout_seconds") or 120),
    }


def _validate_expose_args(args: dict[str, Any]) -> str | None:
    strategy = args["strategy"]
    if strategy not in _STRATEGY_SPECS:
        return f"Unsupported strategy: {strategy}"
    if args["local_port"] in (None, ""):
        return "'local_port' is required when action='expose'"
    if not isinstance(args["local_port"], int):
        return "'local_port' must be an integer"
    if strategy == _COMMAND and not args["command_template"]:
        return "'command_template' is required when strategy='command'"
    return None


def _build_localhost_result(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": True,
        "strategy": args["strategy"],
        "url": args["local_url"],
        "public": _STRATEGY_SPECS[args["strategy"]]["public"],
        "service_name": args["service_name"],
    }


def _resolve_strategy_template(args: dict[str, Any]) -> tuple[str | None, str | None]:
    if args["command_template"]:
        return args["command_template"], None

    spec = _STRATEGY_SPECS[args["strategy"]]
    template_names = [spec["env_template"]] if spec.get("env_template") else []

    for env_name in template_names:
        template = os.getenv(env_name, "").strip()
        if template:
            return template, None

    if args["strategy"] == _COMMAND:
        return None, "'command_template' is required when strategy='command'"

    hint = " or ".join(template_names) if template_names else "command_template"
    return None, f"No command template configured for strategy '{args['strategy']}'. Set {hint} or pass command_template directly."


def _run_template_strategy(args: dict[str, Any], template: str) -> dict[str, Any]:
    try:
        execution = run_command_template(
            template,
            variables={
                "strategy": args["strategy"],
                "local_host": args["local_host"],
                "local_port": args["local_port"],
                "path": args["path"],
                "service_name": args["service_name"],
                "requested_host": args["requested_host"],
                "local_url": args["local_url"],
            },
            cwd=args["cwd"],
            timeout=args["timeout_seconds"],
        )
    except KeyError as exc:
        return {"error": f"Command template references unknown placeholder: {exc}"}
    except Exception as exc:
        logger.exception("service_expose failed")
        return {"error": f"Failed to run strategy '{args['strategy']}': {type(exc).__name__}: {exc}"}

    if execution["exit_code"] != 0:
        return _build_execution_error("Exposure command failed with exit code {exit_code}", args, execution)
    if not execution["url"]:
        return _build_execution_error("Exposure command succeeded but did not report a URL.", args, execution)

    return {
        "success": True,
        "strategy": args["strategy"],
        "service_name": args["service_name"],
        "url": execution["url"],
        "pid": execution["pid"],
        "log": execution["log"],
        "command": execution["command"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
        "local_url": args["local_url"],
        "public": _STRATEGY_SPECS[args["strategy"]]["public"],
    }


def _build_execution_error(message_template: str, args: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": message_template.format(exit_code=execution["exit_code"]),
        "strategy": args["strategy"],
        "command": execution["command"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
    }


registry.register(
    name="service_expose",
    toolset="service_exposure",
    schema=_SERVICE_EXPOSE_SCHEMA,
    handler=service_expose_tool,
    emoji="🌐",
)
