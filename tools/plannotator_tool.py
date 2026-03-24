"""Native Plannotator session launcher.

This tool wraps operator-configured launch commands so Hermes can open
Plannotator review/annotation sessions without relying on skills alone.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from tools.exposure_helpers import run_command_template
from tools.registry import registry

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATES = {
    "review": "python3 ~/services/plannotator-bridge/start_session.py review{review_target_arg}",
    "annotate": "python3 ~/services/plannotator-bridge/start_session.py annotate {artifact_path}",
    "last": "python3 ~/services/plannotator-bridge/start_session.py last",
}

_ENV_TEMPLATE_BY_ACTION = {
    "review": "HERMES_PLANNOTATOR_REVIEW_TEMPLATE",
    "annotate": "HERMES_PLANNOTATOR_ANNOTATE_TEMPLATE",
    "last": "HERMES_PLANNOTATOR_LAST_TEMPLATE",
}

_PLANNOTATOR_SCHEMA = {
    "name": "plannotator_session",
    "description": (
        "Launch a Plannotator review or annotation session using operator-configured command templates. "
        "Returns the live review URL plus any PID/LOG metadata printed by the launcher."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["review", "annotate", "last"],
                "description": "review: current diff or PR/MR URL. annotate: markdown artifact path. last: last-message flow if the launcher supports it."
            },
            "review_target": {
                "type": "string",
                "description": "Optional PR/MR URL or other review target for action='review'. Omit to review the current local diff."
            },
            "artifact_path": {
                "type": "string",
                "description": "Absolute path to a markdown artifact when action='annotate'."
            },
            "repo_path": {
                "type": "string",
                "description": "Optional working directory for review launches against the current local diff."
            },
            "exposure_strategy": {
                "type": "string",
                "enum": ["auto", "localhost", "cloud77", "tailscale-serve", "tailscale-funnel"],
                "description": "Hint passed through to the launcher template so one Plannotator launcher can support multiple exposure backends."
            },
            "command_template": {
                "type": "string",
                "description": "Optional one-off shell template override. Available placeholders: {artifact_path}, {review_target}, {review_target_arg}, {exposure_strategy}, {repo_path}."
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Launcher timeout in seconds. Default 120."
            }
        },
        "required": ["action"]
    }
}


def plannotator_session_tool(args: dict[str, Any], **_kw) -> str:
    action = (args.get("action") or "").strip().lower()
    if action not in _DEFAULT_TEMPLATES:
        return json.dumps({"error": f"Unsupported action: {action}"})

    artifact_path = (args.get("artifact_path") or "").strip()
    review_target = (args.get("review_target") or "").strip()
    repo_path = (args.get("repo_path") or "").strip() or None
    exposure_strategy = (args.get("exposure_strategy") or "auto").strip().lower() or "auto"
    timeout = int(args.get("timeout_seconds") or 120)

    if action == "annotate" and not artifact_path:
        return json.dumps({"error": "'artifact_path' is required when action='annotate'"})
    if action == "annotate" and not os.path.isabs(artifact_path):
        return json.dumps({"error": "'artifact_path' must be an absolute path"})

    template = _resolve_template(action, args.get("command_template"))
    if not template:
        env_name = _ENV_TEMPLATE_BY_ACTION[action]
        return json.dumps({
            "error": (
                f"No Plannotator launcher template configured for action '{action}'. "
                f"Set {env_name} or pass command_template directly."
            )
        })

    review_target_arg = f" {review_target}" if review_target else ""
    variables = {
        "artifact_path": artifact_path,
        "review_target": review_target,
        "review_target_arg": review_target_arg,
        "exposure_strategy": exposure_strategy,
        "repo_path": repo_path or "",
    }

    try:
        execution = run_command_template(
            template,
            variables=variables,
            cwd=repo_path,
            timeout=timeout,
            env={"PLANNOTATOR_EXPOSURE_STRATEGY": exposure_strategy},
        )
    except KeyError as exc:
        return json.dumps({"error": f"Command template references unknown placeholder: {exc}"})
    except Exception as exc:
        logger.exception("plannotator_session failed")
        return json.dumps({"error": f"Failed to launch Plannotator: {type(exc).__name__}: {exc}"})

    if execution["exit_code"] != 0:
        return json.dumps({
            "error": f"Plannotator launcher failed with exit code {execution['exit_code']}",
            "command": execution["command"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
        })

    if not execution["url"]:
        return json.dumps({
            "error": "Plannotator launcher succeeded but did not report a URL.",
            "command": execution["command"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
        })

    return json.dumps({
        "success": True,
        "action": action,
        "url": execution["url"],
        "pid": execution["pid"],
        "log": execution["log"],
        "command": execution["command"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
        "exposure_strategy": exposure_strategy,
        "suggested_message": _build_suggested_message(execution["url"]),
    })


def _resolve_template(action: str, template_override: str | None) -> str:
    override = (template_override or "").strip()
    if override:
        return override
    env_template = os.getenv(_ENV_TEMPLATE_BY_ACTION[action], "").strip()
    if env_template:
        return env_template
    return _DEFAULT_TEMPLATES[action]


def _build_suggested_message(url: str) -> str:
    return (
        f"Temporary review URL:\n{url}\n\n"
        "What to do\n"
        "- open the link\n"
        "- add comments / replacements\n"
        "- press Send Annotations when done"
    )


registry.register(
    name="plannotator_session",
    toolset="plannotator",
    schema=_PLANNOTATOR_SCHEMA,
    handler=plannotator_session_tool,
    emoji="📝",
)
