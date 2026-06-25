#!/usr/bin/env python3
"""Cursor SDK agent tool.

Runs Cursor Composer/agent sessions through the official ``@cursor/sdk`` bridge.
This is intentionally exposed as an agent tool, not a Hermes model provider: the
Cursor SDK controls its own runtime, state, tools, and file operations.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.registry import registry, tool_error

BRIDGE_DIR = Path(__file__).resolve().parent / "cursor_agent_bridge"
BRIDGE_SCRIPT = BRIDGE_DIR / "cursor-agent.mjs"
DEFAULT_MODEL = "composer-2.5"
DEFAULT_TIMEOUT_SECONDS = 600
MAX_TIMEOUT_SECONDS = 3600


def check_cursor_agent_requirements() -> bool:
    """Cursor agent needs Node plus a Cursor API key.

    The Node package dependency is checked in the handler so the tool can return
    a useful setup error instead of silently disappearing when ``npm install``
    has not been run yet.
    """

    return bool(os.getenv("CURSOR_API_KEY")) and shutil.which("node") is not None


def _coerce_timeout(value: Any) -> int:
    try:
        timeout = int(value or DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS
    return max(1, min(timeout, MAX_TIMEOUT_SECONDS))


def _json_error(message: str, **extra: Any) -> str:
    payload = {"success": False, "error": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _clean_env() -> dict[str, str]:
    """Build a child env without echoing secrets into argv or output."""

    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    return env


def _bridge_dependency_hint() -> str:
    return (
        "Cursor agent bridge dependencies are not installed. Run: "
        f"cd {BRIDGE_DIR} && npm install --omit=dev"
    )


def _bridge_ready() -> bool:
    return BRIDGE_SCRIPT.exists() and (BRIDGE_DIR / "node_modules" / "@cursor" / "sdk").exists()


def cursor_agent(
    *,
    prompt: str | None = None,
    cwd: str | None = None,
    runtime: str = "local",
    model: str = DEFAULT_MODEL,
    thinking: str | None = None,
    cloud_repo_url: str | None = None,
    cloud_starting_ref: str | None = None,
    auto_create_pr: bool = False,
    resume_agent_id: str | None = None,
    timeout_seconds: int | None = None,
    action: str = "run",
) -> str:
    """Run or inspect a Cursor SDK agent.

    Returns a JSON string. The Cursor API key is read only from ``CURSOR_API_KEY``;
    it is deliberately not accepted as a tool argument.
    """

    action = (action or "run").strip().lower()
    runtime = (runtime or "local").strip().lower()
    timeout = _coerce_timeout(timeout_seconds)

    if shutil.which("node") is None:
        return tool_error("Node.js >= 18 is required for the Cursor SDK bridge.")
    if not os.getenv("CURSOR_API_KEY"):
        return tool_error("CURSOR_API_KEY is required for the Cursor SDK bridge.")
    if not BRIDGE_SCRIPT.exists():
        return tool_error(f"Cursor agent bridge script is missing: {BRIDGE_SCRIPT}")
    if not _bridge_ready():
        return _json_error(_bridge_dependency_hint(), setup_needed=True)

    if action not in {"run", "list_models"}:
        return tool_error("action must be 'run' or 'list_models'.")

    payload: dict[str, Any] = {"action": action}

    if action == "run":
        if not prompt or not str(prompt).strip():
            return tool_error("prompt is required when action='run'.")
        if runtime not in {"local", "cloud"}:
            return tool_error("runtime must be 'local' or 'cloud'.")
        if runtime == "cloud" and not cloud_repo_url:
            return tool_error("cloud_repo_url is required when runtime='cloud'.")

        payload.update(
            {
                "prompt": str(prompt),
                "cwd": str(Path(cwd).expanduser()) if cwd else os.getcwd(),
                "runtime": runtime,
                "model": model or DEFAULT_MODEL,
                "thinking": thinking,
                "cloud_repo_url": cloud_repo_url,
                "cloud_starting_ref": cloud_starting_ref,
                "auto_create_pr": bool(auto_create_pr),
                "resume_agent_id": resume_agent_id,
            }
        )

    try:
        proc = subprocess.run(
            ["node", str(BRIDGE_SCRIPT)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(BRIDGE_DIR),
            env=_clean_env(),
        )
    except subprocess.TimeoutExpired:
        return _json_error(
            f"Cursor agent timed out after {timeout} seconds.",
            timeout_seconds=timeout,
        )
    except Exception as exc:
        return _json_error(f"Failed to run Cursor agent bridge: {type(exc).__name__}: {exc}")

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return _json_error(
            "Cursor agent bridge failed.",
            exit_code=proc.returncode,
            stderr=stderr[-4000:],
        )

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return _json_error(
            "Cursor agent bridge returned non-JSON output.",
            stdout=stdout[-4000:],
            stderr=stderr[-4000:],
        )

    if stderr:
        parsed.setdefault("stderr", stderr[-4000:])
    return json.dumps(parsed, ensure_ascii=False)


CURSOR_AGENT_SCHEMA = {
    "name": "cursor_agent",
    "description": (
        "Run Cursor SDK agents, especially Cursor Composer 2.5, as a specialist "
        "coding/design agent from Hermes. Use for repo-aware coding, UI/design "
        "implementation, multi-file edits, refactors, tests, PR prep, or when "
        "the user explicitly asks for Cursor/Composer. This is not a raw chat "
        "model; it invokes Cursor's own agent runtime. The API key is read from "
        "CURSOR_API_KEY and must never be passed as an argument."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "list_models"],
                "default": "run",
                "description": "Use 'run' for an agent task or 'list_models' to inspect Cursor model availability.",
            },
            "prompt": {
                "type": "string",
                "description": "The task prompt for Cursor's agent. Required for action='run'.",
            },
            "cwd": {
                "type": "string",
                "description": "Local working directory for runtime='local'. Defaults to Hermes' current working directory.",
            },
            "runtime": {
                "type": "string",
                "enum": ["local", "cloud"],
                "default": "local",
                "description": "Run against local files or Cursor cloud agents.",
            },
            "model": {
                "type": "string",
                "default": DEFAULT_MODEL,
                "description": "Cursor model id. Defaults to composer-2.5.",
            },
            "thinking": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Optional model thinking/reasoning effort parameter when supported by the selected model.",
            },
            "cloud_repo_url": {
                "type": "string",
                "description": "GitHub repository URL for runtime='cloud'.",
            },
            "cloud_starting_ref": {
                "type": "string",
                "description": "Branch/ref for cloud runtime. Defaults to Cursor's server default if omitted.",
            },
            "auto_create_pr": {
                "type": "boolean",
                "default": False,
                "description": "For cloud runtime only: allow Cursor to create a PR. Defaults to false.",
            },
            "resume_agent_id": {
                "type": "string",
                "description": "Existing Cursor agent id to resume instead of creating a new agent.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_SECONDS,
                "default": DEFAULT_TIMEOUT_SECONDS,
                "description": "Maximum time to wait for the Cursor SDK bridge.",
            },
        },
    },
}


registry.register(
    name="cursor_agent",
    toolset="cursor_agent",
    schema=CURSOR_AGENT_SCHEMA,
    handler=lambda args, **kw: cursor_agent(
        prompt=args.get("prompt"),
        cwd=args.get("cwd"),
        runtime=args.get("runtime", "local"),
        model=args.get("model", DEFAULT_MODEL),
        thinking=args.get("thinking"),
        cloud_repo_url=args.get("cloud_repo_url"),
        cloud_starting_ref=args.get("cloud_starting_ref"),
        auto_create_pr=args.get("auto_create_pr", False),
        resume_agent_id=args.get("resume_agent_id"),
        timeout_seconds=args.get("timeout_seconds"),
        action=args.get("action", "run"),
    ),
    check_fn=check_cursor_agent_requirements,
    requires_env=["CURSOR_API_KEY"],
    emoji="🎼",
    max_result_size_chars=80_000,
)
