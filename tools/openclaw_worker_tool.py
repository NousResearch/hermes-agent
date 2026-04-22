#!/usr/bin/env python3
"""OpenClaw worker delegation tool.

Wraps the local OpenClaw worker agent so Hermes can offload substantial coding
and debugging tasks to a separate agent process.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from typing import Optional


def _find_openclaw_binary() -> Optional[str]:
    """Resolve an OpenClaw binary path with sensible fallbacks."""
    candidates = [
        os.getenv("OPENCLAW_BIN"),
        shutil.which("openclaw"),
        "/opt/homebrew/bin/openclaw",
        "/usr/local/bin/openclaw",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def check_openclaw_worker_requirements() -> bool:
    """Tool is available when an OpenClaw CLI binary is installed."""
    return _find_openclaw_binary() is not None


def _run_openclaw(cmd: list[str], timeout: int, cwd: Optional[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


def openclaw_worker_tool(
    prompt: str,
    timeout: int = 600,
    workdir: Optional[str] = None,
) -> str:
    """Delegate a single instruction to the local OpenClaw worker agent."""
    prompt = (prompt or "").strip()
    if not prompt:
        return json.dumps({"error": "'prompt' is required"}, ensure_ascii=False)

    binary = _find_openclaw_binary()
    if not binary:
        return json.dumps(
            {
                "error": "OpenClaw CLI not found.",
                "fallback": "Use delegate_task with toolsets=['terminal','file'] or install OpenClaw and retry.",
            },
            ensure_ascii=False,
        )

    cwd = os.path.abspath(os.path.expanduser(workdir)) if workdir else None

    base_cmd = [binary, "agent", "--agent", "worker", "--message", prompt, "--json"]
    with_local = [binary, "agent", "--local", "--agent", "worker", "--message", prompt, "--json"]

    try:
        # Preferred path: explicit local mode.
        proc = _run_openclaw(with_local, timeout=timeout, cwd=cwd)

        # Fallback for older OpenClaw CLIs that don't support --local.
        if proc.returncode != 0 and "--local" in (proc.stderr or ""):
            proc = _run_openclaw(base_cmd, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "error": f"openclaw worker timed out after {timeout}s",
                "command": " ".join(shlex.quote(c) for c in with_local),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "error": f"Failed to execute openclaw worker: {e}",
                "command": " ".join(shlex.quote(c) for c in with_local),
            },
            ensure_ascii=False,
        )

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        return json.dumps(
            {
                "error": "openclaw worker command failed",
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "fallback": "Use delegate_task when worker is unavailable.",
            },
            ensure_ascii=False,
        )

    # OpenClaw may emit JSON on stdout or stderr depending on CLI mode.
    payload_text = stdout or stderr

    # Prefer structured JSON passthrough when possible.
    try:
        parsed = json.loads(payload_text)
        payloads = parsed.get("payloads") if isinstance(parsed, dict) else None
        first_text = None
        if isinstance(payloads, list) and payloads:
            first = payloads[0]
            if isinstance(first, dict):
                first_text = first.get("text")

        meta = parsed.get("meta") if isinstance(parsed, dict) else None
        duration_ms = meta.get("durationMs") if isinstance(meta, dict) else None

        return json.dumps(
            {
                "ok": True,
                "worker_text": first_text,
                "duration_ms": duration_ms,
                "result": parsed if first_text is None else None,
            },
            ensure_ascii=False,
        )
    except Exception:
        return json.dumps(
            {
                "ok": True,
                "worker_text": payload_text,
            },
            ensure_ascii=False,
        )


OPENCLAW_WORKER_SCHEMA = {
    "name": "openclaw_worker",
    "description": (
        "Delegate a coding/debugging instruction to the local OpenClaw worker agent. "
        "This wraps `openclaw agent --local --agent worker` and returns the worker response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Self-contained instruction for the OpenClaw worker.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 600)",
                "minimum": 1,
            },
            "workdir": {
                "type": "string",
                "description": "Optional absolute/relative working directory for the command.",
            },
        },
        "required": ["prompt"],
    },
}


# --- Registry ---
from tools.registry import registry


registry.register(
    name="openclaw_worker",
    toolset="delegation",
    schema=OPENCLAW_WORKER_SCHEMA,
    handler=lambda args, **kw: openclaw_worker_tool(
        prompt=args.get("prompt", ""),
        timeout=int(args.get("timeout", 600)),
        workdir=args.get("workdir"),
    ),
    check_fn=check_openclaw_worker_requirements,
    emoji="🦞",
)
