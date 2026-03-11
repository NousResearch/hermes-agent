#!/usr/bin/env python3
"""Native xint/xint-rs tool integration.

Provides an LLM-callable wrapper around xint-rs (preferred) or xint (fallback)
for X/Twitter intelligence workflows.
"""

import json
import logging
import shutil
import subprocess
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_ARG_COUNT = 64
MAX_ARG_LENGTH = 500
MAX_STDIO_CHARS = 20_000


def _find_xint_executable(prefer: str = "auto") -> Optional[str]:
    """Resolve executable path by preference order."""
    prefer = (prefer or "auto").strip().lower()

    if prefer == "xint-rs":
        return shutil.which("xint-rs")
    if prefer == "xint":
        return shutil.which("xint")

    # auto
    return shutil.which("xint-rs") or shutil.which("xint")


def check_xint_requirements() -> bool:
    """Tool is available when xint-rs or xint is installed in PATH."""
    return _find_xint_executable("auto") is not None


def _validate_args(action: str, args: Optional[List[str]]) -> Tuple[str, List[str], Optional[str]]:
    """Validate and normalize action/args.

    Returns: (normalized_action, normalized_args, error)
    """
    action = (action or "").strip()

    if args is None:
        args = []

    if not isinstance(args, list):
        return action, [], "args must be an array of strings"

    if len(args) > MAX_ARG_COUNT:
        return action, [], f"Too many args: {len(args)} (max {MAX_ARG_COUNT})"

    normalized: List[str] = []
    for i, arg in enumerate(args):
        s = str(arg)
        if len(s) > MAX_ARG_LENGTH:
            return action, [], f"arg[{i}] too long (max {MAX_ARG_LENGTH} chars)"
        if "\n" in s or "\r" in s or "\x00" in s:
            return action, [], f"arg[{i}] contains forbidden control characters"
        normalized.append(s)

    if any(ch in action for ch in ["\n", "\r", "\x00"]):
        return action, [], "action contains forbidden control characters"

    return action, normalized, None


def _truncate(text: str) -> Tuple[str, bool]:
    if len(text) <= MAX_STDIO_CHARS:
        return text, False
    return text[:MAX_STDIO_CHARS] + "\n...[truncated]", True


def xint_tool(
    action: str,
    args: Optional[List[str]] = None,
    prefer: str = "auto",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    parse_json: bool = False,
    task_id: str = None,
) -> str:
    """Run xint-rs/xint and return structured execution output."""
    action, args, err = _validate_args(action=action, args=args)
    if err:
        return json.dumps({"success": False, "error": err}, ensure_ascii=False)

    executable = _find_xint_executable(prefer)
    if not executable:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "xint tool unavailable: neither 'xint-rs' nor 'xint' was found in PATH. "
                    "Install one of them first."
                ),
                "install_refs": [
                    "https://github.com/0xNyk/xint-rs",
                    "https://github.com/0xNyk/xint",
                ],
            },
            ensure_ascii=False,
        )

    try:
        timeout = int(timeout)
    except Exception:
        timeout = DEFAULT_TIMEOUT_SECONDS
    timeout = max(1, min(timeout, MAX_TIMEOUT_SECONDS))

    cmd = [executable]
    if action:
        cmd.append(action)
    cmd.extend(args)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        stdout, stdout_truncated = _truncate((e.stdout or "") if isinstance(e.stdout, str) else "")
        stderr, stderr_truncated = _truncate((e.stderr or "") if isinstance(e.stderr, str) else "")
        return json.dumps(
            {
                "success": False,
                "error": f"xint command timed out after {timeout}s",
                "executable": executable,
                "command": cmd,
                "timeout_seconds": timeout,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("xint execution failed: %s", e)
        return json.dumps(
            {
                "success": False,
                "error": f"xint execution failed: {type(e).__name__}: {e}",
                "executable": executable,
                "command": cmd,
            },
            ensure_ascii=False,
        )

    stdout, stdout_truncated = _truncate(completed.stdout or "")
    stderr, stderr_truncated = _truncate(completed.stderr or "")

    parsed = None
    parse_error = None
    if parse_json:
        try:
            parsed = json.loads(completed.stdout or "")
        except Exception as e:
            parse_error = str(e)

    result = {
        "success": completed.returncode == 0,
        "executable": executable,
        "command": cmd,
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    if parse_json:
        result["parsed_json"] = parsed
        if parse_error:
            result["parse_error"] = parse_error

    return json.dumps(result, ensure_ascii=False)


XINT_SCHEMA = {
    "name": "xint",
    "description": "Run xint-rs (preferred) or xint (fallback) for X/Twitter intelligence tasks. Use this for account/post analytics and evidence-backed research pulls.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Subcommand to run (for example: search, profile, analyze, report, help)."
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional CLI args as a list of strings, e.g. ['--handle', '0xNyk', '--limit', '100']."
            },
            "prefer": {
                "type": "string",
                "enum": ["auto", "xint-rs", "xint"],
                "description": "Executable preference: auto tries xint-rs first then xint."
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait for command completion (1-600)."
            },
            "parse_json": {
                "type": "boolean",
                "description": "If true, attempts to parse stdout as JSON and include parsed_json in the result."
            },
        },
        "required": ["action"],
    },
}


from tools.registry import registry

registry.register(
    name="xint",
    toolset="xint",
    schema=XINT_SCHEMA,
    handler=lambda args, **kw: xint_tool(
        action=args.get("action", ""),
        args=args.get("args"),
        prefer=args.get("prefer", "auto"),
        timeout=args.get("timeout", DEFAULT_TIMEOUT_SECONDS),
        parse_json=bool(args.get("parse_json", False)),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_xint_requirements,
)
