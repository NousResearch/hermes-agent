"""Shared command-template execution helpers.

These helpers are reused by both service_expose_tool.py and plannotator_tool.py
so operator-configured launcher commands can be rendered and executed through a
single, testable path.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Any

_KEY_VALUE_RE = re.compile(r"^([A-Z][A-Z0-9_]*?)=(.*)$")
_URL_RE = re.compile(r"https?://[^\s'\"]+")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def render_command_template(template: str, variables: dict[str, Any]) -> str:
    """Render a shell command template with safely shell-quoted values."""
    quoted = {key: shlex.quote(_stringify(value)) for key, value in variables.items()}
    return template.format_map(_SafeFormatDict(quoted))


def parse_key_value_lines(*texts: str) -> dict[str, str]:
    """Parse KEY=value lines from one or more text blobs."""
    parsed: dict[str, str] = {}
    for text in texts:
        if not text:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            match = _KEY_VALUE_RE.match(line)
            if match:
                parsed[match.group(1)] = match.group(2).strip()
    return parsed


def extract_first_url(*texts: str) -> str | None:
    """Find the first HTTP(S) URL in stdout/stderr."""
    for text in texts:
        if not text:
            continue
        match = _URL_RE.search(text)
        if match:
            return match.group(0)
    return None


def run_command_template(
    template: str,
    *,
    variables: dict[str, Any],
    cwd: str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render and execute a shell template via ``bash -lc``.

    The template itself is operator-controlled configuration. Dynamic values from
    tool calls are shell-quoted before formatting.
    """
    command = render_command_template(template, variables)
    child_env = os.environ.copy()
    if env:
        child_env.update({k: _stringify(v) for k, v in env.items()})

    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=child_env,
        timeout=timeout,
    )
    parsed = parse_key_value_lines(result.stdout, result.stderr)
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed": parsed,
        "url": parsed.get("URL") or parsed.get("PUBLIC_URL") or extract_first_url(result.stdout, result.stderr),
        "pid": parsed.get("PID"),
        "log": parsed.get("LOG"),
    }
