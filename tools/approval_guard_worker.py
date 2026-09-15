"""Parent-side protocol for isolated long-command classification."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from hermes_cli._subprocess_compat import windows_hide_flags

MIN_COMMAND_CHARS = 4096
TIMEOUT_SECONDS = 10.0


def classify_long_command(command: str, *, full: bool) -> dict | None:
    """Return a validated worker verdict, or None for the in-process short path."""
    if len(command) < MIN_COMMAND_CHARS:
        return None
    from hermes_constants import get_hermes_home
    from tools import approval_context

    try:
        patterns = approval_context._get_approval_config().get("deny") or []
    except Exception:
        patterns = []
    try:
        from agent.secret_scope import get_secret
        has_sudo_password = bool(get_secret("SUDO_PASSWORD", ""))
    except Exception:
        has_sudo_password = False
    env = os.environ.copy()
    env["HERMES_HOME"] = str(get_hermes_home())
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-m", "tools.command_guard_worker"],
        input=json.dumps({
            "mode": "full" if full else "user_deny",
            "command": command,
            "sudo_password_configured": has_sudo_password,
            "deny_patterns": patterns,
        }, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=TIMEOUT_SECONDS,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        creationflags=windows_hide_flags(),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command guard worker exited {completed.returncode}")
    result = json.loads(completed.stdout)
    required = {
        "allow": (),
        "hardline": ("description",),
        "sudo_stdin": ("description",),
        "user_deny": ("pattern",),
        "dangerous": ("pattern_key", "description"),
    }
    kind = result.get("kind") if isinstance(result, dict) else None
    allowed_kinds = (
        {"allow", "hardline", "sudo_stdin", "user_deny", "dangerous"}
        if full else {"allow", "user_deny"}
    )
    if kind not in allowed_kinds:
        raise ValueError("invalid command guard worker response")
    fields = required.get(kind) if isinstance(kind, str) else None
    if fields is None or any(not isinstance(result.get(key), str) or not result[key] for key in fields):
        raise ValueError("invalid command guard worker response")
    return result