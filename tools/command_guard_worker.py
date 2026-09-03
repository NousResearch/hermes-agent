"""Subprocess entry point for CPU-bound command guard classification.

The parent sends the command on stdin so payloads and secrets never appear in
argv or process listings.  This module performs classification only; approval
state and human interaction remain in the parent process.
"""

from __future__ import annotations

import json
import sys


def _classify(
    mode: str,
    command: str,
    *,
    sudo_password_configured: bool,
    deny_patterns: list,
) -> dict:
    from tools import approval

    if mode == "unconditional":
        matched, description = approval.detect_hardline_command(command)
        if matched:
            return {"kind": "hardline", "description": description}
        matched, description = approval._check_sudo_stdin_guard(
            command, sudo_password_configured
        )
        if matched:
            return {"kind": "sudo_stdin", "description": description}
        pattern = approval._match_user_deny_globs(command, deny_patterns)
        if pattern is not None:
            return {"kind": "user_deny", "pattern": pattern}
        matched, pattern_key, description = approval.detect_dangerous_command(command)
        if matched:
            return {
                "kind": "dangerous",
                "pattern_key": pattern_key,
                "description": description,
            }
        return {"kind": "allow"}

    raise ValueError("unknown command guard worker mode")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        mode = request["mode"]
        command = request["command"]
        sudo_password_configured = request["sudo_password_configured"]
        deny_patterns = request["deny_patterns"]
        if (
            not isinstance(mode, str)
            or not isinstance(command, str)
            or not isinstance(sudo_password_configured, bool)
            or not isinstance(deny_patterns, list)
        ):
            raise TypeError("invalid command guard request")
        result = _classify(
            mode,
            command,
            sudo_password_configured=sudo_password_configured,
            deny_patterns=deny_patterns,
        )
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        sys.stderr.write(f"command guard worker failed: {type(exc).__name__}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
