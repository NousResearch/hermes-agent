"""Killable subprocess entry point for long command classification."""

from __future__ import annotations

import json
import sys


def _classify(request: dict) -> dict:
    from tools.approval_detection import (
        _check_sudo_stdin_guard,
        detect_dangerous_command,
        detect_hardline_command,
    )
    from tools.approval_floors import _match_user_deny_globs

    command = request["command"]
    mode = request["mode"]
    patterns = request["deny_patterns"]
    sudo_password_configured = request.get("sudo_password_configured")
    if not isinstance(mode, str) or not isinstance(sudo_password_configured, bool):
        raise TypeError("invalid request")
    if mode == "full":
        matched, description = detect_hardline_command(command)
        if matched:
            return {"kind": "hardline", "description": description}
        matched, description = _check_sudo_stdin_guard(
            command, sudo_password_configured,
        )
        if matched:
            return {"kind": "sudo_stdin", "description": description}
    elif mode != "user_deny":
        raise ValueError("unknown mode")
    pattern = _match_user_deny_globs(command, patterns)
    if pattern is not None:
        return {"kind": "user_deny", "pattern": pattern}
    if mode == "full":
        matched, pattern_key, description = detect_dangerous_command(command)
        if matched:
            return {"kind": "dangerous", "pattern_key": pattern_key, "description": description}
    return {"kind": "allow"}


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict) or not isinstance(request.get("command"), str):
            raise TypeError("invalid request")
        if not isinstance(request.get("deny_patterns"), list):
            raise TypeError("invalid request")
        sys.stdout.write(json.dumps(_classify(request), ensure_ascii=False))
        return 0
    except Exception as exc:
        sys.stderr.write(f"command guard worker failed: {type(exc).__name__}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())