"""
DCG (Destructive Command Guard) integration.

Delegates terminal command safety decisions to the external `dcg` binary
(https://github.com/Dicklesworthstone/destructive_command_guard). DCG is a
battle-tested pattern-based guard that ships curated rule packs and emits a
Claude Code PreToolUse-shaped JSON verdict. When dcg-mode is enabled, Hermes
uses dcg as the sole command guard — Hermes's own dangerous-command heuristics
and tirith prompts are bypassed in favor of dcg's single, higher-signal gate.

The user's stated preference (2026-04): "destructive command guard should block
the worst, everything else generally is runnable."

Contract:
    stdin:  {"tool_name": "Bash", "tool_input": {"command": "..."}}
    stdout: JSON with {"hookSpecificOutput": {"permissionDecision":
            "allow"|"deny"|"ask", "permissionDecisionReason": "...", ...}}
            Empty stdout means allow.
    exit:   always 0 for normal evaluation; non-zero on internal dcg error.

Fail-open policy: if dcg is missing, errors, or times out, we return a
soft-allow and let the caller fall back to other guards. We do NOT silently
allow on genuine deny; deny is only returned when dcg explicitly says so.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3.0


def _dcg_path() -> str | None:
    """Return the path to the dcg binary, or None if not installed."""
    try:
        from hermes_cli.config import load_config
        cfg = (load_config().get("approvals") or {}).get("dcg") or {}
        configured = cfg.get("path") or ""
    except Exception:
        configured = ""
    candidate = configured or "dcg"
    return shutil.which(candidate)


def _dcg_timeout() -> float:
    try:
        from hermes_cli.config import load_config
        cfg = (load_config().get("approvals") or {}).get("dcg") or {}
        return float(cfg.get("timeout", _DEFAULT_TIMEOUT))
    except Exception:
        return _DEFAULT_TIMEOUT


def is_dcg_mode_enabled() -> bool:
    """True when the user has opted into dcg as the sole command guard."""
    if os.getenv("HERMES_DCG_MODE") in ("1", "true", "yes", "on"):
        return True
    try:
        from hermes_cli.config import load_config
        approvals = load_config().get("approvals") or {}
    except Exception:
        return False
    if str(approvals.get("mode", "")).strip().lower() == "dcg":
        return True
    dcg_cfg = approvals.get("dcg") or {}
    return bool(dcg_cfg.get("enabled"))


def check_with_dcg(command: str) -> dict[str, Any]:
    """Ask the dcg binary whether `command` is safe to run.

    Returns a dict with:
        ok:      bool — whether dcg actually produced a verdict
        allow:   bool — True if dcg allowed (or ok=False, i.e. soft-allow)
        reason:  str  — human-readable explanation when deny
        rule_id: str  — dcg rule id when deny, empty otherwise
        raw:     dict — the full hookSpecificOutput, if any
    """
    path = _dcg_path()
    if not path:
        logger.debug("dcg not installed, soft-allow")
        return {"ok": False, "allow": True, "reason": "dcg not installed",
                "rule_id": "", "raw": {}}

    payload = json.dumps({
        "tool_name": "Bash",
        "tool_input": {"command": command},
    })

    try:
        proc = subprocess.run(
            [path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=_dcg_timeout(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("dcg timed out after %.1fs, soft-allow", _dcg_timeout())
        return {"ok": False, "allow": True, "reason": "dcg timeout",
                "rule_id": "", "raw": {}}
    except Exception as exc:
        logger.warning("dcg invocation failed: %s, soft-allow", exc)
        return {"ok": False, "allow": True, "reason": f"dcg error: {exc}",
                "rule_id": "", "raw": {}}

    stdout = (proc.stdout or "").strip()
    # Empty stdout == allow (dcg's convention for non-dangerous commands)
    if not stdout:
        return {"ok": True, "allow": True, "reason": "", "rule_id": "", "raw": {}}

    # dcg emits a pretty banner on deny followed by a JSON object on the
    # last non-empty line. Find the last line that parses as JSON.
    hook_output: dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            hook_output = parsed.get("hookSpecificOutput") or parsed
            break

    if not hook_output:
        # dcg printed something we couldn't parse — treat as soft-allow but
        # log loudly so the user notices.
        logger.warning("dcg produced unparseable output: %r", stdout[:200])
        return {"ok": False, "allow": True, "reason": "dcg unparseable",
                "rule_id": "", "raw": {}}

    decision = str(hook_output.get("permissionDecision", "")).lower()
    if decision == "deny":
        rule_id = hook_output.get("ruleId") or ""
        reason = hook_output.get("permissionDecisionReason") or "blocked by dcg"
        # Keep the reason compact for the agent — full banner is noise.
        short_reason = reason.split("\n\n")[0].strip() if reason else "blocked by dcg"
        return {"ok": True, "allow": False, "reason": short_reason,
                "rule_id": rule_id, "raw": hook_output}

    # ask / allow / anything else → allow
    return {"ok": True, "allow": True, "reason": "", "rule_id": "",
            "raw": hook_output}


__all__ = ["check_with_dcg", "is_dcg_mode_enabled"]
