"""Hard guard for commands that mutate the live Hermes gateway service.

Normal dangerous-command approval and yolo mode are not enough for commands that
stop, restart, or SIGKILL the gateway running the agent.  This guard is exact
command + session scoped, short-lived, and blocks cron by default.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from hermes_cli.runtime_safety import classify_gateway_control_text
from utils import env_var_enabled

GATEWAY_CONTROL_TTL_SECONDS = 10 * 60

_lock = threading.Lock()
_approvals: dict[tuple[str, str], float] = {}


@dataclass(frozen=True)
class GatewayControlDecision:
    is_gateway_control: bool
    approved: bool
    reason: str | None = None
    message: str | None = None


def command_hash(command: str) -> str:
    return hashlib.sha256((command or "").encode("utf-8", errors="ignore")).hexdigest()


def approve_gateway_control_command(
    session_key: str,
    command: str,
    *,
    ttl_seconds: int = GATEWAY_CONTROL_TTL_SECONDS,
) -> str:
    """Create a fresh exact-command approval for tests/explicit control flows."""
    digest = command_hash(command)
    expires_at = time.time() + max(1, int(ttl_seconds))
    with _lock:
        _approvals[(session_key or "default", digest)] = expires_at
    return digest


def clear_gateway_control_approvals(session_key: str | None = None) -> None:
    with _lock:
        if session_key is None:
            _approvals.clear()
        else:
            for key in list(_approvals):
                if key[0] == session_key:
                    _approvals.pop(key, None)


def has_fresh_gateway_control_approval(session_key: str, command: str, *, now: float | None = None) -> bool:
    if now is None:
        now = time.time()
    digest = command_hash(command)
    key = (session_key or "default", digest)
    with _lock:
        expires_at = _approvals.get(key)
        if not expires_at:
            return False
        if expires_at < now:
            _approvals.pop(key, None)
            return False
        return True


def check_gateway_control_guard(command: str, session_key: str, *, cron: bool | None = None) -> GatewayControlDecision:
    reason = classify_gateway_control_text(command)
    if not reason:
        return GatewayControlDecision(False, True)
    if cron is None:
        cron = env_var_enabled("HERMES_CRON_SESSION") or os.getenv("HERMES_CRON_SESSION") == "1"
    if cron:
        return GatewayControlDecision(
            True,
            False,
            reason,
            "BLOCKED: Hermes gateway lifecycle command is not allowed from cron. "
            "Run gateway stop/restart/kill commands manually outside cron after explicit review.",
        )
    if has_fresh_gateway_control_approval(session_key, command):
        return GatewayControlDecision(True, True, reason, None)
    return GatewayControlDecision(
        True,
        False,
        reason,
        "BLOCKED: Hermes gateway lifecycle command requires a fresh explicit gateway-control approval "
        "for this exact command and session. Do not retry automatically; run it manually or issue an "
        "explicit approval token from a trusted control flow.",
    )


def gateway_control_block_result(decision: GatewayControlDecision) -> dict[str, Any]:
    return {
        "approved": False,
        "hardline": True,
        "gateway_control": True,
        "description": decision.reason or "gateway_control",
        "message": decision.message or "BLOCKED: Hermes gateway lifecycle command blocked.",
    }
