"""Focused authority-boundary tests for approval.recover."""

import hashlib

from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


SESSION = "recovery-session"
COMMAND = "rm -rf /tmp/approval-recovery"
DIGEST = hashlib.sha256(COMMAND.encode()).hexdigest()


def _entry(status="pending"):
    entry = _ApprovalEntry({
        "request_id": "request-recovery-1",
        "command": COMMAND,
        "command_digest": DIGEST,
        "execution_generation": 7,
        "turn_id": "turn-7",
    })
    approval._record_approval_recovery(entry, SESSION, status)
    return entry


def setup_function():
    approval._approval_recovery_records.clear()
    approval._approval_recovery_used_tokens.clear()


def test_live_approval_is_not_recovery_admitted_or_consumed():
    entry = _entry()
    result = approval.recover_gateway_approval(
        SESSION, entry.data["request_id"], command_digest=DIGEST,
        recovery_token=entry.data["recovery_token"], execution_generation=7,
        turn_id="turn-7",
    )
    assert result["status"] == "stale"
    assert "still pending" in result["blocker"]
    assert entry.data["recovery_token"] not in approval._approval_recovery_used_tokens


def test_expired_recovery_reserves_fresh_prompt_without_executing_old_command():
    entry = _entry("expired")
    result = approval.recover_gateway_approval(
        SESSION, entry.data["request_id"], command_digest=DIGEST,
        recovery_token=entry.data["recovery_token"], execution_generation=7,
        turn_id="turn-7",
    )
    assert result["status"] == "admitting"
    assert result["command_snapshot"] == COMMAND
    duplicate = approval.recover_gateway_approval(
        SESSION, entry.data["request_id"], command_digest=DIGEST,
        recovery_token=entry.data["recovery_token"], execution_generation=7,
        turn_id="turn-7")
    assert duplicate["status"] == "failed"
    assert "in progress" in duplicate["blocker"]


def test_wrong_session_digest_generation_or_token_is_stale():
    entry = _entry()
    checks = [
        ("other-session", DIGEST, entry.data["recovery_token"], 7, "turn-7"),
        (SESSION, "0" * 64, entry.data["recovery_token"], 7, "turn-7"),
        (SESSION, DIGEST, entry.data["recovery_token"], 8, "turn-7"),
        (SESSION, DIGEST, entry.data["recovery_token"], 7, "turn-8"),
        (SESSION, DIGEST, "wrong-token", 7, "turn-7"),
    ]
    for args in checks:
        assert approval.recover_gateway_approval(
            args[0], entry.data["request_id"], command_digest=args[1],
            recovery_token=args[2], execution_generation=args[3], turn_id=args[4],
        )["status"] == "stale"
