"""Behavior contract for stable, recoverable gateway Approval identities."""

from __future__ import annotations

import threading
import json
import math


def _queue_entry(approval, session_key: str, **kwargs):
    entry = approval._ApprovalEntry({"command": "dangerous"}, **kwargs)
    approval._gateway_queues.setdefault(session_key, []).append(entry)
    return entry


def test_pending_approval_exposes_stable_sanitized_public_descriptor(monkeypatch):
    from tools import approval

    session_key = "mobile-approval-public-descriptor"
    fake_secret = "sk-proj-" + "X" * 40
    command = (
        "rm -rf /tmp/hermes-approval-preview && "
        f"export OPENAI_API_KEY={fake_secret}"
    )
    notified: list[dict] = []
    notification = threading.Event()
    result: dict = {}

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 30},
    )

    def notify(descriptor: dict) -> None:
        notified.append(descriptor)
        notification.set()

    approval.register_gateway_notify(session_key, notify)

    def run_guard() -> None:
        token = approval.set_current_session_key(session_key)
        try:
            result.update(approval.check_all_command_guards(command, "local"))
        finally:
            approval.reset_current_session_key(token)

    worker = threading.Thread(target=run_guard, daemon=True)
    worker.start()

    try:
        assert notification.wait(timeout=5), "Approval request was not emitted"
        assert len(notified) == 1

        descriptor = notified[0]
        assert descriptor["approval_id"]
        assert descriptor["state"] == "pending"
        assert descriptor["resolution"] is None
        assert descriptor["created_at"] < descriptor["expires_at"]
        assert fake_secret not in descriptor["command"]

        pending = approval.list_pending_gateway_approvals(session_key)
        assert pending == [descriptor]
    finally:
        approval.resolve_gateway_approval(
            session_key,
            "deny",
            resolve_all=True,
        )
        worker.join(timeout=5)
        approval.unregister_gateway_notify(session_key)
        approval.clear_session(session_key)

    assert not worker.is_alive()
    assert result["approved"] is False


def test_targeted_resolution_is_out_of_order_and_duplicate_safe():
    from tools import approval

    session_key = "mobile-approval-targeted"
    first = _queue_entry(
        approval,
        session_key,
        approval_id="approval-first",
        timeout_seconds=30,
    )
    second = _queue_entry(
        approval,
        session_key,
        approval_id="approval-second",
        timeout_seconds=30,
    )

    try:
        outcome = approval.resolve_gateway_approval_by_id(
            session_key,
            second.approval_id,
            "deny",
            reason="not this one",
            resolution_metadata={"source": "mobile"},
        )

        assert outcome["outcome"] == "resolved"
        assert outcome["approval"]["approval_id"] == second.approval_id
        assert outcome["approval"]["state"] == "resolved"
        assert outcome["approval"]["resolution"]["choice"] == "deny"
        assert outcome["approval"]["resolution"]["reason"] == "not this one"
        assert outcome["approval"]["resolution"]["metadata"] == {
            "source": "mobile"
        }
        assert outcome["approval"]["resolution"]["resolved_at"] >= outcome[
            "approval"
        ]["created_at"]
        assert not first.event.is_set()
        assert second.event.is_set()
        assert [item["approval_id"] for item in approval.list_pending_gateway_approvals(session_key)] == [
            first.approval_id
        ]

        duplicate = approval.resolve_gateway_approval_by_id(
            session_key,
            second.approval_id,
            "once",
        )
        assert duplicate["outcome"] == "already_resolved"
        assert duplicate["approval"] == outcome["approval"]
        assert first.event.is_set() is False
    finally:
        approval.clear_session(session_key)


def test_late_expired_and_stale_resolutions_are_deterministic():
    from tools import approval

    expired_session = "mobile-approval-expired"
    expired = _queue_entry(
        approval,
        expired_session,
        approval_id="approval-expired",
        timeout_seconds=0,
    )

    expired_outcome = approval.resolve_gateway_approval_by_id(
        expired_session,
        expired.approval_id,
        "once",
    )
    assert expired_outcome["outcome"] == "expired"
    assert expired_outcome["approval"]["state"] == "expired"
    assert expired_outcome["approval"]["resolution"]["choice"] is None
    assert expired.event.is_set()

    stale_session = "mobile-approval-stale"
    approval.register_gateway_notify(stale_session, lambda _: None)
    stale = _queue_entry(
        approval,
        stale_session,
        approval_id="approval-stale",
        timeout_seconds=30,
    )
    approval.unregister_gateway_notify(stale_session)

    stale_outcome = approval.resolve_gateway_approval_by_id(
        stale_session,
        stale.approval_id,
        "once",
    )
    assert stale_outcome["outcome"] == "stale"
    assert stale_outcome["approval"]["state"] == "stale"
    assert stale_outcome["approval"]["resolution"]["choice"] is None
    assert stale.event.is_set()

    missing = approval.resolve_gateway_approval_by_id(
        stale_session,
        "a" * 32,
        "once",
    )
    assert missing == {
        "outcome": "not_found",
        "approval_id": "a" * 32,
    }


def test_public_descriptor_redacts_all_payload_and_resolution_metadata_paths():
    from tools import approval

    session_key = "mobile-approval-redaction"
    secret = "ghp_" + "A" * 36
    entry = approval._ApprovalEntry(
        {
            "command": f"curl -H 'Authorization: Bearer {secret}' example.test",
            "description": f"nested {secret}",
            "metadata": {
                "list": [secret, {"deep": secret}],
                "tuple": (secret,),
            },
        },
        approval_id="approval-redacted",
        timeout_seconds=30,
    )
    approval._gateway_queues.setdefault(session_key, []).append(entry)

    pending = approval.list_pending_gateway_approvals(session_key)[0]
    assert secret not in repr(pending)

    resolved = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "deny",
        reason=f"reason {secret}",
        resolution_metadata={"nested": [secret, {"deep": secret}]},
    )
    assert secret not in repr(resolved)

    # Returned descriptors are snapshots, not mutable views into core state.
    resolved["approval"]["command"] = secret
    duplicate = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "deny",
    )
    assert secret not in repr(duplicate)

    missing = approval.resolve_gateway_approval_by_id(
        session_key,
        f"missing-{secret}",
        "deny",
    )
    assert secret not in repr(missing)


def test_concurrent_targeted_responses_resolve_exactly_once():
    from tools import approval

    session_key = "mobile-approval-concurrent"
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-concurrent",
        timeout_seconds=30,
    )
    barrier = threading.Barrier(3)
    outcomes: list[dict] = []

    def resolve(choice: str) -> None:
        barrier.wait()
        outcomes.append(
            approval.resolve_gateway_approval_by_id(
                session_key,
                entry.approval_id,
                choice,
            )
        )

    workers = [
        threading.Thread(target=resolve, args=(choice,))
        for choice in ("once", "deny")
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert sorted(item["outcome"] for item in outcomes) == [
        "already_resolved",
        "resolved",
    ]
    assert outcomes[0]["approval"] == outcomes[1]["approval"]


def test_tombstones_are_short_lived(monkeypatch):
    from tools import approval

    session_key = "mobile-approval-tombstone-ttl"
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="approval-short-lived",
        timeout_seconds=30,
    )
    monkeypatch.setattr(
        approval,
        "_GATEWAY_APPROVAL_TOMBSTONE_TTL_SECONDS",
        0,
    )

    first = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "once",
    )
    after_ttl = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "once",
    )

    assert first["outcome"] == "resolved"
    assert after_ttl == {
        "outcome": "not_found",
        "approval_id": "invalid-approval-id",
    }


def test_invalid_choice_fails_without_consuming_pending_approval():
    from tools import approval

    session_key = "mobile-approval-invalid-choice"
    entry = _queue_entry(
        approval,
        session_key,
        approval_id="b" * 32,
        timeout_seconds=30,
    )

    outcome = approval.resolve_gateway_approval_by_id(
        session_key,
        entry.approval_id,
        "approve-ish",
    )

    assert outcome == {
        "outcome": "invalid_choice",
        "approval_id": "b" * 32,
    }
    assert entry.event.is_set() is False
    assert approval.resolve_gateway_approval(session_key, "approve-ish") == 0
    assert approval.list_pending_gateway_approvals(session_key)[0][
        "approval_id"
    ] == entry.approval_id
    approval.clear_session(session_key)


def test_non_finite_timeouts_fall_back_to_finite_public_expiry():
    from tools import approval

    for timeout in (math.nan, math.inf, -math.inf):
        entry = approval._ApprovalEntry(
            {"command": "dangerous"},
            timeout_seconds=timeout,
        )
        descriptor = entry.public_descriptor()
        assert descriptor["expires_at"] - descriptor["created_at"] == 300
        json.dumps(descriptor, allow_nan=False)
