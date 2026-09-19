from tools import approval


def test_submit_pending_assigns_stable_approval_id():
    session_key = "test-pending-approval-identity"
    try:
        approval.submit_pending(session_key, {"command": "echo ok"})
        first = dict(approval._pending[session_key])

        assert first["approval_id"]
        assert len(first["approval_id"]) == 32

        approval.submit_pending(
            session_key,
            {"command": "echo preserved", "approval_id": "upstream-id"},
        )
        assert approval._pending[session_key]["approval_id"] == "upstream-id"
    finally:
        with approval._lock:
            approval._pending.pop(session_key, None)


def test_submit_pending_isolates_stored_entry_from_caller_mutation():
    session_key = "test-pending-approval-copy-isolation"
    payload = {"command": "echo original"}
    try:
        approval.submit_pending(session_key, payload)
        stored_id = approval._pending[session_key]["approval_id"]

        payload["command"] = "echo mutated"
        payload["approval_id"] = "caller-rewrite"

        stored = approval._pending[session_key]
        assert stored["command"] == "echo original"
        assert stored["approval_id"] == stored_id
    finally:
        with approval._lock:
            approval._pending.pop(session_key, None)
