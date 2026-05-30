from __future__ import annotations

from tools import approval


def test_dangerous_approval_persists_to_control_db(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "worker-a")
    monkeypatch.setenv("HERMES_APPROVER_PROFILE", "default")
    token = approval.set_current_session_key("sess-a")
    try:
        approval._control_approval_create("gw-test-approval", "rm -rf /tmp/x", "delete files", ttl_seconds=60)
        approval._control_approval_decide("gw-test-approval", "once", reason="test")
        assert approval._control_approval_get_choice("gw-test-approval") == "once"
        approval._control_approval_consume("gw-test-approval")
    finally:
        approval.reset_current_session_key(token)

    from hermes_cli import control_db as cp
    conn = cp.connect(root=tmp_path)
    try:
        row = conn.execute("SELECT * FROM cp_approvals WHERE approval_id='gw-test-approval'").fetchone()
        assert row is not None
        assert row["requester_profile"] == "worker-a"
        assert row["requester_instance_id"] == "sess-a"
        assert row["approver_profile"] == "default"
        assert row["status"] == "consumed"
        assert row["decision"] == "approved"
        assert row["consumed_by_instance_id"] == "sess-a"
        outbox_count = conn.execute("SELECT COUNT(*) FROM cp_outbox WHERE subject_type='approval' AND subject_id='gw-test-approval'").fetchone()[0]
        assert outbox_count == 3
    finally:
        conn.close()
