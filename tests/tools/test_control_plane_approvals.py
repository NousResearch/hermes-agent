from __future__ import annotations

from tools import approval


def test_dangerous_approval_persists_to_control_db(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "worker-a")
    monkeypatch.setenv("HERMES_APPROVER_PROFILE", "default")
    token = approval.set_current_session_key("sess-a")
    try:
        approver_instance = approval._control_approval_create("gw-test-approval", "rm -rf /tmp/x", "delete files", ttl_seconds=60)
        assert approver_instance
        approval._control_approval_decide("gw-test-approval", "once", reason="test", approver_instance_id=approver_instance)
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


def test_control_plane_approval_prefers_control_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "legacy-worker")
    monkeypatch.setenv("HERMES_PROFILE_ID", "worker-a")
    monkeypatch.setenv("HERMES_CONTROL_INSTANCE_ID", "worker-a:inst")
    monkeypatch.setenv("HERMES_APPROVER_PROFILE", "default")
    approver_instance = approval._control_approval_create("identity-approval", "rm -rf /tmp/x", "delete files", ttl_seconds=60)
    assert approver_instance
    from hermes_cli import control_db as cp

    conn = cp.connect(root=tmp_path)
    try:
        row = conn.execute("SELECT requester_profile, requester_instance_id FROM cp_approvals WHERE approval_id='identity-approval'").fetchone()
        assert tuple(row) == ("worker-a", "worker-a:inst")
    finally:
        conn.close()


def test_strict_control_db_blocks_missing_approval_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE_ID", "worker-a")
    monkeypatch.setenv("HERMES_CONTROL_INSTANCE_ID", "worker-a:missing")
    monkeypatch.setenv("HERMES_APPROVER_PROFILE", "default")
    monkeypatch.setenv("HERMES_APPROVER_INSTANCE_ID", "default:missing")
    from hermes_cli import control_db as cp

    conn = cp.connect(root=tmp_path)
    try:
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        cp.register_instance(conn, "default", instance_id="default:bootstrap", actor_type="bootstrap")
        cp.set_authority_mode(conn, "control_db", actor_type="admin", actor_profile="default", actor_instance_id="default:bootstrap")
    finally:
        conn.close()
    assert approval._control_approval_create("strict-missing", "rm -rf /tmp/x", "delete files", ttl_seconds=60) is None


def test_control_plane_message_tool_uses_db_authorization(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CONTROL_INSTANCE_ID", "worker:live")
    from hermes_cli import control_db as cp
    from tools import control_plane

    conn = cp.connect(root=tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message")
        cp.register_instance(conn, "worker", instance_id="worker:live")
        cp.register_instance(conn, "other", instance_id="other:live")
        mid = cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="close me")
    finally:
        conn.close()

    missing = control_plane._handle_control_plane_message({"root": str(tmp_path), "action": "resolve", "actor_instance_id": "worker:live"})
    assert "message_id" in missing
    denied = control_plane._handle_control_plane_message({"root": str(tmp_path), "action": "resolve", "message_id": mid, "actor_instance_id": "other:live"})
    assert "error" in denied and "does not belong" in denied
    resolved = control_plane._handle_control_plane_message({"root": str(tmp_path), "action": "resolve", "message_id": mid, "actor_instance_id": "worker:live"})
    assert '"status": "resolved"' in resolved
