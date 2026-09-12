"""Verified dashboard identity survives session creation and deferred builds."""
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


@pytest.mark.parametrize("identity", [
    {"user_id": "alice", "provider": "oauth"},
    {"user_id": "server-internal", "provider": "server-internal"},
    None,
])
def test_session_creation_ignores_rpc_identity_and_keeps_verified_principal(monkeypatch, tmp_path, identity):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setenv("HERMES_TUI_USER_ID", "stale-user")
    monkeypatch.setenv("HERMES_TUI_USER_PROVIDER", "stale-provider")
    transport = SimpleNamespace(auth_identity=identity, write=lambda _: True)
    token = bind_transport(transport)
    try:
        result = server._methods["session.create"](1, {
            "source": "desktop", "pty_user_id": "forged", "pty_provider": "forged",
            "dashboard_principal": {"user_id": "forged", "provider": "forged"},
        })
    finally:
        reset_transport(token)
    try:
        assert "error" not in result, result
        record = server._sessions[result["result"]["session_id"]]
        expected = identity if identity and identity["user_id"] == "alice" else {}
        assert record.get("dashboard_principal") == expected
        assert server._ensure_session_db_row(record)
        assert db.get_session(record["session_key"])["user_id"] == expected.get("user_id")
        server._seed_branch_row(record, "branch-child", record["session_key"],
                                [{"role": "user", "content": "hello"}], "desktop", None)
        assert db.get_session("branch-child")["user_id"] == expected.get("user_id")
        # A later transport swap cannot change the principal of a live session.
        record["transport"] = SimpleNamespace(auth_identity={"user_id": "other", "provider": "other"})
        frame = server._compute_host_turn_frame("rid", "sid", record, "prompt")
        assert frame["dashboard_principal"] == expected
    finally:
        db.close()


def test_agent_build_uses_session_principal_without_process_fallback(monkeypatch):
    import run_agent

    captured = {}
    monkeypatch.setattr(run_agent, "AIAgent", lambda **kw: captured.update(kw) or SimpleNamespace())
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_startup_system_prompt", lambda *a: "")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *a: ("model", {}))
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *a: [])
    monkeypatch.setattr(server, "_agent_cbs", lambda *a: {})
    monkeypatch.setenv("HERMES_TUI_USER_ID", "stale-user")
    monkeypatch.setattr(server, "_sessions", {
        "alice": {"dashboard_principal": {"user_id": "alice", "provider": "oauth"}},
        "unowned": {"dashboard_principal": {}},
    })
    server._make_agent("alice", "alice-key")
    assert captured.get("user_id") == "alice"
    server._make_agent("unowned", "unowned-key")
    assert captured.get("user_id") is None
    monkeypatch.setenv("HERMES_TUI_USER_PROVIDER", "oauth")
    token = bind_transport(server._stdio_transport)
    try:
        server._make_agent("profile-child", "profile-key")
        assert captured.get("user_id") == "stale-user"
    finally:
        reset_transport(token)


def test_deferred_and_compute_host_records_preserve_the_principal(monkeypatch):
    import io
    from tui_gateway.compute_host import ComputeHost

    identity = {"user_id": "alice", "provider": "oauth"}
    transport = SimpleNamespace(auth_identity=identity, write=lambda _: True)
    token = bind_transport(transport)
    try:
        record = server._deferred_session_record(
            "key", cols=80, cwd="/tmp", history=[], lease=None)
    finally:
        reset_transport(token)
    assert record["dashboard_principal"] == identity
    frame = server._compute_host_turn_frame("rid", "sid", record, "hello")
    captured = {}
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_make_agent", lambda *a, **kw: captured.update(kw) or SimpleNamespace())
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a: False)
    monkeypatch.setattr(server, "_start_session_services", lambda *a: None)
    monkeypatch.setattr(server, "_hydrate_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_emit", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a: None)
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        result = host._build_server_session(server, frame, "computed")
        assert captured["dashboard_principal"] == identity
        assert result["dashboard_principal"] == identity
        assert result["transport"] is host._transport
    finally:
        host.close()
