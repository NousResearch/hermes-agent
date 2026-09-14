"""The wire refusal is retryable and happens before queue/history side effects."""
from types import SimpleNamespace
from unittest.mock import Mock


def test_status_and_submit_refusal_share_durable_usage(tmp_path, monkeypatch):
    from hermes_cli import auth, auth_nous, free_tier_usage as usage
    from tui_gateway import server
    from hermes_constants import get_hermes_home
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    guest = {"auth_method": "anonymous", "anon_token": "anon_wire_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    for _ in range(10):
        usage.record_completed_tool(usage.current_identity())
    history = [{"role": "user", "content": "kept"}]
    session = {"agent": SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1"),
               "history": history, "running": False, "profile_home": str(get_hermes_home())}
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *a: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda *a: (session, None))
    response = server.handle_request({"id": "status", "method": "free_tier.status", "params": {"session_id": "session"}})
    assert response["result"]["tool_calls_used"] == 10
    assert response["result"]["capped"] is True
    assert response["result"]["continuation_required"] is True
    claim = Mock(side_effect=AssertionError("must refuse before queue/claim"))
    monkeypatch.setattr(server, "_ensure_active_session_slot", claim)
    blocked = server.handle_request({"id": "send", "method": "prompt.submit", "params": {
        "session_id": "session", "text": "not stored", "queued": True}})
    assert blocked["error"]["code"] == 4092
    assert blocked["error"]["data"] == {"reason": usage.LIMIT_REASON, "retryable": True, "free_tier": usage.status()}
    assert session["history"] is history and len(history) == 1
    claim.assert_not_called()


def test_onboarding_rpc_marks_only_canonical_profile(tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import profiles
    from hermes_cli.onboarding_profile import is_onboarding_profile
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tui_gateway import server
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    response = server.handle_request({"id": "guide", "method": "profiles.ensure_onboarding", "params": {"soul": "Guide"}})
    assert response.get("error") is None
    guide = profiles.get_profile_dir("hermes-setup")
    token = set_hermes_home_override(str(guide))
    try:
        assert is_onboarding_profile()
    finally:
        reset_hermes_home_override(token)
    assert not is_onboarding_profile()
