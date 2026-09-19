import time

import pytest

from tests.fakes.mcp_oauth_peer import (
    FakeOAuthMCPPeer,
    KnownCredentialLoss,
    OAuthFailurePoint,
    capture_oauth_state,
    raise_known_mutation,
    seed_old_oauth_state,
)
from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions


_GH_76590 = pytest.mark.xfail(
    strict=True,
    raises=KnownCredentialLoss,
    reason="GH #76590: TUI reauthorization partial state suppresses rollback",
)


@pytest.fixture(autouse=True)
def _clear_oauth_sessions():
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions.clear()
    yield
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions.clear()


@pytest.mark.parametrize(
    ("failure_point", "broken_labels"),
    [
        pytest.param(OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY, None),
        pytest.param(
            OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
            ("MISSING", "PARTIAL", "PARTIAL"),
            marks=_GH_76590,
        ),
        pytest.param(
            OAuthFailurePoint.MCP_INITIALIZATION,
            ("NEW", "PARTIAL", "PARTIAL"),
            marks=_GH_76590,
        ),
    ],
)
def test_tui_failed_reauth_preserves_active_state(
    tmp_path, monkeypatch, failure_point, broken_labels
):
    from hermes_cli import mcp_config
    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    reset_manager_for_tests()
    before = seed_old_oauth_state(tmp_path, "reports")
    peer = FakeOAuthMCPPeer(failure_point)
    monkeypatch.setattr(mcp_config, "_probe_single_server", peer.probe)
    monkeypatch.setattr(mcp_config, "_save_mcp_server", lambda *_args: True)
    session_id = f"tui-{failure_point.value}"
    flow = DashboardOAuthFlow(
        flow_id=session_id,
        server_name="reports",
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri="http://127.0.0.1:43113/callback",
    )
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions[session_id] = {
            "session_id": session_id,
            "server_name": "reports",
            "hermes_home": str(tmp_path),
            "flow": flow,
            "httpd": None,
            "created_at": time.time(),
        }

    mcp_oauth_sessions._worker(
        session_id,
        str(tmp_path),
        "reports",
        {"url": "https://mcp.invalid/mcp", "auth": "oauth"},
        False,
    )

    assert peer.connect_timeouts == [315]
    assert peer.completed_events[-1] == failure_point.value
    assert flow.status == "error"
    assert flow.worker_done is True
    assert failure_point.value in (flow.error or "")
    assert mcp_oauth_sessions._sessions[session_id]["httpd"] is None
    after = capture_oauth_state(tmp_path, "reports")
    if broken_labels is None:
        assert after == before
    else:
        raise_known_mutation(
            before=before,
            after=after,
            expected_labels=broken_labels,
            surface="tui",
            failure_point=failure_point,
        )
