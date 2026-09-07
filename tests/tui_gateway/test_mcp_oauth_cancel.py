"""Cancellation wakes the actual callback worker without crossing profile ownership."""

import asyncio
import threading

import pytest

from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions as sessions


@pytest.mark.parametrize("client_redirect", [False, True])
def test_cancel_is_scoped_idempotent_and_releases_worker(
    tmp_path, monkeypatch, client_redirect
):
    monkeypatch.setattr(sessions, "_sessions", {})
    finished = threading.Event()

    def worker(session_id, *_args):
        flow = sessions._sessions[session_id]["flow"]
        try:
            asyncio.run(
                flow.publish_authorization_url(
                    "https://idp.example/authorize?state=test"
                )
            )
            asyncio.run(flow.wait_for_callback(timeout=10))
            flow.mark_approved()
        except RuntimeError as exc:
            flow.mark_error(str(exc))
        finally:
            flow.mark_worker_done()
            finished.set()

    monkeypatch.setattr(sessions, "_worker", worker)
    home = str(tmp_path / "origin")
    result = sessions.start_flow(
        home,
        "reports",
        {"url": "https://mcp.example"},
        client_redirect_uri="http://127.0.0.1:49152/callback"
        if client_redirect
        else None,
    )
    sid = result["session_id"]
    rec = sessions._sessions[sid]
    try:
        assert (
            sessions.cancel_flow(sid, "reports", str(tmp_path / "other"))["ok"] is False
        )
        assert sessions.cancel_flow(sid, "other", home)["ok"] is False
        assert rec["flow"].snapshot()["status"] == "authorization_required"
        assert sessions.cancel_flow(sid, "reports", home)["ok"] is True
        assert finished.wait(5), (
            "cancel must wake the worker, not leave a 5-minute occupied slot"
        )
        assert sessions.poll_flow(sid, "reports")["status"] == "error"
        assert rec["httpd"] is None
        assert sessions.cancel_flow(sid, "reports", home)["ok"] is True
        assert (
            sessions.deliver_callback_flow(sid, "reports", code="late", state="test")[
                "ok"
            ]
            is False
        )
        # A new start can take the per-server slot as soon as the old worker exits.
        finished.clear()
        retry = sessions.start_flow(
            home,
            "reports",
            {"url": "https://mcp.example"},
            client_redirect_uri="http://127.0.0.1:49152/callback",
        )
        assert sessions.cancel_flow(retry["session_id"], "reports", home)["ok"] is True
        assert finished.wait(5)
    finally:
        rec["flow"].mark_error("test cleanup")
        sessions._shutdown_listener(rec)
        finished.wait(5)


def test_cancel_does_not_revoke_an_approved_flow(tmp_path, monkeypatch):
    home = str(tmp_path)
    flow = DashboardOAuthFlow(
        "approved", "reports", None, home, "http://127.0.0.1:49152/callback"
    )
    flow.mark_approved()
    flow.mark_worker_done()
    monkeypatch.setattr(
        sessions,
        "_sessions",
        {
            "approved": {
                "flow": flow,
                "server_name": "reports",
                "hermes_home": home,
                "httpd": None,
            }
        },
    )
    assert sessions.cancel_flow("approved", "reports", home) == {
        "ok": True,
        "status": "approved",
    }
    assert sessions.cancel_flow("missing", "reports", home)["ok"] is False
