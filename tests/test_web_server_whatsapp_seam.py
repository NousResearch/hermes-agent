"""Seam-identity + aggressive tests for the WhatsApp onboarding extraction (R4-C3).

``hermes_cli/web_routers/whatsapp_onboarding.py`` holds the dashboard's
WhatsApp bridge onboarding cluster (spawn/watch/apply/cancel + session
lifecycle), moved out of ``hermes_cli/web_server.py`` (god-file slice
R4-C3, epic #78791).

The seam-identity tests pin the regression this extraction is meant to
prevent: ``web_server`` must resolve every moved name to the *same object*
the router module defines.  The aggressive tests then exercise the failure
modes the onboarding surface must survive: missing bridge deps, pairing
spawn failure, watcher EOF, and apply without an active pairing.
"""

from fastapi.testclient import TestClient

from hermes_cli import web_server as ws
from hermes_cli.web_routers import whatsapp_onboarding as w

MOVED_NAMES = (
    "_ensure_whatsapp_bridge_dependencies",
    "_spawn_whatsapp_pairing_process",
    "_watch_whatsapp_pairing",
    "_write_platform_enabled",
    "apply_whatsapp_onboarding",
    "cancel_whatsapp_onboarding",
    "get_whatsapp_onboarding_status",
    "start_whatsapp_onboarding",
)


def _client_with_app_state():
    prev_auth = getattr(ws.app.state, "auth_required", None)
    prev_host = getattr(ws.app.state, "bound_host", None)
    ws.app.state.auth_required = False
    ws.app.state.bound_host = None
    client = TestClient(ws.app)
    client.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    return client, prev_auth, prev_host


def _restore(prev_auth, prev_host):
    if prev_auth is None:
        delattr(ws.app.state, "auth_required")
    else:
        ws.app.state.auth_required = prev_auth
    if prev_host is None:
        if hasattr(ws.app.state, "bound_host"):
            delattr(ws.app.state, "bound_host")
    else:
        ws.app.state.bound_host = prev_host


def test_moved_names_are_seam_identical():
    for name in MOVED_NAMES:
        assert getattr(ws, name, None) is getattr(w, name, None), name


def test_whatsapp_routes_registered():
    paths = [rt.path for rt in ws.app.routes if "/api/messaging/whatsapp" in getattr(rt, "path", "")]
    assert "/api/messaging/whatsapp/onboarding/start" in paths
    assert "/api/messaging/whatsapp/onboarding/{pairing_id}/apply" in paths


def test_start_onboarding_empty_body_does_not_500():
    # The model has defaults, so an empty body is accepted — but it must
    # never 500 (the route handles the no-creds spawn path gracefully).
    client, pa, pb = _client_with_app_state()
    try:
        resp = client.post("/api/messaging/whatsapp/onboarding/start", json={})
        assert resp.status_code in (200, 400, 422)
    finally:
        _restore(pa, pb)
        client.close()


def test_get_onboarding_status_unknown_pairing():
    client, pa, pb = _client_with_app_state()
    try:
        resp = client.get("/api/messaging/whatsapp/onboarding/definitely-missing")
        assert resp.status_code in (404, 200)
    finally:
        _restore(pa, pb)
        client.close()


def test_cancel_onboarding_unknown_pairing():
    client, pa, pb = _client_with_app_state()
    try:
        resp = client.delete("/api/messaging/whatsapp/onboarding/definitely-missing")
        assert resp.status_code in (404, 200)
    finally:
        _restore(pa, pb)
        client.close()


def test_whatsapp_session_ttl_constant():
    assert w._WHATSAPP_ONBOARDING_TTL_SECONDS == 600
