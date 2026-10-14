"""Controller ownership comes from the session record, never the ambient profile."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.browser_control_broker import get_browser_control_broker
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server
from tui_gateway.methods_browser_control import _principal_digest


@pytest.mark.parametrize("home_kind", ["secondary", "missing", "unknown"])
def test_controller_registration_resolves_session_profile(tmp_path, monkeypatch, home_kind):
    root = tmp_path / ".hermes"
    launch = root / "profiles" / "launch"
    secondary = root / "profiles" / "secondary"
    for home in (launch, secondary):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("gateway.browser_control_broker.browser_control_enabled", lambda: True)
    identity = {"user_id": "profile-test", "provider": "fixture"}
    transport = SimpleNamespace(auth_identity=identity, write=lambda frame: True)
    session_id = "profile-owner-test"
    record = {"transport": transport, "session_key": "stored-profile-owner-test"}
    if home_kind != "missing":
        record["profile_home"] = str(secondary if home_kind == "secondary" else tmp_path / "unknown")
    monkeypatch.setitem(server._sessions, session_id, record)
    broker = get_browser_control_broker()
    token = set_hermes_home_override(None)
    scope = None
    try:
        response = server.dispatch({
            "jsonrpc": "2.0", "id": 1, "method": "browser.controller.register",
            "params": {
                "protocol_version": 1, "session_id": session_id,
                "controller_id": "controller", "browser_profile_id": "browser",
                "capabilities": ["browser_navigate"],
            },
        }, transport)
        scope = broker.scope_for_session(
            session_id=session_id, principal_id=_principal_digest(identity),
            transport_family="cloud-ticket-ws")
        if home_kind == "secondary":
            assert "result" in response, response
            assert response["result"]["scope"]["profile_id"] == "secondary"
            assert scope is not None and scope.profile_id == "secondary"
        else:
            assert "error" in response, response
            assert response["error"]["code"] == 4403
            assert scope is None
    finally:
        if scope is not None:
            broker.detach(scope, owner=transport, notify_controller=False)
        reset_hermes_home_override(token)
