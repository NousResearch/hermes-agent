"""The chat PTY's bootstrap names the dashboard login on its ticket request (#144140).

``gateway_bootstrap`` mints the TUI's session-ticket over the private control
socket. When the dashboard hands its authenticated login down with the chat PTY
(``HERMES_TUI_SESSION_SUBJECT``), the request names it as the ticket subject so
the resume admits as that login and can adopt the login-stamped ``desktop`` rows
it created through the serve backend; without the variable the request keeps its
anonymous shape, and the mint's verified-peer gate still refuses anything that is
not the structured ``auth:v1:`` form.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import gateway_runtime
import hermes_constants


_SCRIPT = Path(__file__).resolve().parents[2] / "ui-tui" / "scripts" / "gateway_bootstrap.py"

_FORWARDED = 'auth:v1:["basic","","david"]'


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("ink_gateway_bootstrap_subject_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_discovery(monkeypatch, module, endpoint, seen):
    receipt = SimpleNamespace(state="ready", endpoint=endpoint, reason_code=None)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: Path("/tmp/any-home"))
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda home, timeout=5: receipt)

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def readline(self, _limit):
            return json.dumps({
                "ok": True, "id": 1,
                "result": {"instance_id": endpoint.instance_id,
                           "profile_id": endpoint.profile_id, "ticket": "grant-1"},
            }).encode()

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def sendall(self, request):
            seen["request"] = json.loads(request)

        def makefile(self, _mode):
            return FakeStream()

    import hermes_cli.gateway_runtime_discovery as discovery
    monkeypatch.setattr(discovery, "connect_private", lambda home, timeout: FakeSocket())


def _endpoint(tmp_path):
    return SimpleNamespace(
        profile_id=str(tmp_path), instance_id="inst-1",
        api_origin="http://127.0.0.1:1234", runtime_protocol=1,
        control_home=str(tmp_path))


def test_the_forwarded_login_is_named_on_the_ticket_request(monkeypatch, tmp_path):
    module = _load_bootstrap()
    endpoint = _endpoint(tmp_path)
    seen = {}
    _fake_discovery(monkeypatch, module, endpoint, seen)
    monkeypatch.setenv("HERMES_TUI_SESSION_SUBJECT", _FORWARDED)

    module.bootstrap(False)

    assert seen["request"]["params"]["subject"] == _FORWARDED


def test_without_the_variable_the_request_stays_anonymous(monkeypatch, tmp_path):
    module = _load_bootstrap()
    endpoint = _endpoint(tmp_path)
    seen = {}
    _fake_discovery(monkeypatch, module, endpoint, seen)
    monkeypatch.delenv("HERMES_TUI_SESSION_SUBJECT", raising=False)

    module.bootstrap(False)

    assert seen["request"]["params"] == {
        "profile_id": endpoint.profile_id,
        "instance_id": endpoint.instance_id,
        "purpose": "interactive",
    }


def test_the_named_subject_survives_the_mint_as_the_resume_principal(tmp_path):
    """End to end through the trust boundary: the same-user gate accepts the
    forwarded login, and the redeemed grant's subject is what
    ``validate_legacy_row`` compares the serve-stamped row's owner against."""
    from gateway.control_socket import GatewayControlServer
    from gateway.runtime_bootstrap import TicketStore
    from hermes_state import SessionDB
    from hermes_state_local_migration import validate_legacy_row

    server = GatewayControlServer(tmp_path)
    server.ticket_store = TicketStore("inst-1", frozenset({str(tmp_path)}))
    response = json.loads(server.handle_request_line(json.dumps(
        {"protocol": 1, "id": 1, "verb": "session-ticket",
         "params": {"profile_id": str(tmp_path), "instance_id": "inst-1",
                    "purpose": "interactive", "subject": _FORWARDED}}).encode(),
        peer_subject="uid:501"))
    assert response["ok"] is True
    grant = server.ticket_store.redeem(response["result"]["ticket"],
                                       profile_id=None, purpose="interactive")
    assert grant["subject"] == _FORWARDED

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("serve-chat", source="desktop", user_id="basic:david")
        validate_legacy_row(db, db.get_session("serve-chat"), subject=grant["subject"])
    finally:
        db.close()
