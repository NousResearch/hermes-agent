"""vault.preview.fill: the Desktop Browser bar's key button fills a saved login into one pinned in-app
browser tab. The renderer is faked at the server-request boundary (``server_requests.send``); the vault,
classifier, origin guard and fill script are real. The password reaches only the page script, never the
RPC result."""

from __future__ import annotations

import json
import threading
import types

import pytest

import hermes_agent.tui_gateway.server as srv

PASSWORD = "synthetic-key-button-pw"
ORIGIN = "https://login.example.test"
CONTROLS = [
    {"index": 0, "formIndex": 0, "type": "email", "autocomplete": "username", "name": "user", "label": ""},
    {"index": 1, "formIndex": 0, "type": "password", "autocomplete": "current-password", "name": "pw", "label": ""},
]


@pytest.fixture
def login(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({"vault": {"onepassword": {"enabled": False},
                                                             "bitwarden": {"enabled": False}}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    out = srv._methods["vault.add"](1, {"kind": "login", "label": "Example", "origin": ORIGIN, "secret": {
        "identifier_type": "email", "identifier": "me@example.test", "password": PASSWORD}})
    return out["result"]["id"]


@pytest.fixture
def window(monkeypatch):
    """The renderer answering ``preview.act`` vault requests for its pinned tab."""
    from hermes_agent.tui_gateway import server_requests

    state = {"origin": ORIGIN, "calls": []}

    def send(method, sid, params, *, timeout, transport=None):
        assert method == "preview.act" and sid == "" and params["action"] == "vault"
        req = params["vault"]
        state["calls"].append(req)
        if req.get("target") != "pinned-by-renderer":
            return {"value": json.dumps({"success": False, "error": "unknown target"})}
        if req["operation"] == "close":
            return {"value": json.dumps({"success": True})}
        expr = req["expression"]
        if "location.href" in expr:
            result = state["origin"] + "/signin"
        elif "data-hermes-vault-slot\", nonce" in expr:
            result = json.dumps(CONTROLS)
        else:
            state["fill"] = expr
            result = json.dumps({"filled": sum(json.dumps(v) in expr for v in (PASSWORD, "me@example.test"))})
        return {"value": json.dumps({"success": True, "result": result})}

    monkeypatch.setattr(server_requests, "send", send)
    return state


def test_key_button_fills_identifier_and_password_into_the_pinned_tab(login, window):
    out = srv._methods["vault.preview.fill"](7, {"handle": login, "target": "pinned-by-renderer"})
    result = out["result"]
    assert result == {"success": True, "filled_fields": 2, "origin": ORIGIN}
    assert PASSWORD not in json.dumps(out)
    assert json.dumps(PASSWORD) in window["fill"] and "me@example.test" in window["fill"]
    assert [c["operation"] for c in window["calls"]][-1] == "close"


def test_key_button_refuses_another_origin_before_resolving_the_password(login, window, monkeypatch):
    from agent.vault_backends.local import LocalLoginBackend

    def never(*_a, **_k):
        raise AssertionError("password resolved for a page on another origin")

    monkeypatch.setattr(LocalLoginBackend, "resolve_password", never)
    window["origin"] = "https://phish.example.test"
    out = srv._methods["vault.preview.fill"](8, {"handle": login, "target": "pinned-by-renderer"})
    assert out["result"]["success"] is False and out["result"]["error_type"] == "origin_mismatch"
    assert "fill" not in window


@pytest.mark.parametrize("relayed", [False, True])
def test_preview_vault_request_is_live_only_and_still_resolves(monkeypatch, relayed):
    from hermes_agent.tui_gateway import server_requests

    server = srv

    sid = "preview-vault-live-only"
    session = {"history_lock": threading.Lock()}
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server, "write_json", lambda _frame: True)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: relayed)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda: types.SimpleNamespace(
        respond=lambda _sid, params: server_requests.resolve_response(params["frame"])))
    expression = '(() => "synthetic-secret-never-replayed")()'

    def deliver(frame):
        assert frame["params"]["vault"]["expression"] == expression
        assert server_requests._open[frame["id"]].params == {}
        if relayed:
            assert server._relay_compute_host_rpc(frame)
            assert expression not in json.dumps(session["_compute_host_open_request"])
        assert server._open_requests(sid) == []
        answer = {"id": frame["id"], "result": {"value": '{"success": true}'}}
        resolve = server._relay_compute_host_response if relayed else server_requests.resolve_response
        assert resolve(answer)

    monkeypatch.setattr(server_requests, "_write", deliver)
    try:
        result = server._ask("preview.act", sid, {
            "action": "vault", "vault": {"operation": "evaluate", "target": "bound-page", "expression": expression},
        }, timeout=0)
    finally:
        server._sessions.pop(sid, None)
    assert json.loads(result) == {"success": True}
    assert server._open_requests(sid) == []
    assert server_requests.open_request_count() == 0


def _forward_preview_read(monkeypatch, method, echoed_value):
    """Drive one child preview reply through the parent's compute-host response relay and return the
    value the supervisor was handed for the child."""
    server = srv
    sid = "preview-read-isolation"
    session = {"history_lock": threading.Lock(), "profile_home": "",
               "_compute_host_open_request": {"id": "srq-abc", "method": method}}
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    captured = {}
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda: types.SimpleNamespace(
        respond=lambda _sid, params: captured.__setitem__("frame", params["frame"])))
    assert server._relay_compute_host_response({"id": "srq-abc", "result": {"value": echoed_value}})
    return captured["frame"]["result"]["value"]


@pytest.mark.parametrize("method", ["preview.read", "preview.act"])
def test_isolated_preview_read_is_redacted_in_the_owning_profile_before_the_child(monkeypatch, method):
    """#150283 P1-5: a UI key-button fill registers the password in the PARENT's redaction registry; a
    compute-host child's separate registry has no record, so the parent must scrub a forwarded preview
    reply that echoes the filled value. The echo can land in text, a nested label, or a diagnostic field."""
    from agent.redact import clear_vault_redaction_values, register_vault_redaction_value

    secret = "synthetic-isolated-fill-pw-150283"
    register_vault_redaction_value(secret)
    try:
        echoed = json.dumps({"success": True, "text": secret, "elements": [{"label": secret}],
                             "error": f"login failed for {secret}"})
        forwarded = _forward_preview_read(monkeypatch, method, echoed)
    finally:
        clear_vault_redaction_values()
    srv._sessions.pop("preview-read-isolation", None)
    assert secret not in forwarded and "redacted-vault-secret" in forwarded


def test_non_preview_relayed_reply_is_forwarded_untouched(monkeypatch):
    """The scrub is scoped to preview replies: an unrelated relayed response passes through verbatim
    (mutant guard — the fix must not blanket-redact every forwarded frame)."""
    from agent.redact import clear_vault_redaction_values, register_vault_redaction_value

    secret = "synthetic-isolated-fill-pw-150283"
    register_vault_redaction_value(secret)
    try:
        value = json.dumps({"ok": True, "note": secret})  # a clarify/other reply, not a preview read
        forwarded = _forward_preview_read(monkeypatch, "clarify", value)
    finally:
        clear_vault_redaction_values()
    srv._sessions.pop("preview-read-isolation", None)
    assert forwarded == value
