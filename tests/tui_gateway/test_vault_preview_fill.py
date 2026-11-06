"""vault.preview.fill: the Desktop Browser bar's key button fills a saved login into one pinned in-app
browser tab. The renderer is faked at the server-request boundary (``server_requests.send``); the vault,
classifier, origin guard and fill script are real. The password reaches only the page script, never the
RPC result."""

from __future__ import annotations

import json

import pytest

import tui_gateway.server as srv

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
    from tui_gateway import server_requests

    state = {"origin": ORIGIN, "calls": []}

    def send(method, sid, params, *, timeout):
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
