"""Headless turns keep the real scoped vault and never require a Desktop bridge.

Only the external browser endpoint is substituted. Registry dispatch, gateway
profile scope, encrypted storage, classification and secret redaction are real.
"""

import json

import pytest

from agent import secret_scope
from agent.redact import clear_vault_redaction_values
from agent.vault_store import get_vault_store
from gateway.run import _profile_runtime_scope
from tools import browser_vault_tool
from tools.registry import registry


class BrowserEndpoint:
    def __init__(self, origin="https://login.example.test"):
        self.origin = origin
        self.expressions = []
        self.inspected = False

    def focus_page(self, origin, *, accept):
        return {"ok": not origin or origin == self.origin, "url": self.origin + "/login"}

    def evaluate_runtime(self, expression):
        self.expressions.append(expression)
        if expression == "window.location.href":
            return {"ok": True, "result": self.origin + "/login"}
        if not self.inspected:
            self.inspected = True
            return {"ok": True, "result": [{
                "index": 0, "formIndex": 0, "type": "password",
                "autocomplete": "current-password", "name": "password",
            }]}
        return {"ok": True, "result": {"filled": 1}}


@pytest.fixture
def headless_homes(tmp_path, monkeypatch):
    from agent.vault_backends import unlock

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setattr(unlock, "can_prompt_here", lambda: False)
    homes = {}
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            "vault:\n  onepassword:\n    enabled: false\n"
            "  bitwarden:\n    enabled: false\n", encoding="utf-8")
        homes[name] = home
    try:
        yield homes
    finally:
        clear_vault_redaction_values()
        secret_scope.set_multiplex_active(previous)


def _endpoint(monkeypatch, endpoint):
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get", lambda _task: endpoint)
    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", lambda _task: endpoint)


def _login(password):
    return get_vault_store().add_item(
        "login", "Synthetic headless login",
        {"identifier": "demo", "identifier_type": "username", "password": password},
        origin="https://login.example.test")


def test_gateway_headless_fill_uses_each_encrypted_home_through_a_b_a(headless_homes, monkeypatch):
    handles = {}
    for name in ("a", "b", "a"):
        password = f"synthetic-headless-{name}-secret"
        endpoint = BrowserEndpoint()
        _endpoint(monkeypatch, endpoint)
        with _profile_runtime_scope(headless_homes[name]):
            if name not in handles:
                handles[name] = _login(password).id
            # No preview callback or explicit target: the headless default is
            # the supervised browser, even when another profile used Desktop.
            raw = registry.dispatch("browser_vault_fill", {"handle": handles[name]}, task_id=f"task-{name}")
            result = json.loads(raw)
            assert result["success"] is True, result
            assert result["filled_fields"] == 1
            assert json.dumps(password) in endpoint.expressions[-1]
            other_password = f"synthetic-headless-{'b' if name == 'a' else 'a'}-secret"
            assert other_password not in "".join(endpoint.expressions)
            assert password not in raw
            assert [m.id for m in get_vault_store().list_items()] == [handles[name]]


def test_headless_wrong_origin_refuses_before_secret_resolution(headless_homes, monkeypatch):
    from agent.vault_backends.local import LocalLoginBackend

    endpoint = BrowserEndpoint("https://wrong.example.test")
    _endpoint(monkeypatch, endpoint)

    def forbidden(*_args, **_kwargs):
        pytest.fail("A wrong-origin page caused a secret read")

    monkeypatch.setattr(LocalLoginBackend, "resolve_password", forbidden)
    with _profile_runtime_scope(headless_homes["a"]):
        item = _login("synthetic-origin-secret")
        result = json.loads(registry.dispatch("browser_vault_fill", {"handle": item.id}, task_id="headless"))
    assert result["error_type"] == "origin_mismatch"
    assert endpoint.expressions == ["window.location.href"]


def test_headless_save_login_refuses_unavailable_user_prompt(headless_homes, monkeypatch):
    from agent.vault_backends import unlock

    endpoint = BrowserEndpoint()
    _endpoint(monkeypatch, endpoint)

    def forbidden(*_args, **_kwargs):
        pytest.fail("A headless turn tried to open a login prompt")

    monkeypatch.setattr(unlock, "get_save_login_prompt_callback", lambda: forbidden)
    with _profile_runtime_scope(headless_homes["a"]):
        result = json.loads(registry.dispatch("browser_vault_save_login", {}, task_id="headless"))
        assert get_vault_store().list_items() == []
    assert result["error_type"] == "prompt_unavailable"
    assert endpoint.expressions == ["window.location.href"]


def test_missing_supervisor_never_sends_secret_through_cli(headless_homes, monkeypatch):
    from tools import browser_tool_session

    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", lambda _task: None)
    calls = []
    monkeypatch.setattr(browser_tool_session, "_run_browser_command", lambda *args, **kwargs: calls.append(args))
    with _profile_runtime_scope(headless_homes["a"]):
        result = browser_vault_tool._eval_js_secret("headless", '"synthetic-not-an-argv-value"')
    assert result["error_type"] == "supervisor_required"
    assert calls == []
