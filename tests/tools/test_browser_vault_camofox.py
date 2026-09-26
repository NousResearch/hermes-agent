"""Vault transport contract against an isolated, synthetic Camofox HTTP endpoint."""

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from agent.vault_store import get_vault_store
from tools import browser_camofox as camofox
from tools import browser_vault_tool as vault
from tools.registry import registry


CANARY = "synthetic vault password -- never a real credential"
ORIGIN = "https://login.example.test"
TASK = "raw-task-not-a-cdp-session"


@pytest.fixture
def remote(monkeypatch):
    calls = []
    controls = [{"index": 0, "type": "password", "autocomplete": "current-password"}]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"snapshot": f"page text: {CANARY}", "refsCount": 0}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, dict(self.headers), body))
            if self.path.endswith("/navigate"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "url": ORIGIN, "title": CANARY}).encode())
                return
            expression = body["expression"]
            if "const fills =" in expression and session.get("redirect") and self.path != "/stolen":
                self.send_response(307)
                self.send_header("Location", "/stolen")
                self.end_headers()
                return
            result = (session.get("page_url", ORIGIN + "/login") if expression == "window.location.href" else
                      session.get("fill_result", json.dumps({"filled": 1})) if "const fills =" in expression else
                      json.dumps(session.get("controls", controls)))
            if "const fills =" in expression and session.get("execute"):
                result = session["execute"](expression)
            if expression == "document.querySelector('input').value":
                result = CANARY
            self.send_response(session.get("status", 200))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(session.get("raw_response", json.dumps({"result": result})).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("CAMOFOX_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("CAMOFOX_API_KEY", "synthetic-camofox-token")
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(camofox, "_sessions", {})
    session = camofox._get_session(TASK)
    session["tab_id"] = "existing-tab"
    no_local = Mock(return_value={"success": False})
    monkeypatch.setattr("tools.browser_tool_session._run_browser_command", no_local)
    no_supervisor = Mock(return_value=None)
    monkeypatch.setattr(vault, "_ensure_supervisor", no_supervisor)
    try:
        yield calls, session, no_local, no_supervisor
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        from agent.redact import clear_vault_redaction_values
        clear_vault_redaction_values()


def add_login(password=CANARY):
    return get_vault_store().add_item(
        "login", "Synthetic login", {"identifier": "test-user", "identifier_type": "username", "password": password},
        origin=ORIGIN,
    )


def test_fill_uses_authenticated_camofox_tab_not_local_browser(remote):
    calls, session, no_local, no_supervisor = remote
    meta = add_login()
    entry = registry.get_entry("browser_vault_fill")
    assert entry is not None
    raw = entry.handler({"handle": meta.id}, task_id=TASK)
    out = json.loads(raw)
    assert out["success"] is True, out
    assert out["filled_fields"] == 1
    assert CANARY not in raw
    assert all(path == "/tabs/existing-tab/evaluate" for path, _, _ in calls)
    assert all(headers["Authorization"] == "Bearer synthetic-camofox-token" for _, headers, _ in calls)
    assert all(body["userId"] == session["user_id"] for _, _, body in calls)
    assert CANARY in calls[-1][2]["expression"]
    no_local.assert_not_called()
    no_supervisor.assert_not_called()


def test_fill_pins_tab_endpoint_and_auth_for_the_whole_operation(remote, monkeypatch):
    calls, session, no_local, no_supervisor = remote
    meta = add_login()
    from agent.vault_backends.local import LocalLoginBackend
    original = LocalLoginBackend.resolve_password

    def resolve(self, handle):
        session["tab_id"] = "replacement-tab"
        monkeypatch.setenv("CAMOFOX_API_KEY", "replacement-token")
        monkeypatch.setenv("CAMOFOX_URL", camofox.get_camofox_url() + "/replacement-server")
        return original(self, handle)

    monkeypatch.setattr(LocalLoginBackend, "resolve_password", resolve)
    out = json.loads(vault.browser_vault_fill(meta.id, task_id=TASK))
    assert out["success"] is True, out
    assert all(path == "/tabs/existing-tab/evaluate" for path, _, _ in calls)
    assert all(headers["Authorization"] == "Bearer synthetic-camofox-token" for _, headers, _ in calls)
    no_local.assert_not_called()
    no_supervisor.assert_not_called()


@pytest.mark.parametrize("url", [
    "http://remote.example.test:9377", "http://192.168.1.20:9377",
    "http://localhost:9377", "http://127.0.0.1.example.test:9377", "http://127.1:9377",
    "http://2130706433:9377", "http://[::ffff:127.0.0.1]:9377",
    "https://user:password@remote.example.test", "https://remote.example.test#fragment",
    "https://remote.example.test?query=value", "https://remote.example.test#", "https://remote.example.test?",
    "ftp://127.0.0.1:9377", "",
])
def test_unsafe_transport_refuses_before_any_request(remote, monkeypatch, url):
    monkeypatch.setenv("CAMOFOX_URL", url)
    monkeypatch.setattr(camofox, "is_camofox_mode", lambda: True)
    post = Mock()
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("requests.Session.post", post)
    out = json.loads(vault.browser_vault_fill(add_login().id, task_id=TASK))
    assert out["success"] is False
    assert out.get("error_type") == "camofox_transport_required"
    post.assert_not_called()


@pytest.mark.parametrize("url", ["https://camofox.example.test/api", "http://127.0.0.1:9377", "http://[::1]:9377"])
def test_secure_transport_urls_are_accepted(url):
    from tools.browser_vault_camofox import _secure_url
    assert _secure_url(url)


def test_secret_request_never_follows_redirects(remote):
    calls, session, _, _ = remote
    session["redirect"] = True
    out = json.loads(vault.browser_vault_fill(add_login().id, task_id=TASK))
    assert not any(path == "/stolen" for path, _, _ in calls)
    assert out["success"] is False


def test_vault_transport_ignores_ambient_proxy_settings(remote, monkeypatch):
    import requests
    original = requests.Session.send
    observed = []

    def send(self, request, **kwargs):
        observed.append(kwargs.get("proxies"))
        kwargs["proxies"] = {}  # keep the test request on its isolated server
        return original(self, request, **kwargs)

    monkeypatch.setenv("HTTP_PROXY", "http://untrusted-proxy.test:8080")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(requests.Session, "send", send)
    out = json.loads(vault.browser_vault_fill(add_login().id, task_id=TASK))
    assert out["success"] is True
    assert observed and all(not proxies for proxies in observed)


@pytest.mark.parametrize("result", [
    {"filled": CANARY}, {"filled": True}, {"filled": 2}, {"filled": -1}, {"filled": 1.5},
    {"filled": 1, "echo": CANARY}, [CANARY], CANARY,
    {"refused": "origin_changed", "found": CANARY}, {"refused": CANARY},
])
def test_secret_response_is_allowlisted_not_echoed(remote, result):
    _, session, _, _ = remote
    session["fill_result"] = json.dumps(result)
    raw = vault.browser_vault_fill(add_login().id, task_id=TASK)
    assert CANARY not in raw
    out = json.loads(raw)
    assert out["success"] is False
    if isinstance(result, dict) and result.get("refused") == "origin_changed":
        assert out["error_type"] == "origin_changed"


def test_secret_js_catches_page_exceptions_before_the_server_can_log_them(remote):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the actual JS expression")
    _, session, _, _ = remote
    executions = []

    def execute(expression):
        program = """
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
global.window = {location: {get origin() { throw new Error(input.canary); }}};
(async () => {
  try { console.log(JSON.stringify({result: await eval(input.expression), threw: false})); }
  catch(e) { console.log(JSON.stringify({threw: true})); }
})();
"""
        completed = subprocess.run(
            [node, "-e", program], input=json.dumps({"expression": expression, "canary": CANARY}),
            text=True, capture_output=True, check=True, timeout=10,
        )
        value = json.loads(completed.stdout)
        executions.append(value)
        return value.get("result")

    session["execute"] = execute
    out = json.loads(vault.browser_vault_fill(add_login().id, task_id=TASK))
    assert executions and executions[0]["threw"] is False
    assert CANARY not in json.dumps(executions)
    assert out["success"] is False


@pytest.mark.parametrize("change", ["none", "nonce", "origin"])
def test_transported_js_preserves_origin_nonce_and_literal_values(remote, change):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the actual JS expressions")
    calls, session, _, _ = remote
    password = 'synthetic "quoted";\\value\nwith unicode \u2028 and )}; //'
    executions = []

    def execute(expression):
        program = """
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
global.HTMLInputElement = class {
  constructor() { this.attrs = {}; this.type = 'password'; this.tagName = 'INPUT'; this._value = ''; }
  set value(v) { this._value = v; } get value() { return this._value; }
  setAttribute(k,v) { this.attrs[k]=v; } getAttribute(k) { return this.attrs[k] || ''; }
  removeAttribute(k) { delete this.attrs[k]; }
  getClientRects() { return [1]; } focus() {} dispatchEvent() {}
};
const el = new HTMLInputElement();
global.window = {location: {origin: input.origin}};
global.document = {
  forms: [], getElementById() { return null; }, querySelectorAll() { return [el]; },
  querySelector(selector) {
    return selector === '[data-hermes-vault-slot="' + el.attrs['data-hermes-vault-slot'] + '"]' ? el : null;
  }
};
global.getComputedStyle = () => ({display: 'block', visibility: 'visible'});
global.InputEvent = class {}; global.Event = class {};
(async () => {
  eval(input.inspection);
  if (input.change === 'nonce') el.attrs['data-hermes-vault-slot'] = 'different:0';
  if (input.change === 'origin') window.location.origin = 'https://other.example.test';
  const result = await eval(input.expression);
  console.log(JSON.stringify({result, matched: el.value === input.password, written: el.value.length > 0}));
})();
"""
        completed = subprocess.run(
            [node, "-e", program], text=True, capture_output=True, check=True, timeout=10,
            input=json.dumps({"expression": expression, "inspection": calls[-2][2]["expression"],
                              "origin": ORIGIN, "password": password, "change": change}),
        )
        value = json.loads(completed.stdout)
        executions.append(value)
        return value["result"]

    session["execute"] = execute
    raw = vault.browser_vault_fill(add_login(password).id, task_id=TASK)
    out = json.loads(raw)
    assert out["success"] is (change == "none"), out
    assert executions[0]["written"] is (change == "none")
    assert executions[0]["matched"] is (change == "none")
    assert password not in raw
    if change == "origin":
        assert out["error_type"] == "origin_changed"


def test_same_raw_task_is_isolated_across_profile_scopes(remote, tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    calls, _, _, _ = remote
    url = camofox.get_camofox_url()
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    sessions = {}
    metas = {}
    for profile in ("a", "b", "a"):
        home_token = set_hermes_home_override(tmp_path / profile)
        secret_token = secret_scope.set_secret_scope({
            "CAMOFOX_URL": url, "CAMOFOX_API_KEY": f"synthetic-token-{profile}",
        })
        try:
            if profile not in sessions:
                session = camofox._get_session(TASK)
                assert all(session is not other for other in sessions.values())
                session["tab_id"] = f"tab-{profile}"
                sessions[profile] = session
                metas[profile] = add_login()
            assert get_vault_store().list_items()[0].id == metas[profile].id
            before = len(calls)
            out = json.loads(vault.browser_vault_fill(metas[profile].id, task_id=TASK))
            assert out["success"] is True, out
            assert all(path == f"/tabs/tab-{profile}/evaluate" for path, _, _ in calls[before:])
            assert all(headers["Authorization"] == f"Bearer synthetic-token-{profile}"
                       for _, headers, _ in calls[before:])
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)


def test_changing_server_does_not_reuse_a_tab_from_the_old_server(remote, monkeypatch):
    calls, _, no_local, no_supervisor = remote
    monkeypatch.setenv("CAMOFOX_URL", camofox.get_camofox_url() + "/other-server")
    out = json.loads(vault.browser_vault_fill(add_login().id, task_id=TASK))
    assert out["success"] is False
    assert not calls
    no_local.assert_not_called()
    no_supervisor.assert_not_called()


@pytest.mark.parametrize("name", ["fill", "save_login", "enter_code"])
def test_vault_operations_require_camofox_authentication(remote, monkeypatch, name):
    calls, _, _, _ = remote
    monkeypatch.delenv("CAMOFOX_API_KEY")
    out = json.loads(getattr(vault, "browser_vault_" + name)(add_login().id, task_id=TASK))
    assert out.get("error_type") == "camofox_auth_required"
    assert not calls


def test_redactor_is_registered_before_fill_and_public_console_cannot_echo(remote, monkeypatch):
    import requests
    from agent.redact import redact_sensitive_text
    from model_tools import handle_function_call

    original = requests.Session.send
    registered_before_send = []

    def send(self, request, **kwargs):
        if "const fills =" in str(request.body):
            registered_before_send.append(CANARY not in redact_sensitive_text(CANARY, force=True))
        return original(self, request, **kwargs)

    monkeypatch.setattr(requests.Session, "send", send)
    out = json.loads(handle_function_call("browser_vault_fill", {"handle": add_login().id}, task_id=TASK))
    assert out["success"] is True
    assert registered_before_send == [True]
    raw = handle_function_call("browser_console", {"expression": "document.querySelector('input').value"}, task_id=TASK)
    assert CANARY not in raw
    readback = json.loads(raw)
    assert readback["success"] is True, readback
    assert "redacted-vault-secret" in readback["result"]


def test_public_snapshot_cannot_echo_a_filled_secret(remote):
    from model_tools import handle_function_call

    out = json.loads(handle_function_call("browser_vault_fill", {"handle": add_login().id}, task_id=TASK))
    assert out["success"] is True
    raw = handle_function_call("browser_snapshot", {}, task_id=TASK)
    assert json.loads(raw)["success"] is True
    assert CANARY not in raw


def test_navigation_metadata_cannot_echo_a_filled_secret(remote):
    from model_tools import handle_function_call

    out = json.loads(handle_function_call("browser_vault_fill", {"handle": add_login().id}, task_id=TASK))
    assert out["success"] is True
    raw = handle_function_call("browser_navigate", {"url": "https://example.com"}, task_id=TASK)
    assert json.loads(raw)["success"] is True, raw
    assert CANARY not in raw


def test_console_errors_cannot_echo_a_filled_secret(remote, monkeypatch):
    from model_tools import handle_function_call

    out = json.loads(handle_function_call("browser_vault_fill", {"handle": add_login().id}, task_id=TASK))
    assert out["success"] is True
    monkeypatch.setattr(camofox, "_post", Mock(side_effect=RuntimeError(CANARY)))
    raw = handle_function_call("browser_console", {"expression": "document.title"}, task_id=TASK)
    assert json.loads(raw)["success"] is False
    assert CANARY not in raw


def test_saved_otp_is_not_resolved_for_another_origin(remote, monkeypatch):
    from agent.vault_backends.local import LocalLoginBackend

    calls, session, _, _ = remote
    session["page_url"] = "https://other.example.test/otp"
    session["controls"] = [{"index": 0, "type": "text", "autocomplete": "one-time-code"}]
    resolve = Mock(return_value="246810")
    monkeypatch.setattr(LocalLoginBackend, "resolve_otp", resolve)
    out = json.loads(vault.browser_vault_enter_code(add_login().id, task_id=TASK))
    assert out.get("error_type") == "origin_mismatch"
    resolve.assert_not_called()
    assert all("const fills =" not in body["expression"] for _, _, body in calls)


def test_save_login_reuses_the_prompted_tab_for_its_nested_fill(remote, monkeypatch):
    from agent.vault_backends import unlock

    calls, session, _, _ = remote

    def prompt(origin, site):
        assert origin == ORIGIN
        session["tab_id"] = "different-tab-during-prompt"
        return {"identifier": "synthetic-user", "password": CANARY}

    monkeypatch.setattr(unlock, "can_prompt_here", lambda: True)
    monkeypatch.setattr(unlock, "get_save_login_prompt_callback", lambda: prompt)
    raw = vault.browser_vault_save_login(task_id=TASK)
    out = json.loads(raw)
    assert out["success"] is True and out["fill"]["success"] is True
    assert CANARY not in raw
    [meta] = get_vault_store().list_items()
    assert meta.origin == ORIGIN
    assert all(path == "/tabs/existing-tab/evaluate" for path, _, _ in calls)


def test_user_otp_uses_pinned_transport_and_split_field_count(remote, monkeypatch):
    from agent.vault_backends import unlock

    calls, session, _, _ = remote
    session["controls"] = [{"index": i, "type": "tel", "autocomplete": "one-time-code", "formIndex": 0,
                            "maxLength": 1} for i in range(6)]
    session["fill_result"] = {"filled": 6}

    def prompt(site, hint):
        session["tab_id"] = "different-tab-during-prompt"
        return "246810"

    monkeypatch.setattr(unlock, "can_prompt_here", lambda: True)
    monkeypatch.setattr(unlock, "get_code_prompt_callback", lambda: prompt)
    raw = vault.browser_vault_enter_code(task_id=TASK)
    out = json.loads(raw)
    assert out["success"] is True and out["filled_fields"] == 6
    assert "246810" not in raw
    assert all(path == "/tabs/existing-tab/evaluate" for path, _, _ in calls)


def test_payment_approval_is_required_before_remote_fill(remote, monkeypatch):
    calls, session, _, _ = remote
    session["controls"] = [{"index": 0, "type": "text", "autocomplete": "cc-number"},
                            {"index": 1, "type": "text", "autocomplete": "cc-csc"}]
    session["fill_result"] = {"filled": 2}
    card = {"card_number": "4111111111111111", "cardholder_name": "Test User", "exp_month": "7",
            "exp_year": "2029", "cvc": "123", "billing_postal_code": "94110"}
    meta = get_vault_store().add_item("payment", "Synthetic card", card, origin=ORIGIN)
    monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent", lambda *a, **k: "decline")
    out = json.loads(vault.browser_vault_fill(meta.id, task_id=TASK))
    assert out["error_type"] == "payment_declined" and not calls
    monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent", lambda *a, **k: "accept")
    raw = vault.browser_vault_fill(meta.id, task_id=TASK)
    assert json.loads(raw)["filled_fields"] == 2
    assert card["card_number"] not in raw and card["cvc"] not in raw


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 404, 405, 500, 501])
def test_endpoint_errors_do_not_fallback_or_echo(remote, status, caplog):
    calls, session, no_local, no_supervisor = remote
    session["status"] = status
    session["raw_response"] = CANARY
    raw = vault.browser_vault_fill(add_login().id, task_id=TASK)
    assert json.loads(raw)["success"] is False
    assert CANARY not in raw + caplog.text
    assert len(calls) == 1
    no_local.assert_not_called()
    no_supervisor.assert_not_called()


def test_secret_transport_exception_is_static_and_not_logged(remote, monkeypatch, caplog):
    import requests

    original = requests.Session.post

    def post(self, url, **kwargs):
        if "const fills =" in kwargs["json"]["expression"]:
            raise requests.ConnectionError(CANARY)
        return original(self, url, **kwargs)

    monkeypatch.setattr(requests.Session, "post", post)
    raw = vault.browser_vault_fill(add_login().id, task_id=TASK)
    assert json.loads(raw)["success"] is False
    assert CANARY not in raw + caplog.text


@pytest.mark.parametrize("name", ["fill", "save_login", "enter_code"])
def test_missing_tab_never_creates_or_adopts_one(remote, monkeypatch, name):
    calls, _, no_local, no_supervisor = remote
    camofox._drop_session(TASK)
    ensure = Mock(side_effect=AssertionError("must not create or adopt a tab"))
    monkeypatch.setattr(camofox, "_ensure_tab", ensure)
    out = json.loads(getattr(vault, "browser_vault_" + name)(add_login().id, task_id=TASK))
    assert out["success"] is False
    assert not calls
    ensure.assert_not_called()
    no_local.assert_not_called()
    no_supervisor.assert_not_called()
