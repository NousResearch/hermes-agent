"""Live Chromium coverage for Shadow-DOM-aware vault discovery and filling.

The fixture starts a fresh, headless Chromium profile and a loopback HTTP server.
It deliberately uses synthetic values only; the credential-bearing fill expression
is evaluated through the Hermes CDP supervisor, never through a subprocess.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from agent.vault_login_classifier import build_fill_js, build_inspection_js
from agent.vault_store import VaultStore
from tools import browser_vault_tool as vault
from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.browser_use_cli import _attach_vault_supervisor
from tools.registry import registry


_DUMMY_PASSWORD = "dummy-live-value"


@pytest.fixture
def browser(tmp_path, request):
    executable = next(
        (shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome") if shutil.which(name)),
        None,
    )
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if executable is None and mac.exists():
        executable = str(mac)
    if executable is None:
        pytest.skip("Chrome/Chromium is required for the live Shadow-DOM regression")
    assert executable is not None

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib HTTPServer hook
            body = b"<!doctype html><title>vault shadow regression</title><body></body>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    profile = tmp_path / "chromium-profile"
    process = subprocess.Popen(
        [
            executable,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            f"{origin}/unrelated",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    task = f"vault-shadow-{request.node.name}"
    try:
        active_port = profile / "DevToolsActivePort"
        address = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and process.poll() is None:
            if active_port.exists():
                address = active_port.read_text(encoding="utf-8").splitlines()
                if len(address) >= 2 and address[0].isdigit() and address[1].startswith("/devtools/browser/"):
                    break
            time.sleep(0.05)
        if len(address) < 2:
            pytest.fail("isolated Chromium did not publish its DevTools address")

        port, ws_path = address[:2]
        _attach_vault_supervisor({"BU_CDP_WS": f"ws://127.0.0.1:{port}{ws_path}"}, task)
        supervisor = SUPERVISOR_REGISTRY.get(task)
        if supervisor is None:
            pytest.fail("browser_exec must attach the vault supervisor")
        assert supervisor is not None

        def wait_for_page(url: str) -> None:
            page_origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
            wait_deadline = time.monotonic() + 20
            while time.monotonic() < wait_deadline:
                focused = supervisor.focus_page(page_origin)
                if focused.get("ok"):
                    ready = supervisor.evaluate_runtime(
                        f"location.href === {json.dumps(url)} && document.readyState === 'complete'"
                    )
                    if ready.get("ok") and ready.get("result"):
                        return
                time.sleep(0.05)
            pytest.fail(f"test page did not finish loading: {url}")

        wait_for_page(f"{origin}/unrelated")

        def new_page(url: str) -> None:
            request_url = f"http://127.0.0.1:{port}/json/new?{url}"
            request = urllib.request.Request(request_url, method="PUT")
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
            wait_for_page(url)

        yield supervisor, task, origin, new_page
    finally:
        SUPERVISOR_REGISTRY.stop(task)
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def evaluate(supervisor, expression):
    result = supervisor.evaluate_runtime(expression)
    assert result["ok"], result
    return result["result"]


def create_nested_shadow_form(supervisor):
    evaluate(
        supervisor,
        """(() => {
          const host = document.createElement('div');
          document.body.append(host);
          const outer = host.attachShadow({mode: 'open'});
          outer.innerHTML = '<div id="nested"></div>';
          const nested = outer.querySelector('#nested');
          const inner = nested.attachShadow({mode: 'open'});
          inner.innerHTML = '<form><label id="pw-label" for="pw">Password</label>' +
            '<input id="pw" type="password" aria-labelledby="pw-label" autocomplete="current-password">' +
            '<input autocomplete="cc-number"><input autocomplete="postal-code">' +
            '<input autocomplete="one-time-code"></form>';
          // Expose the same root through a second synthetic host. queryAll must
          // deduplicate roots and preserve one stable stamp per control.
          const alias = document.createElement('div');
          Object.defineProperty(alias, 'shadowRoot', {get: () => outer});
          document.body.append(alias);
          window.testRoot = inner;
          window.testHost = host;
          window.testInputObservers = [];
          for (const [name, target] of [['host', host], ['document', document]]) {
            for (const eventName of ['input', 'change']) {
              target.addEventListener(eventName, event => {
                window.testInputObservers.push({
                  observer: name,
                  event: event.type,
                  composed: event.composed,
                  retargeted: event.target === host,
                  inputType: event.inputType || null,
                });
              });
            }
          }
        })()""",
    )


def test_shadow_probes_inspection_nonce_and_events_share_scope(browser):
    supervisor, _task, origin, new_page = browser
    login_origin = origin.replace("127.0.0.1", "localhost")
    new_page(f"{login_origin}/login")
    create_nested_shadow_form(supervisor)

    for kind in ("login", "payment", "address", "otp"):
        assert evaluate(supervisor, vault._TAB_PROBES[kind]) is True, kind

    controls = json.loads(evaluate(supervisor, build_inspection_js("test-nonce")))
    assert len(controls) == 4
    assert [control["index"] for control in controls] == list(range(4))
    password = next(control for control in controls if control["type"] == "password")
    assert "Password" in password["label"]
    assert password["formIndex"] == 0

    fills = [{"index": password["index"], "token": "current-password", "value": _DUMMY_PASSWORD}]
    refused = json.loads(evaluate(supervisor, build_fill_js(fills, login_origin + ".invalid", "test-nonce")))
    assert refused["refused"] == "origin_changed"
    assert evaluate(supervisor, "testRoot.querySelector('input').value === ''") is True

    stale = json.loads(evaluate(supervisor, build_fill_js(fills, login_origin, "stale-nonce")))
    assert stale["filled"] == 0
    assert evaluate(supervisor, "testRoot.querySelector('input').value === ''") is True

    # A fresh inspection re-establishes the nonce after the stale attempt's cleanup.
    evaluate(supervisor, build_inspection_js("test-nonce"))
    evaluate(supervisor, "window.testInputObservers = []")
    filled = json.loads(evaluate(supervisor, build_fill_js(fills, login_origin, "test-nonce")))
    assert filled == {"filled": 1}
    assert evaluate(supervisor, "testRoot.querySelector('input').value === 'dummy-live-value'") is True
    assert evaluate(supervisor, "testRoot.querySelectorAll('[data-hermes-vault-slot]').length") == 0
    assert evaluate(supervisor, "window.testInputObservers") == [
        {"observer": observer, "event": event, "composed": True, "retargeted": True, "inputType": input_type}
        for event, input_type in (("input", "insertText"), ("change", None))
        for observer in ("host", "document")
    ]

    # A password fill never follows a type mutation after inspection.
    controls = json.loads(evaluate(supervisor, build_inspection_js("type-nonce")))
    password_index = next(control["index"] for control in controls if control["type"] == "password")
    evaluate(supervisor, "testRoot.querySelector('input').type = 'text'; testRoot.querySelector('input').value = ''")
    type_changed = json.loads(
        evaluate(
            supervisor,
            build_fill_js(
                [{"index": password_index, "token": "current-password", "value": _DUMMY_PASSWORD}],
                login_origin,
                "type-nonce",
            ),
        ),
    )
    assert type_changed["filled"] == 0
    assert evaluate(supervisor, "testRoot.querySelector('input').value === ''") is True

    # Open-shadow traversal must not cross into an iframe document.
    evaluate(
        supervisor,
        """document.body.innerHTML = '<iframe srcdoc=\"<input type=password>\"></iframe>';
          document.querySelector('iframe').contentDocument.body.innerHTML = '<input type=password>'""",
    )
    assert evaluate(supervisor, vault._TAB_PROBES["login"]) is False
    assert json.loads(evaluate(supervisor, build_inspection_js("frame-nonce"))) == []

    # Existing light-DOM forms remain supported.
    evaluate(supervisor, "document.body.innerHTML = '<input type=password>'")
    assert evaluate(supervisor, vault._TAB_PROBES["login"]) is True
    light_controls = json.loads(evaluate(supervisor, build_inspection_js("light-nonce")))
    assert len(light_controls) == 1
    light_fill = json.loads(
        evaluate(
            supervisor,
            build_fill_js(
                [{"index": light_controls[0]["index"], "token": "current-password", "value": _DUMMY_PASSWORD}],
                login_origin,
                "light-nonce",
            ),
        ),
    )
    assert light_fill == {"filled": 1}


def test_registry_vault_fill_focuses_shadow_login_and_keeps_origin_guards(browser, monkeypatch, tmp_path):
    supervisor, task, origin, new_page = browser
    login_origin = origin.replace("127.0.0.1", "localhost")
    new_page(f"{login_origin}/login")
    create_nested_shadow_form(supervisor)

    store = VaultStore(base_dir=tmp_path / "vault")
    item = store.add_item(
        "login",
        "synthetic login",
        {"identifier": "synthetic-user", "identifier_type": "username", "password": _DUMMY_PASSWORD},
        origin=login_origin,
    )
    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: store)

    # Make the unrelated first tab active before the registry call. The vault
    # handler must focus the second tab by exact origin and shared probe scope.
    assert supervisor.focus_page(origin)["ok"]
    raw = registry.dispatch("browser_vault", {"action": "fill", "handle": item.id}, task_id=task)
    result = json.loads(raw if isinstance(raw, str) else json.dumps(raw))
    assert result["success"] is True
    assert result["filled_fields"] == 1
    assert result["origin"] == login_origin
    assert _DUMMY_PASSWORD not in raw

    assert supervisor.focus_page(login_origin)["ok"]
    assert evaluate(supervisor, "testRoot.querySelector('input').value === 'dummy-live-value'") is True
    assert evaluate(supervisor, "testRoot.querySelectorAll('[data-hermes-vault-slot]').length") == 0
    assert evaluate(supervisor, "document.querySelectorAll('input').length") == 0

    # A handle bound to an absent or different origin is rejected before any
    # secret-bearing evaluation, even while the matching login tab is active.
    absent = store.add_item(
        "login",
        "absent origin",
        {"identifier": "synthetic-user", "identifier_type": "username", "password": "alternate-dummy-value"},
        origin="https://absent.invalid",
    )
    refusal_raw = registry.dispatch("browser_vault", {"action": "fill", "handle": absent.id}, task_id=task)
    refusal = json.loads(refusal_raw if isinstance(refusal_raw, str) else json.dumps(refusal_raw))
    assert refusal["success"] is False
    assert refusal["error_type"] == "origin_mismatch"
    assert "alternate-dummy-value" not in refusal_raw
