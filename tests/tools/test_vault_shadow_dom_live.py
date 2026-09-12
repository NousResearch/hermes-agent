"""Real Chromium regression: tab discovery, inspection and fill share DOM scope.

Runs when Chrome/Chromium is installed; uses only an isolated browser profile,
loopback pages and a temporary local vault. No personal browser or manager.
"""
import http.server
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from agent.vault_login_classifier import build_fill_js, build_inspection_js
from tools import browser_vault_tool as vault
from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.browser_use_cli import _attach_vault_supervisor
from tools.registry import registry


@pytest.fixture
def browser(tmp_path):
    executable = next((shutil.which(n) for n in ("chromium", "chromium-browser", "google-chrome")
                       if shutil.which(n)), None)
    mac = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    if not executable and mac.exists():
        executable = str(mac)
    if not executable:
        pytest.skip("Chrome/Chromium is required for the live DOM regression")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<!doctype html><title>vault regression</title><body></body>")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    profile = tmp_path / 'chrome'
    process = subprocess.Popen([executable, '--headless=new', '--no-sandbox',
                                '--disable-dev-shm-usage', '--no-first-run',
                                '--no-default-browser-check', '--remote-debugging-port=0',
                                f'--user-data-dir={profile}', origin + '/unrelated'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    task = 'vault-shadow-regression'
    try:
        deadline = time.monotonic() + 20
        active_port = profile / 'DevToolsActivePort'
        address = []
        while time.monotonic() < deadline and process.poll() is None:
            if active_port.exists():
                address = active_port.read_text().splitlines()
                if len(address) >= 2 and address[0].isdigit() and address[1].startswith('/devtools/browser/'):
                    break
            time.sleep(.05)
        assert len(address) >= 2, 'isolated Chrome did not publish its DevTools address'
        port, ws_path = address[:2]
        _attach_vault_supervisor({'BU_CDP_WS': f'ws://127.0.0.1:{port}{ws_path}'}, task)
        sup = SUPERVISOR_REGISTRY.get(task)
        assert sup is not None, 'browser_exec must attach the vault supervisor'
        def wait_page(url):
            from urllib.parse import urlsplit

            parsed = urlsplit(url)
            page_origin = f'{parsed.scheme}://{parsed.netloc}'
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if sup.focus_page(page_origin).get('ok'):
                    ready = sup.evaluate_runtime(
                        f'location.href === {json.dumps(url)} && document.readyState === "complete"')
                    if ready.get('ok') and ready.get('result'):
                        return
                time.sleep(.05)
            pytest.fail('test page did not finish loading')

        wait_page(origin + '/unrelated')

        def new_page(url):
            request = urllib.request.Request(f'http://127.0.0.1:{port}/json/new?{url}', method='PUT')
            with urllib.request.urlopen(request, timeout=5) as response:
                json.load(response)
            wait_page(url)
        yield sup, task, origin, new_page
    finally:
        SUPERVISOR_REGISTRY.stop(task)
        process.terminate()
        process.wait(timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def evaluate(sup, expression):
    result = sup.evaluate_runtime(expression)
    assert result['ok'], result
    return result['result']


def shadow_form(sup):
    evaluate(sup, """(() => {
      const host = document.createElement('div'); document.body.append(host);
      const outer = host.attachShadow({mode:'open'});
      outer.innerHTML = '<div id="nested"></div>';
      const inner = outer.querySelector('#nested').attachShadow({mode:'open'});
      inner.innerHTML = '<form><label id="pw-label">Password</label>' +
        '<input type="password" aria-labelledby="pw-label" autocomplete="current-password">' +
        '<input autocomplete="cc-number"><input autocomplete="postal-code">' +
        '<input autocomplete="one-time-code"></form>';
      window.testRoot = inner;
      window.testInputObservers = [];
      for (const [name, target] of [['host', host], ['document', document]]) {
        target.addEventListener('input', event => {
          window.testInputObservers.push({observer: name, composed: event.composed,
            retargeted: event.target === host, inputType: event.inputType});
        });
      }
    })()""")


def test_shadow_probe_inspection_and_nonce_fill_share_scope(browser):
    sup, _, origin, _ = browser
    shadow_form(sup)
    for kind in ('login', 'payment', 'address', 'otp'):
        assert evaluate(sup, vault._TAB_PROBES[kind]), kind
    controls = json.loads(evaluate(sup, build_inspection_js('test-nonce')))
    password = next(c for c in controls if c['type'] == 'password')
    assert password['label'].strip() == 'Password'
    assert password['formIndex'] >= 0
    fills = [{'index': password['index'], 'token': 'current-password', 'value': 'dummy-live-value'}]
    refused = json.loads(evaluate(sup, build_fill_js(fills, origin + '.invalid', 'test-nonce')))
    assert refused['refused'] == 'origin_changed'
    assert evaluate(sup, "testRoot.querySelector('input').value === ''")
    assert json.loads(evaluate(sup, build_fill_js(fills, origin, 'wrong-nonce')))['filled'] == 0
    # A stale nonce attempt clears stamps; inspect again before the authorized write.
    evaluate(sup, build_inspection_js('test-nonce'))
    assert json.loads(evaluate(sup, build_fill_js(fills, origin, 'test-nonce')))['filled'] == 1
    assert evaluate(sup, "testRoot.querySelector('input').value === 'dummy-live-value'")
    assert evaluate(sup, "testRoot.querySelectorAll('[data-hermes-vault-slot]').length") == 0
    # Stamps alone do not authorize a password write into a field whose type changed.
    evaluate(sup, build_inspection_js('test-nonce'))
    evaluate(sup, "testRoot.querySelector('input').type = 'text'; testRoot.querySelector('input').value = ''")
    assert json.loads(evaluate(sup, build_fill_js(fills, origin, 'test-nonce')))['filled'] == 0
    # Shadow traversal must not widen mutation scope to embedded documents.
    evaluate(sup, """document.body.innerHTML = '<iframe></iframe>';
      document.querySelector('iframe').contentDocument.body.innerHTML = '<input type=password>'""")
    assert not evaluate(sup, vault._TAB_PROBES['login'])
    assert json.loads(evaluate(sup, build_inspection_js('frame-test'))) == []
    # Ordinary light-DOM fields keep working too.
    evaluate(sup, "document.body.innerHTML = '<input type=password>'")
    assert evaluate(sup, vault._TAB_PROBES['login'])
    controls = json.loads(evaluate(sup, build_inspection_js('light-test')))
    fills[0]['index'] = controls[0]['index']
    assert json.loads(evaluate(sup, build_fill_js(fills, origin, 'light-test')))['filled'] == 1


def test_vault_fill_finds_shadow_login_not_unrelated_first_tab(browser):
    from agent.vault_store import get_vault_store

    sup, task, origin, new_page = browser
    # A distinct exact origin on the same server, leaving the original tab intact.
    login_origin = origin.replace('127.0.0.1', 'localhost')
    new_page(login_origin + '/page')
    shadow_form(sup)
    assert sup.focus_page(origin)['ok']
    item = get_vault_store().add_item('login', 'synthetic login',
                                    {'identifier': 'test', 'identifier_type': 'username',
                                     'password': 'dummy-live-value'}, origin=login_origin)
    raw = registry.dispatch('browser_vault_fill', {'handle': item.id}, task_id=task)
    result = json.loads(raw)
    assert result['success'], result
    assert result['filled_fields'] == 1 and result['origin'] == login_origin
    assert 'dummy-live-value' not in raw
    assert evaluate(sup, "testRoot.querySelector('input').value === 'dummy-live-value'")
    # Framework observers outside both shadow roots must see the input event,
    # with normal shadow-DOM retargeting and without exposing the field value.
    assert evaluate(sup, 'window.testInputObservers') == [
        {'observer': observer, 'composed': True, 'retargeted': True, 'inputType': 'insertText'}
        for observer in ('host', 'document')
    ]
    assert sup.focus_page(origin)['ok']
    assert evaluate(sup, "document.querySelectorAll('input').length") == 0
    # No tab on the bound origin: never weaken origin matching to make progress.
    other = get_vault_store().add_item('login', 'absent origin',
                                     {'identifier': 'test', 'identifier_type': 'username',
                                      'password': 'another-dummy'}, origin='https://absent.invalid')
    refusal = json.loads(registry.dispatch('browser_vault_fill', {'handle': other.id}, task_id=task))
    assert refusal['error_type'] == 'origin_mismatch'
