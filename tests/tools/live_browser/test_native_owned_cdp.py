"""Exercise exact page ownership and lifecycle on real headless Chromium/CDP."""
import functools
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('HERMES_E2E_BROWSER') != '1',
    reason='requires prepared task-owned Chromium and native agent-browser',
)


def restore_home(token, previous):
    from hermes_constants import reset_hermes_home_override

    reset_hermes_home_override(token)
    if previous is None:
        os.environ.pop('HERMES_HOME')
    else:
        os.environ['HERMES_HOME'] = previous


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    assert not thread.is_alive()


def restore_runtime(previous):
    if previous is None:
        os.environ.pop('HERMES_RUNTIME_DIR')
    else:
        os.environ['HERMES_RUNTIME_DIR'] = previous


def cleanup_tasks(tasks):
    from tools import browser_tool as browser
    from tools import browser_tool_lifecycle as lifecycle
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    for task in tasks:
        SUPERVISOR_REGISTRY.stop(task)
        assert lifecycle.cleanup_browser(task)
        assert task not in browser._active_sessions


@pytest.fixture
def owned_browser(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(home)
    from hermes_constants import set_hermes_home_override

    token = set_hermes_home_override(home)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(home / 'config.yaml', {
        'terminal': {'backend': 'local'}, 'lsp': {'enabled': False},
        'browser': {'allow_private_urls': True},
    })
    from tests.tools.live_browser.chrome_runtime import OwnedChrome
    with ExitStack() as resources:
        resources.callback(restore_home, token, previous)
        runtime = Path(os.environ['HERMES_E2E_ARTIFACTS']) / 'pm-runtime'
        assert runtime.is_dir() and (runtime / 'facts.json').is_file()
        previous_runtime = os.environ.get('HERMES_RUNTIME_DIR')
        os.environ['HERMES_RUNTIME_DIR'] = str(runtime)
        resources.callback(restore_runtime, previous_runtime)
        chrome = OwnedChrome(tmp_path / 'chrome')
        resources.callback(chrome.stop)
        atomic_config_write(home / 'config.yaml', {
            'terminal': {'backend': 'local'}, 'lsp': {'enabled': False},
            'browser': {'allow_private_urls': True, 'cdp_url': chrome.root},
        })
        public = tmp_path / 'public'
        public.mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[3] / 'README.md', public / 'README.txt')
        server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(SimpleHTTPRequestHandler, directory=str(public)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        resources.callback(stop_server, server, thread)
        tasks = []
        resources.callback(cleanup_tasks, tasks)
        yield chrome, f'http://127.0.0.1:{server.server_port}/README.txt', tasks


def navigate(task, url):
    from model_tools import handle_function_call
    from tools import browser_tool as browser
    from tools.browser_tool_cdp import _pinned_cdp_target_id

    result = json.loads(handle_function_call('browser_navigate', {'url': url}, task_id=task))
    assert result.get('success'), result
    identity = _pinned_cdp_target_id(task)
    assert identity and browser._active_sessions[task]['target_id'] == identity
    return identity


def test_two_real_tasks_keep_pins_and_terminal_cleanup_separate(owned_browser):
    from tools import browser_tool as browser
    from tools.browser_tool_lifecycle import cleanup_browser
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    chrome, url, tasks = owned_browser
    first, second = 'live-pin-' + uuid4().hex, 'live-pin-' + uuid4().hex
    tasks.extend([first, second])
    original = set(chrome.targets())
    a, b = navigate(first, url), navigate(second, url)
    assert a != b and {a, b}.isdisjoint(original)
    supervisor = SUPERVISOR_REGISTRY.get(first)
    assert supervisor is not None and supervisor.target_id == a
    assert supervisor.evaluate_runtime('location.href')['result'] == url
    counter = json.dumps('nativeSideEffects' + uuid4().hex)
    expression = f'(() => {{ globalThis[{counter}] = (globalThis[{counter}] || 0) + 1; const value = {{}}; value.self = value; return value; }})()'
    failed = supervisor.evaluate_runtime(expression)
    assert failed['ok'] is False and failed['kind'] == 'result_serialization_failed', failed
    assert failed['data']['expression_executed'] is True and failed['data']['replayed'] is False
    assert supervisor.evaluate_runtime(f'globalThis[{counter}]')['result'] == 1
    assert cleanup_browser(first)
    assert a not in chrome.targets() and b in chrome.targets()
    assert original.issubset(chrome.targets())
    denied = json.loads(browser._browser_eval('location.href', task_id=first))
    assert denied['code'] == 'browser_session_retired' and denied['data']['terminal'] is True
    assert first not in browser._active_sessions
    replacement = navigate(first, url)
    assert replacement not in {a, b} and b in chrome.targets()


def test_real_supervisor_replacement_invalidates_only_the_previous_capture(owned_browser):
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    chrome, url, tasks = owned_browser
    task = 'live-supervisor-' + uuid4().hex
    tasks.append(task)
    first_page, second_page = chrome.new_page(url), chrome.new_page(url)
    first = SUPERVISOR_REGISTRY.get_or_start(task, chrome.websocket, target_id=first_page)
    captured = SUPERVISOR_REGISTRY.capture(task)
    assert captured.is_valid() and first.evaluate_runtime('location.href')['result'] == url
    second = SUPERVISOR_REGISTRY.get_or_start(task, chrome.websocket, target_id=second_page)
    assert first is not second and not captured.is_valid()
    assert second.target_id == second_page and second.evaluate_runtime('location.href')['result'] == url
    assert SUPERVISOR_REGISTRY._pop(task, expected=first) is None
    assert SUPERVISOR_REGISTRY.get(task) is second
    assert {first_page, second_page}.issubset(chrome.targets())


def test_actual_filesystem_failure_still_finalizes_only_its_browser_task(owned_browser, tmp_path):
    from agent.browser_turn_cleanup import browser_turn_cleanup_boundary
    from tools import browser_tool as browser

    chrome, url, tasks = owned_browser
    first, second = 'live-exit-' + uuid4().hex, 'live-exit-' + uuid4().hex
    tasks.extend([first, second])
    a, b = navigate(first, url), navigate(second, url)
    missing = tmp_path / ('missing-' + uuid4().hex)
    assert not missing.exists()
    with pytest.raises(FileNotFoundError), browser_turn_cleanup_boundary(first):
        with missing.open('rb') as stream:
            stream.read()
    assert first not in browser._active_sessions
    assert a not in chrome.targets() and b in chrome.targets()
