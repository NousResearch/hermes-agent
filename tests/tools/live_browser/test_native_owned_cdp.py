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
    reason='requires prepared task-owned Chromium and native browser-harness',
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
    from tools import browser_session as browser
    from tools import browser_session_lifecycle as lifecycle
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    for task in tasks:
        SUPERVISOR_REGISTRY.stop(task)
        assert lifecycle.cleanup_browser(task)
        assert cache_key(task) not in browser._active_sessions


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


def cache_key(task, session=''):
    from tools.browser_use_cli import _backend_cache_key
    return _backend_cache_key(task, session)


def navigate(task, url, session=''):
    from model_tools import handle_function_call
    from tools import browser_session as browser
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    from tests.waiting import wait_until

    result = json.loads(handle_function_call('browser_exec',
                        {'code': f"new_tab({json.dumps(url)})\nprint(page_info())", 'session': session}, task_id=task))
    assert result.get('success'), result
    identity = browser._active_sessions[cache_key(task, session)]['target_id']
    supervisor = SUPERVISOR_REGISTRY.get(task)
    assert supervisor is not None and supervisor.target_id == identity
    wait_until(lambda: supervisor.evaluate_runtime('location.href').get('result') == url,
               timeout=10, what='the actual harness navigation to commit')
    return identity


def test_two_real_tasks_keep_pins_and_terminal_cleanup_separate(owned_browser):
    from hermes_constants import hermes_home_key
    from tools import browser_session as browser
    from tools.browser_session_lifecycle import cleanup_browser
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    chrome, url, tasks = owned_browser
    first, second = 'live-pin-' + uuid4().hex, 'live-pin-' + uuid4().hex
    tasks.extend([first, second])
    original = set(chrome.targets())
    a, b = navigate(first, url), navigate(second, url)
    assert browser._active_sessions[cache_key(first)]['home_key'] == hermes_home_key()
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
    captured = SUPERVISOR_REGISTRY.capture(first)
    assert cleanup_browser(first)
    assert a not in chrome.targets() and b in chrome.targets()
    assert original.issubset(chrome.targets())
    assert not captured.is_valid() and SUPERVISOR_REGISTRY.get(first) is None
    assert cache_key(first) not in browser._active_sessions
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
    from tools import browser_session as browser

    chrome, url, tasks = owned_browser
    first, second = 'live-exit-' + uuid4().hex, 'live-exit-' + uuid4().hex
    tasks.extend([first, second])
    a, b = navigate(first, url), navigate(second, url)
    missing = tmp_path / ('missing-' + uuid4().hex)
    assert not missing.exists()
    with pytest.raises(FileNotFoundError), browser_turn_cleanup_boundary(first):
        with missing.open('rb') as stream:
            stream.read()
    assert cache_key(first) not in browser._active_sessions
    assert a not in chrome.targets() and b in chrome.targets()


def test_real_profile_shutdown_preserves_the_other_profile_supervisor(owned_browser, tmp_path):
    from hermes_cli.config import atomic_config_write
    from hermes_constants import hermes_home_key, set_hermes_home_override
    from tools import browser_session as browser
    from tools.browser_session_lifecycle import cleanup_profile_browsers
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    chrome, url, tasks = owned_browser
    first_task, second_task = 'profile-a-' + uuid4().hex, 'profile-b-' + uuid4().hex
    tasks.extend([first_task, second_task])
    first_home = hermes_home_key()
    first_key = cache_key(first_task)
    first_page = navigate(first_task, url)
    first_capture = SUPERVISOR_REGISTRY.capture(first_task)
    other_home = tmp_path / 'second-profile'
    other_home.mkdir()
    atomic_config_write(other_home / 'config.yaml', {
        'terminal': {'backend': 'local'}, 'lsp': {'enabled': False},
        'browser': {'allow_private_urls': True, 'cdp_url': chrome.root},
    })
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(other_home)
    token = set_hermes_home_override(other_home)
    with ExitStack() as resources:
        resources.callback(restore_home, token, previous)
        second_home = hermes_home_key()
        assert second_home != first_home
        second_page = navigate(second_task, url)
        second = SUPERVISOR_REGISTRY.get(second_task)
        second_capture = SUPERVISOR_REGISTRY.capture(second_task)
        cleanup_profile_browsers(first_home)
        assert SUPERVISOR_REGISTRY.get(first_task) is None and not first_capture.is_valid()
        assert SUPERVISOR_REGISTRY.get(second_task) is second and second_capture.is_valid()
        assert second.evaluate_runtime('location.href')['result'] == url
        assert first_page not in chrome.targets() and second_page in chrome.targets()
        assert first_key not in browser._active_sessions and cache_key(second_task) in browser._active_sessions
        cleanup_profile_browsers(second_home)
        assert SUPERVISOR_REGISTRY.get(second_task) is None and not second_capture.is_valid()


def test_real_endpoint_publish_during_discovery_recreates_only_the_candidate(owned_browser, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    from tests.tools.live_browser.cdp_discovery_proxy import HoldingDiscoveryProxy
    from tests.tools.live_browser.chrome_runtime import OwnedChrome
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    from tools.browser_session_cdp import set_live_cdp_override
    from tools.browser_session_lifecycle import cleanup_browser

    first, url, tasks = owned_browser
    task = 'live-rebind-' + uuid4().hex
    tasks.append(task)
    original = set(first.targets())
    with ExitStack() as resources:
        second = OwnedChrome(tmp_path / 'replacement-chrome')
        resources.callback(second.stop)
        proxy = HoldingDiscoveryProxy(first.root)
        resources.callback(proxy.close)
        resources.callback(set_live_cdp_override, None)
        resources.callback(cleanup_browser, task)
        set_live_cdp_override(proxy.root)
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(context.run, navigate, task, url)
            try:
                assert proxy.discovered.wait(timeout=30), 'Observe actual Chrome discovery'
                set_live_cdp_override(second.root)
            finally:
                proxy.release.set()
            target = future.result(timeout=60)
        assert target in second.targets()
        assert set(first.targets()) == original
        supervisor = SUPERVISOR_REGISTRY.get(task)
        assert supervisor is not None and supervisor.evaluate_runtime('location.href')['result'] == url
        assert cleanup_browser(task)
        assert target not in second.targets() and set(first.targets()) == original


def test_cleanup_during_real_discovery_cannot_resurrect_the_task(owned_browser):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from model_tools import handle_function_call

    from tests.tools.live_browser.cdp_discovery_proxy import HoldingDiscoveryProxy
    from tools import browser_session as browser
    from tools.browser_session_cdp import set_live_cdp_override
    from tools.browser_session_lifecycle import cleanup_browser
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    chrome, url, tasks = owned_browser
    task = 'live-retire-' + uuid4().hex
    tasks.append(task)
    original = set(chrome.targets())
    with ExitStack() as resources:
        proxy = HoldingDiscoveryProxy(chrome.root)
        resources.callback(proxy.close)
        resources.callback(set_live_cdp_override, None)
        set_live_cdp_override(proxy.root)
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(context.run, handle_function_call, 'browser_exec',
                                    {'code': f"new_tab({json.dumps(url)})"}, task_id=task)
            try:
                assert proxy.discovered.wait(timeout=30), 'Observe actual Chrome discovery'
                assert cleanup_browser(task)
            finally:
                proxy.release.set()
            result = json.loads(future.result(timeout=60))
        assert result.get('code') == 'browser_session_retired', result
        assert cache_key(task) not in browser._active_sessions
        assert SUPERVISOR_REGISTRY.get(task) is None
        assert set(chrome.targets()) == original


def test_headed_configuration_still_releases_only_its_shared_cdp_tab(owned_browser):
    from hermes_cli.config import atomic_config_write
    from hermes_constants import get_hermes_home
    from tools.browser_session_cloud import is_headed_mode
    from tools.browser_session_lifecycle import cleanup_browser_for_turn

    chrome, url, tasks = owned_browser
    atomic_config_write(get_hermes_home() / 'config.yaml', {
        'terminal': {'backend': 'local'}, 'lsp': {'enabled': False},
        'browser': {'allow_private_urls': True, 'headed': True, 'cdp_url': chrome.root},
    })
    assert is_headed_mode()
    first, second = 'headed-a-' + uuid4().hex, 'headed-b-' + uuid4().hex
    tasks.extend([first, second])
    original = set(chrome.targets())
    a, b = navigate(first, url), navigate(second, url)
    assert cleanup_browser_for_turn(first)
    assert a not in chrome.targets() and b in chrome.targets()
    assert original.issubset(chrome.targets())


def test_real_named_session_persists_until_its_own_profile_cleanup(owned_browser):
    from hermes_constants import hermes_home_key
    from tools import browser_session as browser
    from tools.browser_session_lifecycle import cleanup_browser_for_turn, cleanup_profile_browsers, cleanup_single_session

    chrome, url, tasks = owned_browser
    first, second = 'named-a-' + uuid4().hex, 'named-b-' + uuid4().hex
    tasks.extend([first, second])
    session = 'named-' + uuid4().hex
    key = cache_key(first, session)
    original = set(chrome.targets())
    try:
        a = navigate(first, url, session)
        assert cleanup_browser_for_turn(first)
        assert a in chrome.targets() and key in browser._active_sessions
        b = navigate(second, url, session)
        assert b != a and {a, b}.issubset(chrome.targets())
        cleanup_profile_browsers(hermes_home_key())
        assert {a, b}.isdisjoint(chrome.targets()) and key not in browser._active_sessions
        assert set(chrome.targets()) == original
    finally:
        cleanup_single_session(key)


def test_explicitly_selected_existing_tab_remains_unowned(owned_browser):
    from model_tools import handle_function_call
    from tools.browser_session_lifecycle import cleanup_browser
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    chrome, url, tasks = owned_browser
    task = 'borrow-' + uuid4().hex
    tasks.append(task)
    original = set(chrome.targets())
    borrowed = chrome.new_page(url)
    created = navigate(task, url)
    result = json.loads(handle_function_call('browser_exec',
                        {'code': f"switch_tab({json.dumps(borrowed)})\nprint(current_tab())"}, task_id=task))
    assert result.get('success'), result
    assert SUPERVISOR_REGISTRY.get(task).target_id == borrowed
    assert cleanup_browser(task)
    assert borrowed in chrome.targets() and created not in chrome.targets()
    assert original.issubset(chrome.targets())


def test_two_real_cleanup_requests_release_one_task_and_keep_its_sibling(owned_browser):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    from tools import browser_session as browser
    from tools.browser_session_lifecycle import cleanup_browser

    chrome, url, tasks = owned_browser
    first, second = 'cleanup-a-' + uuid4().hex, 'cleanup-b-' + uuid4().hex
    tasks.extend([first, second])
    original = set(chrome.targets())
    a, b = navigate(first, url), navigate(second, url)
    ready = threading.Barrier(3)

    def cleanup():
        ready.wait(timeout=10)
        return cleanup_browser(first)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(copy_context().run, cleanup) for _ in range(2)]
        ready.wait(timeout=10)
        assert all(future.result(timeout=30) for future in futures)
    assert cache_key(first) not in browser._active_sessions
    assert cache_key(first) not in browser._retired_sessions
    assert a not in chrome.targets() and b in chrome.targets()
    assert original.issubset(chrome.targets())
