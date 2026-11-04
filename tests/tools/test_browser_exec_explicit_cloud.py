import json
import subprocess
import sys

import pytest

from tools import browser_use_cli as bu


@pytest.fixture
def harness(monkeypatch):
    calls = []
    monkeypatch.setattr(bu, '_find_cli', lambda: [sys.executable, '-m', 'browser_harness.run'])
    monkeypatch.setattr(bu, '_base_subprocess_env', lambda **kw: {'BH_RUNTIME_DIR': '/scoped', 'BH_RUNTIME_DIR_SHARED': '1'} if kw.get('namespace', True) else {})
    monkeypatch.setattr(bu, '_named_daemon_kind', lambda env: 'cloud', raising=False)
    monkeypatch.setattr(bu, '_live_generation', lambda: 0)
    monkeypatch.setattr(bu, '_workspace_dir', lambda _: None)
    monkeypatch.setattr(bu, '_read_browser_cfg', dict)
    monkeypatch.setattr(bu, '_own_harness_daemon', lambda _: False)
    monkeypatch.setattr(bu, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(bu, '_route_exec_current', lambda *args: pytest.fail('cloud must not resolve configured local CDP'))
    def run(cmd, code, env, timeout):
        calls.append((code, env))
        return subprocess.CompletedProcess(cmd, 0, 'cloud-ok\n', '')
    monkeypatch.setattr(bu, '_run_cli_killing_process_group', run)
    return calls


def test_explicit_named_cloud_bypasses_attached_local_cdp(harness):
    result = json.loads(bu.browser_exec('print("cloud-ok")', session='dashboard', cloud=True))
    assert result['success'], result
    code, env = harness[0]
    assert env['BU_NAME'] == 'dashboard'
    assert env['BH_REQUIRE_EXISTING_DAEMON'] == '1'
    assert 'BU_CDP_WS' not in env and 'BU_CDP_URL' not in env
    assert 'daemon_browser_kind' in code
    assert '_hermes_own_tab' not in code


@pytest.mark.parametrize('kind', [None, 'local', 'cdp'])
def test_missing_or_noncloud_never_dispatches(harness, monkeypatch, kind):
    monkeypatch.setattr(bu, '_named_daemon_kind', lambda env: kind)
    result = json.loads(bu.browser_exec('print(1)', session='dashboard', cloud=True))
    assert 'error' in result
    assert 'Refusing local CDP' in result['error']
    assert not harness


def test_requires_name(harness):
    result = json.loads(bu.browser_exec('print(1)', cloud=True))
    assert 'error' in result
    assert 'requires' in result['error']
    assert not harness


def test_rejects_conflicting_local_intent(harness):
    result = json.loads(bu.browser_exec('print(1)', session='dashboard', cloud=True, local=True))
    assert 'error' in result
    assert not harness


def test_default_profile_can_find_cli_cloud_namespace(harness, monkeypatch):
    monkeypatch.setattr(bu, 'hermes_home_key', lambda path=None: 'default')
    seen = []
    def kind(env):
        seen.append(env.get('BH_RUNTIME_DIR'))
        return 'cloud' if not env.get('BH_RUNTIME_DIR') else 'cdp'
    monkeypatch.setattr(bu, '_named_daemon_kind', kind)
    result = json.loads(bu.browser_exec('print(1)', session='dashboard', cloud=True))
    assert result['success'], result
    assert seen == ['/scoped', None]
    assert 'BH_RUNTIME_DIR' not in harness[0][1]


def test_other_profile_never_searches_default_cli_namespace(harness, monkeypatch):
    monkeypatch.setattr(bu, 'hermes_home_key', lambda path=None: 'other' if path is None else 'default')
    seen = []
    def kind(env):
        seen.append(env.get('BH_RUNTIME_DIR'))
        return 'cloud' if not env.get('BH_RUNTIME_DIR') else 'cdp'
    monkeypatch.setattr(bu, '_named_daemon_kind', kind)
    result = json.loads(bu.browser_exec('print(1)', session='dashboard', cloud=True))
    assert 'error' in result
    assert seen == ['/scoped']
    assert not harness


def test_cloud_dispatch_does_not_own_or_reload_external_daemon(harness, monkeypatch):
    before = dict(bu._driven_daemons)
    monkeypatch.setattr(bu, '_own_harness_daemon', lambda env: pytest.fail('external owner'))
    monkeypatch.setattr(bu, '_reload_if_republished', lambda *args: pytest.fail('external reload'))
    assert json.loads(bu.browser_exec('print(1)', session='dashboard', cloud=True))['success']
    assert bu._driven_daemons == before


def test_cloud_dispatch_guard_blocks_replacement_before_user_code(monkeypatch):
    from types import ModuleType
    admin = ModuleType('browser_harness.admin')
    admin.daemon_browser_kind = lambda: 'cdp'
    monkeypatch.setitem(sys.modules, 'browser_harness.admin', admin)
    scope = {'executed': False}
    with pytest.raises(RuntimeError, match='refusing another browser'):
        exec(bu._CLOUD_DAEMON_GUARD + 'executed = True', scope)
    assert not scope['executed']


def test_cloud_schema_exposes_explicit_intent():
    assert bu.BROWSER_EXEC_SCHEMA['parameters']['properties']['cloud']['default'] is False


def test_registry_forwards_cloud_intent(monkeypatch):
    import hermes_cli.observability.shared_metrics_loop as metrics
    monkeypatch.setattr(metrics, 'record_browser_call', lambda call, backend: call())
    seen = {}
    def browser_exec(**kwargs):
        seen.update(kwargs)
        return 'registry-ok'
    monkeypatch.setattr(bu, 'browser_exec', browser_exec)
    assert bu._handle_browser_exec({'code': 'print(1)', 'session': 'dashboard', 'cloud': True}) == 'registry-ok'
    assert seen['cloud'] is True and seen['session'] == 'dashboard'


def test_readonly_probe_never_spawns(monkeypatch):
    def run(cmd, **kw):
        assert cmd[:2] == [sys.executable, '-c']
        assert 'daemon_browser_kind' in cmd[2]
        assert 'ensure_daemon' not in cmd[2] and 'start_remote_daemon' not in cmd[2]
        return subprocess.CompletedProcess(cmd, 0, 'cloud\n', '')
    monkeypatch.setattr(subprocess, 'run', run)
    assert bu._named_daemon_kind({}) == 'cloud'


def test_explicit_cloud_registry_preserves_native_screenshot(harness, monkeypatch, tmp_path):
    from PIL import Image
    from tools import vision_tools
    path = tmp_path / 'cloud.png'
    Image.new('RGB', (32, 32), 'white').save(path)
    monkeypatch.setattr(vision_tools, '_should_use_native_vision_fast_path', lambda: True)
    monkeypatch.setattr(bu, '_run_cli_killing_process_group', lambda cmd, *args:
                        subprocess.CompletedProcess(cmd, 0, str(path) + '\n', ''))
    result = bu._handle_browser_exec({'code': 'print(capture_screenshot())',
                                     'session': 'dashboard', 'cloud': True})
    assert result['_multimodal'] is True
    assert json.loads(result['text_summary'])['success'] is True
    assert result['meta']['screenshot_path'] == str(path)
    images = [part for part in result['content'] if part['type'] == 'image_url']
    assert len(images) == 1
    assert images[0]['image_url']['url'].startswith('data:image/')
