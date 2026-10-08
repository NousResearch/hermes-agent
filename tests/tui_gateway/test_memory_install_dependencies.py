"""Default-profile RPC handler + real offline PM admission (no socket transport)."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.pm._fixtures import client as client, isolated_python as isolated_python, worker_toolchain, _wheel


@pytest.mark.parametrize('enable', [False, True])
def test_desktop_installer_prepares_dependencies_without_selecting_memory(client, isolated_python, tmp_path, monkeypatch, enable):
    from pm import paths
    from pm.environments import selected_venv
    from hermes_cli import plugin_catalog
    from hermes_cli.config import atomic_config_write
    from tui_gateway import server
    from utils import fast_safe_load

    worker_toolchain(client, monkeypatch, isolated_python)
    # All dependency resolution/publication runs in the real independent PM
    # worker, against an offline project and wheel; only tool acquisition differs.
    project = tmp_path / 'project'
    project.mkdir()
    monkeypatch.setattr(paths, 'repo_root', lambda: project)
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    _wheel(wheels, 'memory_fixture_sdk')
    (project / 'pyproject.toml').write_text(
        '[project]\nname="memory-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\nno-index=true\noffline=true\n'
        f'find-links=[{json.dumps(str(wheels))}]\n')
    client.lock_project(project, offline=True, explicit=True)
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(exist_ok=True)
    atomic_config_write(home / 'config.yaml', {'memory': {'provider': 'previous'}})
    repo = tmp_path / 'plugin-repo'
    repo.mkdir()
    (repo / 'plugin.yaml').write_text('name: memory-proof\nversion: 1.0.0\npython_dependencies: ["memory-fixture-sdk==1.0"]\n')
    (repo / '__init__.py').write_text('''from agent.memory_provider import MemoryProvider
class Provider(MemoryProvider):
    name = "memory-proof"
    def is_available(self):
        import memory_fixture_sdk
        return memory_fixture_sdk.__version__ == "1.0"
    def initialize(self, config, hermes_home): pass
    def shutdown(self): pass
    def get_tool_schemas(self): return []
def register(ctx): ctx.register_memory_provider(Provider())
''')
    env = {**os.environ, 'GIT_AUTHOR_NAME': 'Fixture', 'GIT_AUTHOR_EMAIL': 'fixture@example.test',
           'GIT_COMMITTER_NAME': 'Fixture', 'GIT_COMMITTER_EMAIL': 'fixture@example.test'}
    for args in (('init', '-q'), ('add', '.'), ('commit', '-qm', 'fixture')):
        subprocess.run(['git', *args], cwd=repo, env=env, check=True, capture_output=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    entry = plugin_catalog.PluginCatalogEntry('memory-proof', repo.as_uri(), sha, 'Fixture', 'Fixture', category='memory', featured=True)
    monkeypatch.setattr(plugin_catalog, 'load_catalog', lambda *a: [entry])
    monkeypatch.setattr(plugin_catalog, 'fetch_live_catalog', lambda **kw: None)
    monkeypatch.setattr(plugin_catalog, 'load_removed_list', lambda *a: [])
    # No selected dependency environment exists before this actual handler call.
    response = server.handle_request({'id': 'install', 'method': 'plugins.manage', 'params': {
        'action': 'install', 'catalog_name': entry.name, 'enable': enable, 'profile': 'default',
    }})
    assert 'error' not in response, response
    assert response['result']['enabled'] is enable
    installed = home / 'plugins' / entry.name
    assert (installed / '__init__.py').read_bytes() == (repo / '__init__.py').read_bytes()
    from hermes_cli.plugins_cmd_catalog import catalog_install_record
    assert catalog_install_record(installed)['sha'] == sha
    cfg = fast_safe_load((home / 'config.yaml').read_text())
    assert cfg['memory']['provider'] == 'previous'
    assert entry.name not in cfg.get('plugins', {}).get('disabled', [])
    if enable:
        assert entry.name in cfg['plugins']['enabled']
        python = selected_venv(project) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from starlette.testclient import TestClient
        http = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
        status = http.get('/api/memory', params={'profile': 'default'})
        assert status.status_code == 200, status.text
        row = next(row for row in status.json()['providers'] if row['name'] == entry.name)
        # The running test host still uses its old environment. The normal
        # installer publishes a new generation; it does not mutate sys.path.
        assert row['status'] == 'unavailable', row
        probe_code = (
            "import sys, importlib.util, memory_fixture_sdk; from pathlib import Path; "
            "assert memory_fixture_sdk.__version__ == '1.0'; "
            "assert Path(memory_fixture_sdk.__file__).is_relative_to(Path(sys.prefix)); "
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r}); "
            f"spec=importlib.util.spec_from_file_location('installed_memory', {str(installed / '__init__.py')!r}); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert module.Provider().is_available(); print('provider ready in selected generation')"
        )
        fresh = subprocess.run([str(python), '-I', '-c', probe_code], text=True, capture_output=True, timeout=30)
        assert fresh.returncode == 0, fresh.stderr
        assert fresh.stdout.strip() == 'provider ready in selected generation'

        assert status.json()['active'] == 'previous'

    else:
        assert not selected_venv(project).exists(), 'Code-only install must not claim dependency admission'
