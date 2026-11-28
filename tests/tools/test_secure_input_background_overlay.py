"""Background secret injection uses the real pipe spawn, with synthetic values."""
import json
import shlex
import shutil
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.platforms('posix')
def test_background_explicit_credential_is_overlaid_after_ambient_filter(tmp_path, monkeypatch):
    from tools.process_registry import ProcessRegistry
    from tools.terminal_tool_background import _spawn

    registry = ProcessRegistry.__new__(ProcessRegistry)
    # No global registry, watcher thread or checkpoint is needed for this pipe contract.
    monkeypatch.setattr(registry, '_track_started', lambda *args: None)
    monkeypatch.setattr(registry, '_scope_argv',
                        lambda session, command, *args: [shutil.which('bash'), '-c', command])
    code = "import os,json; print(json.dumps({'explicit':os.environ.get('GH_TOKEN')=='synthetic-explicit-background','ambient':os.environ.get('GITHUB_TOKEN') is not None}))"
    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)
    session = _spawn(registry, env=SimpleNamespace(env={'GITHUB_TOKEN': 'synthetic-ambient-blocked'}),
                     env_type='local', command=command, cwd=str(tmp_path), effective_task_id='fixture',
                     task_id='fixture', session_key='fixture', effective_pty=False,
                     transient_env={'GH_TOKEN': 'synthetic-explicit-background'})
    try:
        stdout, _ = session.process.communicate(timeout=10)
        assert session.process.returncode == 0
        assert json.loads(stdout) == {'explicit': True, 'ambient': False}
    finally:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait(timeout=5)
