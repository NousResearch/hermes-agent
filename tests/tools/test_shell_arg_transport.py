"""Non-path values survive local argv and serialized POSIX-script transport."""
import json
import os
import subprocess

import pytest

from tools.environments.local import LocalEnvironment, _find_bash
from tools.file_operations import ShellFileOperations


class _SerializedShell:
    is_local = False

    def __init__(self, bash, cwd, env):
        self.bash, self.cwd, self.env = bash, str(cwd), env

    def execute(self, command, *, cwd=None, timeout=30, **kwargs):
        # Deliver command text on stdin, without Windows command-line parsing.
        script = json.loads(json.dumps({'command': command}))['command']
        result = subprocess.run([self.bash, '--noprofile', '--norc', '-s'],
                                input=script, cwd=cwd or self.cwd, env=self.env,
                                capture_output=True, text=True, encoding='utf-8', timeout=timeout)
        return {'output': result.stdout, 'returncode': result.returncode}


@pytest.mark.parametrize('serialized', [False, True])
def test_non_path_bytes_survive_real_shell_transport(tmp_path, monkeypatch, serialized):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'store'))
    monkeypatch.setenv('HOME', str(home))
    env = {**os.environ, 'BASH_ENV': '', 'ENV': ''}
    bash = _find_bash()
    backend = (_SerializedShell(bash, tmp_path, env) if serialized
               else LocalEnvironment(cwd=str(tmp_path), timeout=30, env=env))
    ops = ShellFileOperations(backend)
    try:
        for value in ('', 'two words', "quote'and\"quote", '$() ` : ;',
                      'line\nbreak', *('a' + '\\' * length + '.b' for length in (1, 2, 3, 4, 8)),
                      "a\\'b"):
            quoted = ops._escape_shell_arg(value, translate_path=False)
            result = ops._exec(f"printf '%s' {quoted} | od -An -v -tx1")
            assert result.exit_code == 0, (value, result.stdout)
            assert bytes.fromhex(result.stdout) == value.encode('utf-8'), (serialized, value, result.stdout)
    finally:
        if isinstance(backend, LocalEnvironment):
            backend.cleanup()
