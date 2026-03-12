"""Regression tests for terminal subprocess environment scoping."""

import os
import shlex
import sys
import tempfile
import time
import types
from pathlib import Path

_IMPORT_HERMES_HOME = Path(tempfile.mkdtemp(prefix="hermes_test_import_"))
for _subdir in ("sessions", "cron", "memories", "skills"):
    (_IMPORT_HERMES_HOME / _subdir).mkdir()
os.environ["HERMES_HOME"] = str(_IMPORT_HERMES_HOME)

for _key in (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
):
    os.environ.pop(_key, None)

from tools.environments.local import LocalEnvironment
from tools.process_registry import ProcessRegistry


_BLOCKED_VAR = "OPENAI_BASE_URL"
_PARENT_VALUE = "http://parent.invalid/v1"
_EXPLICIT_VALUE = "http://explicit.invalid/v1"


def _print_env_command(var_name: str) -> str:
    code = f'import os; print(os.getenv("{var_name}", ""))'
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _wait_for_exit(session, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.exited:
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for session {session.id} to exit")


class TestLocalEnvironmentEnvScoping:
    def test_execute_scrubs_provider_env_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_BLOCKED_VAR, _PARENT_VALUE)

        env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
        result = env.execute(_print_env_command(_BLOCKED_VAR))

        assert result["returncode"] == 0
        assert result["output"].strip() == ""

    def test_execute_allows_explicit_provider_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_BLOCKED_VAR, _PARENT_VALUE)

        env = LocalEnvironment(
            cwd=str(tmp_path),
            timeout=5,
            env={_BLOCKED_VAR: _EXPLICIT_VALUE},
        )
        result = env.execute(_print_env_command(_BLOCKED_VAR))

        assert result["returncode"] == 0
        assert result["output"].strip() == _EXPLICIT_VALUE


class TestProcessRegistryEnvScoping:
    def test_spawn_local_scrubs_provider_env_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_BLOCKED_VAR, _PARENT_VALUE)

        registry = ProcessRegistry()
        session = registry.spawn_local(
            command=_print_env_command(_BLOCKED_VAR),
            cwd=str(tmp_path),
        )

        _wait_for_exit(session)
        assert session.exit_code == 0
        assert session.output_buffer.strip() == ""

    def test_spawn_local_allows_explicit_provider_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_BLOCKED_VAR, _PARENT_VALUE)

        registry = ProcessRegistry()
        session = registry.spawn_local(
            command=_print_env_command(_BLOCKED_VAR),
            cwd=str(tmp_path),
            env_vars={_BLOCKED_VAR: _EXPLICIT_VALUE},
        )

        _wait_for_exit(session)
        assert session.exit_code == 0
        assert session.output_buffer.strip() == _EXPLICIT_VALUE

    def test_spawn_local_pty_scrubs_provider_env_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_BLOCKED_VAR, _PARENT_VALUE)

        captured = {}

        class _FakePty:
            pid = 123
            exitstatus = 0

            def isalive(self):
                return False

            def read(self, _size):
                return b""

            def wait(self):
                return 0

        class _FakePtyProcess:
            @staticmethod
            def spawn(argv, cwd, env, dimensions):
                captured["argv"] = list(argv)
                captured["cwd"] = cwd
                captured["env"] = dict(env)
                captured["dimensions"] = dimensions
                return _FakePty()

        monkeypatch.setitem(
            sys.modules,
            "ptyprocess",
            types.SimpleNamespace(PtyProcess=_FakePtyProcess),
        )

        registry = ProcessRegistry()
        session = registry.spawn_local(
            command="true",
            cwd=str(tmp_path),
            use_pty=True,
        )

        _wait_for_exit(session)
        assert session.exit_code == 0
        assert captured["cwd"] == str(tmp_path)
        assert captured["env"]["PYTHONUNBUFFERED"] == "1"
        assert _BLOCKED_VAR not in captured["env"]
