"""Terminal-subshell PATH completion in ``tools/environments/local.py``.

A backend started by a non-interactive SSH session, systemd or a GUI launcher
inherits a PATH without ``~/.local/bin`` (only the login shell adds it), so CLIs
installed there were ``command not found`` from the terminal tool (#111778).
"""

import os
import sys

import pytest

from tools.environments import local as local_mod
from tools.environments.local import _append_missing_sane_path_entries, _make_run_env

pytestmark = pytest.mark.platforms("posix")  # POSIX PATH completion only


def test_existing_user_local_bin_appended_after_inherited_entries(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    entries = _make_run_env({})["PATH"].split(os.pathsep)

    assert entries[:2] == ["/usr/bin", "/bin"]
    assert entries.count(str(local_bin)) == 1
    # Already on PATH: position kept, no duplicate appended.
    already = _append_missing_sane_path_entries(f"{local_bin}:/usr/bin").split(":")
    assert already[0] == str(local_bin) and already.count(str(local_bin)) == 1


def test_missing_user_local_bin_not_appended(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])

    assert ".local" not in _append_missing_sane_path_entries("/usr/bin:/bin")


def test_background_spawn_env_completes_path_like_foreground(monkeypatch, tmp_path):
    """``ProcessRegistry._spawn_env`` must resolve the same managed-runtime and
    ``~/.local/bin`` entries as a foreground ``_make_run_env`` run: on an install
    whose only uv is the managed ``$HERMES_HOME/bin`` one, a background job
    printed ``command not found: uv`` while the same command ran in the
    foreground (#124820)."""
    from tools.process_registry import ProcessRegistry

    managed_bin = tmp_path / "hermes-home" / "bin"
    managed_bin.mkdir(parents=True)
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(managed_bin)])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    fg = _make_run_env({})["PATH"].split(":")
    bg = ProcessRegistry._spawn_env({})["PATH"].split(":")

    # Inherited user entries keep precedence in both modes.
    assert bg[:2] == ["/usr/bin", "/bin"]
    # The managed runtime and ~/.local/bin resolve in background as in foreground.
    for entry in (str(managed_bin), str(local_bin)):
        assert entry in bg
        assert entry in fg


def test_ddgs_worker_env_completes_path(monkeypatch, tmp_path):
    """The DDGS search worker spawn passes the same completion: on a backend
    launched by systemd/GUI the worker would inherit no ``~/.local/bin``
    (#124820 follow-up — the surface the ``_sanitize_subprocess_env`` docstring names)."""
    from plugins.web.ddgs import provider as ddgs_mod

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    captured: dict = {}

    class _FakeProc:
        pid = 4242

        def poll(self):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

        def communicate(self, request=None, timeout=None):
            return '{"ok": true, "results": []}', None

    def _fake_spawn(env):
        captured.update(env)
        return _FakeProc()

    monkeypatch.setattr(ddgs_mod, "_spawn_worker", _fake_spawn)
    monkeypatch.setattr(ddgs_mod, "_last_worker_proc", None)
    assert ddgs_mod._run_ddgs_search_bounded("query", 3) == []
    # Inherited user entries keep precedence; ~/.local/bin is appended for the worker.
    assert captured["PATH"].split(":")[:2] == ["/usr/bin", "/bin"]
    assert str(local_bin) in captured["PATH"].split(":")


def test_cua_driver_env_completes_path(monkeypatch, tmp_path):
    """``sanitized_cua_driver_env`` passes the same completion: the cua-driver
    binary can live in ``~/.local/bin`` on a systemd/GUI-launched host (#124820
    follow-up — the other surface the docstring names)."""
    from tools.computer_use import cua_backend

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr(cua_backend, "cua_driver_child_env",
                        lambda base_env=None: {"PATH": "/usr/bin:/bin"})

    env = cua_backend.sanitized_cua_driver_env()

    assert env["PATH"].split(":")[:2] == ["/usr/bin", "/bin"]
    assert str(local_bin) in env["PATH"].split(":")
