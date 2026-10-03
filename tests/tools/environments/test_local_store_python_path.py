"""Keep the PM store Python off terminal children PATH (#129097).

On a store install ``pm.activate()`` puts the pinned store Python's bin dir at
the front of PATH, so the terminal tool's ``python``/``pip`` resolve to
Hermes's own interpreter — a bare interpreter whose site-packages the next
repair deletes. Foreground (``_make_run_env``) and background
(``ProcessRegistry._spawn_env``) terminal spawns strip those dirs; other store
tools (node, git, rg, ffmpeg) stay.
"""

import os

import pytest

from tools.environments import local as local_mod
from tools.environments.local import (
    _make_run_env,
    _store_python_path_dirs,
    _strip_store_python_dirs,
    _strip_store_python_dirs_from_env,
)

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def store_dirs(monkeypatch):
    """Pretend the store python owns one bin dir (never touches a real store)."""
    monkeypatch.setattr(
        local_mod, "_store_python_path_dirs",
        lambda: ["/store/python-3.14.7/bin"])
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
    return ["/store/python-3.14.7/bin"]


class TestStorePythonPathDirs:
    def test_empty_when_python_not_installed(self, monkeypatch):
        import pm

        monkeypatch.setattr(pm, "env_for", lambda *a, **k: {"PATH": ""})
        assert _store_python_path_dirs() == []

    def test_parses_composed_path(self, monkeypatch):
        import pm

        monkeypatch.setattr(
            pm, "env_for",
            lambda *a, **k: {"PATH": "/s/python-3.14/bin:/usr/bin"})
        assert _store_python_path_dirs() == ["/s/python-3.14/bin", "/usr/bin"]

    def test_never_raises(self, monkeypatch):
        import pm

        def _boom(*a, **k):
            raise RuntimeError("lockfile unreadable")

        monkeypatch.setattr(pm, "env_for", _boom)
        assert _store_python_path_dirs() == []


class TestStripStorePythonDirs:
    def test_removes_store_dir_keeps_order(self, store_dirs):
        assert _strip_store_python_dirs(
            "/store/python-3.14.7/bin:/usr/bin:/bin") == "/usr/bin:/bin"

    def test_noop_without_store_python(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_store_python_path_dirs", lambda: [])
        path = "/usr/bin:/bin"
        assert _strip_store_python_dirs(path) == path

    def test_hermes_bin_dir_is_exempt(self, monkeypatch):
        monkeypatch.setattr(
            local_mod, "_store_python_path_dirs",
            lambda: ["/store/python-3.14.7/bin"])
        monkeypatch.setattr(
            local_mod, "_resolve_hermes_bin_dir",
            lambda: "/store/python-3.14.7/bin")
        path = "/store/python-3.14.7/bin:/usr/bin"
        assert _strip_store_python_dirs(path) == path

    def test_case_insensitive_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(
            local_mod, "_store_python_path_dirs", lambda: ["C:\\Store\\PY\\bin"])
        monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
        orig_normcase, orig_pathsep = os.path.normcase, os.pathsep
        os.path.normcase = lambda s: s.lower().replace("/", "\\")
        os.pathsep = ";"
        try:
            assert _strip_store_python_dirs(
                "c:\\store\\py\\bin;C:\\Windows") == "C:\\Windows"
        finally:
            os.path.normcase, os.pathsep = orig_normcase, orig_pathsep

    def test_from_env_dict(self, store_dirs):
        env = {"PATH": "/store/python-3.14.7/bin:/usr/bin", "OTHER": "1"}
        out = _strip_store_python_dirs_from_env(env)
        assert out["PATH"] == "/usr/bin"
        assert out["OTHER"] == "1"

    def test_from_env_dict_missing_path(self, store_dirs):
        assert _strip_store_python_dirs_from_env({"OTHER": "1"}) == {"OTHER": "1"}


class TestMakeRunEnv:
    def test_store_python_off_terminal_path(self, store_dirs, monkeypatch):
        monkeypatch.setenv("PATH", "/store/python-3.14.7/bin:/usr/bin:/bin")
        env = _make_run_env({})
        entries = env["PATH"].split(os.pathsep)
        assert "/store/python-3.14.7/bin" not in entries
        assert "/usr/bin" in entries and "/bin" in entries

    def test_other_store_tools_stay(self, monkeypatch):
        monkeypatch.setattr(
            local_mod, "_store_python_path_dirs",
            lambda: ["/store/python-3.14.7/bin"])
        monkeypatch.setattr(
            local_mod, "_managed_runtime_path_entries",
            lambda: ["/store/node-26/bin"])
        monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
        monkeypatch.setenv(
            "PATH", "/store/python-3.14.7/bin:/usr/bin:/bin")
        env = _make_run_env({})
        entries = env["PATH"].split(os.pathsep)
        assert "/store/python-3.14.7/bin" not in entries
        assert "/store/node-26/bin" in entries

    def test_hermes_bin_still_prepended(self, store_dirs, monkeypatch, tmp_path):
        monkeypatch.setattr(
            local_mod, "_resolve_hermes_bin_dir", lambda: str(tmp_path))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        env = _make_run_env({})
        assert env["PATH"].split(os.pathsep)[0] == str(tmp_path)


class TestSpawnEnv:
    def test_background_spawn_strips_store_python(self, store_dirs, monkeypatch):
        from tools.process_registry import ProcessRegistry

        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        env = ProcessRegistry._spawn_env(
            {"PATH": "/store/python-3.14.7/bin:/usr/bin:/bin"})
        assert "/store/python-3.14.7/bin" not in env["PATH"].split(os.pathsep)
        assert "/usr/bin" in env["PATH"].split(os.pathsep)
        assert env["PYTHONUNBUFFERED"] == "1"
