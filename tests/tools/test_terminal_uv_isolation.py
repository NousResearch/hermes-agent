"""The terminal target never receives Hermes' private uv toolchain."""

import os

from tools.environments.local import _make_run_env


def test_private_uv_directory_is_absent_from_terminal_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("tools.environments.local._managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr("tools.environments.local._resolve_hermes_bin_dir", lambda: None)

    path = _make_run_env({"PATH": "/usr/bin:/bin"})["PATH"].split(os.pathsep)

    assert str(tmp_path / ".hermes" / "uv") not in path


def test_target_uv_environment_does_not_inject_internal_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("tools.environments.local._managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr("tools.environments.local._resolve_hermes_bin_dir", lambda: None)

    env = _make_run_env({"PATH": "/usr/bin:/bin", "UV_CACHE_DIR": "/workspace/cache"})

    assert env["UV_CACHE_DIR"] == "/workspace/cache"
