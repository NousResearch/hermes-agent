"""The terminal sandbox is uv-capable without polluting the user's uv state.

The terminal subshell PATH gets ``<default-root>/uv`` appended so the model can
run ``uv`` on a managed-only install, with a user's own uv winning
(first-occurrence-wins).  These tests pin the contract of the follow-up:
:func:`tools.environments.local._pin_managed_uv_state_when_managed_uv_would_run`
must pin uv's write dirs into Hermes' tree **only** when the managed copy is the
one that would actually run — a user's own uv keeps its own state.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

from tools.environments.local import _pin_managed_uv_state_when_managed_uv_would_run

_EXE = ".exe" if sys.platform == "win32" else ""


def _make_uv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho uv 0.1.2\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_pins_when_managed_uv_is_the_only_uv(tmp_path, monkeypatch):
    root = tmp_path / "root"
    home = tmp_path / "home"
    managed = root / "uv"
    _make_uv(managed / f"uv{_EXE}")
    empty = tmp_path / "empty"
    empty.mkdir()

    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("hermes_cli.managed_uv.get_hermes_home", lambda: home)

    run_env = {"PATH": f"{empty}{os.pathsep}{managed}"}
    _pin_managed_uv_state_when_managed_uv_would_run(run_env)

    assert run_env["UV_CACHE_DIR"] == str(home / "cache" / "uv")
    assert run_env["UV_TOOL_DIR"] == str(home / "uv" / "tools")
    assert run_env["UV_TOOL_BIN_DIR"] == str(home / "bin")


def test_user_uv_wins_and_keeps_its_state(tmp_path, monkeypatch):
    root = tmp_path / "root"
    home = tmp_path / "home"
    managed = root / "uv"
    _make_uv(managed / f"uv{_EXE}")
    user = tmp_path / "userbin"
    _make_uv(user / f"uv{_EXE}")

    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("hermes_cli.managed_uv.get_hermes_home", lambda: home)

    run_env = {"PATH": f"{user}{os.pathsep}{managed}"}
    _pin_managed_uv_state_when_managed_uv_would_run(run_env)

    assert "UV_CACHE_DIR" not in run_env
    assert "UV_TOOL_DIR" not in run_env


def test_noop_when_managed_uv_dir_absent(tmp_path, monkeypatch):
    root = tmp_path / "root"
    home = tmp_path / "home"
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("hermes_cli.managed_uv.get_hermes_home", lambda: home)

    run_env = {"PATH": str(tmp_path / "somewhere")}
    _pin_managed_uv_state_when_managed_uv_would_run(run_env)

    assert "UV_CACHE_DIR" not in run_env
