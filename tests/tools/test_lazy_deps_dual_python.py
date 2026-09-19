"""Lazy installs must land in the venv the process imports from (#88355).

Hermes Studio Desktop ships a dual-Python bundle: ``python/base/python.exe``
launches the gateway, ``python/venv`` holds the plugin code and its
site-packages, and the launcher wires the two together with ``PYTHONPATH``
(it never exports ``VIRTUAL_ENV``). In that layout ``sys.executable``,
``sys.prefix`` and ``sys.base_prefix`` all name ``base``, so an installer
that derives its target from ``sys.executable`` writes into
``base/Lib/site-packages`` — a directory nothing on the venv's ``sys.path``
can see. Every plugin that lazy-installs a dependency then fails to import
the package it just installed.

These tests pin the resolution ladder and, more importantly, the command the
installer actually spawns.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_constants import venv_python_path
from tools import lazy_deps as ld


def _make_venv(root: Path) -> Path:
    """Create a minimally-real venv on disk: pyvenv.cfg, bin dir, python."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    python = venv_python_path(root)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    return python


def _site_packages(root: Path) -> Path:
    """The venv's site-packages, in this host's venv layout."""
    if os.name == "nt":
        sp = root / "Lib" / "site-packages"
    else:
        sp = root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    return sp


@pytest.fixture
def dual_python(tmp_path, monkeypatch):
    """Simulate the Studio bundle and return (base_dir, venv_dir).

    ``base`` is a plain (non-venv) interpreter directory with a *sibling*
    ``venv`` it must never be confused with; only the venv's site-packages
    is on ``sys.path``, exactly as the desktop launcher arranges it.
    """
    base = tmp_path / "python" / "base"
    base_python = base / ("python.exe" if os.name == "nt" else "python")
    base_python.parent.mkdir(parents=True, exist_ok=True)
    base_python.write_text("", encoding="utf-8")

    venv = tmp_path / "python" / "venv"
    _make_venv(venv)
    venv_sp = _site_packages(venv)

    # The base interpreter is not itself a venv: prefix == base_prefix.
    monkeypatch.setattr(sys, "executable", str(base_python))
    monkeypatch.setattr(sys, "prefix", str(base))
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setattr(sys, "path", [str(tmp_path / "src"), str(venv_sp)])
    # The launcher exports PYTHONPATH and PATH, never VIRTUAL_ENV.
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
    return base, venv


# ---------------------------------------------------------------------------
# Resolution ladder
# ---------------------------------------------------------------------------


class TestDualPythonResolution:
    def test_resolves_the_venv_not_the_base_dir(self, dual_python):
        base, venv = dual_python
        assert ld._active_venv_root() == venv
        assert ld._active_venv_root() != base

    def test_pip_interpreter_is_the_venv_python(self, dual_python):
        _base, venv = dual_python
        assert ld._venv_python(ld._active_venv_root()) == str(venv_python_path(venv))
        assert ld._venv_python(ld._active_venv_root()) != sys.executable

    def test_base_dir_without_a_venv_falls_back_unchanged(self, tmp_path, monkeypatch):
        # No pyvenv.cfg anywhere on sys.path (system python, --user install):
        # behaviour must stay exactly what it was before the fix.
        base = tmp_path / "usr" / "bin"
        base.mkdir(parents=True)
        exe = base / "python"
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "executable", str(exe))
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "usr"))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "usr"))
        monkeypatch.setattr(sys, "path", [str(tmp_path / "home" / ".local" / "lib" / "python3.12" / "site-packages")])
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        assert ld._active_venv_root() == tmp_path / "usr"

    def test_running_venv_wins_over_a_stale_virtual_env(self, tmp_path, monkeypatch):
        # A leftover VIRTUAL_ENV export must not redirect installs away from
        # the venv this interpreter actually imports from.
        running = tmp_path / "running"
        _make_venv(running)
        stale = tmp_path / "stale"
        _make_venv(stale)
        monkeypatch.setattr(sys, "prefix", str(running))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "sysbase"))
        monkeypatch.setenv("VIRTUAL_ENV", str(stale))
        assert ld._active_venv_root() == running

    def test_virtual_env_used_when_the_interpreter_is_not_in_a_venv(self, tmp_path, monkeypatch):
        # `uv run` activates a venv without changing sys.prefix.
        venv = tmp_path / "uv-env"
        _make_venv(venv)
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "sysbase"))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "sysbase"))
        monkeypatch.setattr(sys, "path", [])
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        assert ld._active_venv_root() == venv


# ---------------------------------------------------------------------------
# The command actually spawned
# ---------------------------------------------------------------------------


class TestDualPythonInstallCommand:
    def test_pip_tier_installs_with_the_venv_interpreter(self, dual_python, monkeypatch):
        _base, venv = dual_python
        # No uv anywhere -> the pip tier is what runs.
        monkeypatch.setattr("hermes_cli.managed_uv.resolve_uv", lambda: None)
        monkeypatch.setattr(ld.shutil, "which", lambda _n: None)
        monkeypatch.setattr(ld, "_warm_installed_bytecode", lambda *_a, **_k: None)

        captured = {}

        def fake_run(cmd, *a, **k):
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 25.0.1", "")
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(("yantrikdb==0.13.0",))
        assert result.success
        assert captured["cmd"][0] == str(venv_python_path(venv))
        assert captured["cmd"][0] != sys.executable

    def test_uv_tier_points_virtual_env_at_the_venv(self, dual_python, monkeypatch):
        _base, venv = dual_python
        monkeypatch.setattr("hermes_cli.managed_uv.resolve_uv", lambda: "/usr/bin/uv")
        monkeypatch.setattr(ld, "_warm_installed_bytecode", lambda *_a, **_k: None)

        captured = {}

        def fake_run(cmd, *a, **k):
            captured["cmd"] = cmd
            captured["env"] = k.get("env") or {}
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(("yantrikdb==0.13.0",))
        assert result.success
        assert captured["cmd"][:3] == ["/usr/bin/uv", "pip", "install"]
        assert captured["env"]["VIRTUAL_ENV"] == str(venv)

    def test_durable_target_mode_still_redirects_to_the_target(
        self, dual_python, tmp_path, monkeypatch
    ):
        # The immutable-image path is orthogonal to which interpreter drives
        # pip: --target still wins, and sys.path activation still happens.
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        monkeypatch.setattr("hermes_cli.managed_uv.resolve_uv", lambda: None)
        monkeypatch.setattr(ld.shutil, "which", lambda _n: None)
        monkeypatch.setattr(ld, "_warm_installed_bytecode", lambda *_a, **_k: None)
        activated = []
        monkeypatch.setattr(ld, "_activate_target_on_syspath", activated.append)

        captured = {}

        def fake_run(cmd, *a, **k):
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 25.0.1", "")
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(("yantrikdb==0.13.0",))
        assert result.success
        assert "--target" in captured["cmd"]
        assert str(target) in captured["cmd"]
        assert activated == [target]
