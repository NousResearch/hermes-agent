"""Regression for #119035: child commands must bypass the bare Python entrypoint."""

import os
import subprocess
import sys

import pytest

from tools.environments import local

pytestmark = pytest.mark.platforms("posix")


@pytest.mark.parametrize("path_candidate", ["missing", "entrypoint", "console_script"])
def test_child_hermes_uses_launchable_command(monkeypatch, tmp_path, path_candidate):
    root = tmp_path / "install"
    root.mkdir()
    entrypoint = root / "hermes"
    entrypoint.write_text("#!/usr/bin/env python3\nraise RuntimeError('bare entrypoint')\n", encoding="utf-8")
    entrypoint.chmod(0o755)
    venv_bin = root / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    console_script = venv_bin / "hermes"
    console_script.write_text(f"#!{sys.executable}\nprint('venv console script')\n", encoding="utf-8")
    console_script.chmod(0o755)
    other_bin = tmp_path / "otherbin"
    other_bin.mkdir()
    shim = other_bin / "hermes"
    shim.write_text("#!/bin/sh\nprintf 'PATH shim\\n'\n", encoding="utf-8")
    shim.chmod(0o755)
    path_dir = {"missing": None, "entrypoint": root, "console_script": other_bin}[path_candidate]
    path = os.pathsep.join([*([str(path_dir)] if path_dir else []), "/usr/bin", "/bin"])
    monkeypatch.setenv("PATH", path)
    monkeypatch.setattr(sys, "argv", [str(entrypoint), "serve"])
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
    monkeypatch.setattr(local, "_HERMES_BIN_DIR", local._SENTINEL)
    monkeypatch.setattr(local, "_HERMES_BIN_DIR_IS_PAYLOAD", False)

    env = local.build_subprocess_env()
    result = subprocess.run(["hermes"], env=env, capture_output=True, text=True, timeout=10)

    expected_dir = other_bin if path_candidate == "console_script" else venv_bin
    expected_output = "PATH shim" if path_candidate == "console_script" else "venv console script"
    assert env["PATH"].split(os.pathsep)[0] == str(expected_dir)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output
    assert local.build_subprocess_env(env)["PATH"].split(os.pathsep).count(str(expected_dir)) == 1


def test_launchable_hermes_requires_executable_non_entrypoint(tmp_path):
    assert not local._launchable_hermes(str(tmp_path))
    candidate = tmp_path / "hermes"
    candidate.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    candidate.chmod(0o755)
    assert not local._launchable_hermes(str(tmp_path))
    candidate.write_text(f"#!{sys.executable}\n", encoding="utf-8")
    assert local._launchable_hermes(str(tmp_path))
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    assert local._launchable_hermes(str(tmp_path))
    candidate.chmod(0o644)
    assert not local._launchable_hermes(str(tmp_path))
