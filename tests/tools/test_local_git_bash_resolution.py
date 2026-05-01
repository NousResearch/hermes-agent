"""Regression: Windows PATH can put Microsoft\\WindowsApps\\bash.exe ahead of Git Bash."""

import os
from unittest.mock import patch

import pytest

from tools.environments import local as loc


@pytest.mark.parametrize(
    ("sample", "want_shim"),
    [
        (r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\bash.exe", True),
        (r"C:\Program Files\Git\bin\bash.exe", False),
    ],
)
def test_windows_store_bash_shim_flag(sample: str, want_shim: bool) -> None:
    assert loc._windows_store_bash_shim(sample) is want_shim


def test_find_bash_git_default_paths_before_windowsapps_which(tmp_path, monkeypatch) -> None:
    """Issue #18454: ``shutil.which`` can resolve to the stub; prefer Git-for-Windows paths."""
    program_files = tmp_path / "Program Files"
    git_bash = program_files / "Git" / "bin" / "bash.exe"
    git_bash.parent.mkdir(parents=True)
    git_bash.write_bytes(b"")

    shim_dir = tmp_path / "Microsoft" / "WindowsApps"
    shim_dir.mkdir(parents=True)
    shim_exe = shim_dir / "bash.exe"
    shim_exe.write_bytes(b"")

    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "Program Files (x86)"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("PATH", str(shim_dir))
    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)

    def _which(cmd: str):
        return str(shim_exe) if cmd == "bash" else None

    monkeypatch.setattr(loc.shutil, "which", _which)

    with patch("tools.environments.local._IS_WINDOWS", True):
        resolved = loc._find_bash()

    assert os.path.normcase(resolved) == os.path.normcase(str(git_bash))


def test_find_bash_path_walk_skips_shim_when_which_returns_stub(tmp_path, monkeypatch) -> None:
    shim_dir = tmp_path / "Microsoft" / "WindowsApps"
    shim_dir.mkdir(parents=True)
    (shim_dir / "bash.exe").write_bytes(b"")

    portable = tmp_path / "portable" / "bin"
    portable.mkdir(parents=True)
    good = portable / "bash.exe"
    good.write_bytes(b"")

    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "pfx86"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("PATH", os.pathsep.join([str(shim_dir), str(portable)]))
    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)

    def _which(cmd: str):
        return str(shim_dir / "bash.exe") if cmd == "bash" else None

    monkeypatch.setattr(loc.shutil, "which", _which)

    with patch("tools.environments.local._IS_WINDOWS", True):
        resolved = loc._find_bash()

    assert resolved == str(good)


def test_find_bash_raises_when_only_shims_exist(tmp_path, monkeypatch) -> None:
    shim_dir = tmp_path / "Microsoft" / "WindowsApps"
    shim_dir.mkdir(parents=True)
    (shim_dir / "bash.exe").write_bytes(b"")

    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "pfx86"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("PATH", str(shim_dir))
    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)

    def _which(cmd: str):
        return str(shim_dir / "bash.exe") if cmd == "bash" else None

    monkeypatch.setattr(loc.shutil, "which", _which)

    with patch("tools.environments.local._IS_WINDOWS", True):
        with pytest.raises(RuntimeError, match="Git Bash not found"):
            loc._find_bash()
