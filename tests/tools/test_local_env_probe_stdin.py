"""Local-env Windows probes must not inherit the caller's stdin (#78820).

When the TUI gateway spawns a local environment, PowerShell/Git-Bash probes
run with capture_output=True. Without stdin=DEVNULL the child inherits the
gateway's Node→Python command pipe; on Windows that can corrupt the pipe and
kill the gateway on the next readline with OSError(EINVAL).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import tools.environments.local as local_env


def _local_py_source() -> str:
    return Path(local_env.__file__).read_text(encoding="utf-8")


def test_mandatory_aslr_probe_source_uses_devnull_stdin():
    """Source guard: ASLR PowerShell probe detaches stdin."""
    src = _local_py_source()
    # Narrow window around the ForceRelocateImages probe body.
    marker = "ForceRelocateImages.ToString()"
    idx = src.index(marker)
    window = src[idx : idx + 400]
    assert "stdin=subprocess.DEVNULL" in window


def test_bash_starts_probe_source_uses_devnull_stdin():
    """Source guard: bash external-program probe detaches stdin."""
    src = _local_py_source()
    marker = "_BASH_EXTERNAL_PROGRAM_PROBE"
    # Second occurrence is inside _bash_starts's subprocess.run call.
    first = src.index(marker)
    second = src.index(marker, first + 1)
    window = src[second : second + 350]
    assert "stdin=subprocess.DEVNULL" in window


def test_bash_starts_passes_devnull_at_runtime(monkeypatch):
    """Runtime: _bash_starts forwards stdin=DEVNULL into subprocess.run."""
    local_env._bash_starts_cache.clear()
    local_env._bash_probe_details_cache.clear()

    calls: list[dict] = []

    def _fake_run(*args, **kwargs):
        calls.append(kwargs)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(local_env.subprocess, "run", _fake_run)
    assert local_env._bash_starts("/usr/bin/bash") is True
    assert calls, "subprocess.run was not called"
    assert calls[0].get("stdin") is subprocess.DEVNULL


def test_mandatory_aslr_passes_devnull_at_runtime(monkeypatch):
    """Runtime: _mandatory_aslr_enabled forwards stdin=DEVNULL on Windows path."""
    local_env._mandatory_aslr_enabled_cache = None

    calls: list[dict] = []

    def _fake_run(*args, **kwargs):
        calls.append(kwargs)
        return MagicMock(returncode=0, stdout="OFF\n", stderr="")

    monkeypatch.setattr(local_env.subprocess, "run", _fake_run)
    monkeypatch.setattr(local_env.shutil, "which", lambda _name: "powershell.exe")
    # Force the Windows probe path even on Linux CI hosts.
    monkeypatch.setattr(local_env, "_IS_WINDOWS", True, raising=False)

    result = local_env._mandatory_aslr_enabled()
    assert result is False
    assert calls, "subprocess.run was not called"
    assert calls[0].get("stdin") is subprocess.DEVNULL
