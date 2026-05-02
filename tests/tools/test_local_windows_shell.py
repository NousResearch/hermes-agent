import base64
from unittest.mock import patch

import pytest
import sys
from pathlib import Path

import tools.environments.local as local_env


@pytest.fixture(autouse=True)
def _restore_windows_flag():
    original = local_env._IS_WINDOWS
    yield
    local_env._IS_WINDOWS = original


def test_find_bash_prefers_git_bash_over_wsl_launcher(monkeypatch):
    local_env._IS_WINDOWS = True
    git_bash = r"C:\Program Files\Git\bin\bash.exe"

    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Alice\AppData\Local")
    monkeypatch.setattr(local_env.shutil, "which", lambda _name: r"C:\Windows\System32\bash.exe")
    monkeypatch.setattr(local_env.os.path, "isfile", lambda path: path == git_bash)

    assert local_env._find_bash() == git_bash


def test_find_bash_rejects_wsl_launcher_without_git_bash(monkeypatch):
    local_env._IS_WINDOWS = True

    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Alice\AppData\Local")
    monkeypatch.setattr(local_env.shutil, "which", lambda _name: r"C:\Windows\System32\bash.exe")
    monkeypatch.setattr(local_env.os.path, "isfile", lambda _path: False)

    with pytest.raises(RuntimeError, match="Git Bash not found"):
        local_env._find_bash()


def _decode_encoded_command(wrapped: str) -> str:
    encoded = wrapped.rsplit(" ", 1)[-1]
    return base64.b64decode(encoded).decode("utf-16le")


def test_windows_wraps_direct_powershell_cmdlets(monkeypatch):
    local_env._IS_WINDOWS = True
    monkeypatch.setattr(local_env, "_find_powershell", lambda: "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")

    command = "Get-ChildItem | Select-Object -First 1 -ExpandProperty Name"
    wrapped = local_env._wrap_windows_powershell_command(command)

    assert "powershell.exe" in wrapped
    assert "-EncodedCommand" in wrapped
    decoded = _decode_encoded_command(wrapped)
    assert "$ProgressPreference='SilentlyContinue'" in decoded
    assert decoded.endswith(command)


def test_windows_wraps_powershell_variables_and_dotnet_calls(monkeypatch):
    local_env._IS_WINDOWS = True
    monkeypatch.setattr(local_env, "_find_powershell", lambda: "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")

    assert _decode_encoded_command(
        local_env._wrap_windows_powershell_command("$PSVersionTable.PSVersion.ToString()")
    ).endswith("$PSVersionTable.PSVersion.ToString()")
    assert _decode_encoded_command(
        local_env._wrap_windows_powershell_command("[Environment]::GetEnvironmentVariable('Path', 'User')")
    ).endswith("[Environment]::GetEnvironmentVariable('Path', 'User')")


def test_windows_does_not_wrap_bash_or_explicit_shell_commands(monkeypatch):
    local_env._IS_WINDOWS = True
    monkeypatch.setattr(local_env, "_find_powershell", lambda: "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")

    bash_command = "[ -f pyproject.toml ] && echo yes"
    assert local_env._wrap_windows_powershell_command("git status --short") == "git status --short"
    assert local_env._wrap_windows_powershell_command("powershell -Command Get-ChildItem") == "powershell -Command Get-ChildItem"
    assert local_env._wrap_windows_powershell_command(bash_command) == bash_command


def test_find_powershell_prefers_newest_stable_pwsh(monkeypatch):
    local_env._IS_WINDOWS = True
    monkeypatch.delenv("HERMES_POWERSHELL_PATH", raising=False)
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("SystemRoot", r"C:\Windows")

    existing = {
        r"C:\Program Files\PowerShell\6\pwsh.exe",
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        r"C:\Program Files\PowerShell\7-preview\pwsh.exe",
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    }

    monkeypatch.setattr(local_env.os.path, "isdir", lambda path: path == r"C:\Program Files\PowerShell")
    monkeypatch.setattr(local_env.os, "listdir", lambda path: ["6", "7-preview", "7"] if path == r"C:\Program Files\PowerShell" else [])
    monkeypatch.setattr(local_env.os.path, "isfile", lambda path: path in existing)
    monkeypatch.setattr(
        local_env.shutil,
        "which",
        lambda name: r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        if name in {"powershell", "powershell.exe"} else None,
    )

    assert local_env._find_powershell() == "C:/Program Files/PowerShell/7/pwsh.exe"


def test_find_powershell_allows_explicit_override(monkeypatch):
    local_env._IS_WINDOWS = True
    override = r"D:\Tools\PowerShell\9\pwsh.exe"
    monkeypatch.setenv("HERMES_POWERSHELL_PATH", override)
    monkeypatch.setattr(local_env.os.path, "isfile", lambda path: path == override)

    assert local_env._find_powershell() == "D:/Tools/PowerShell/9/pwsh.exe"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only live regression")
def test_local_environment_captures_stdout_on_windows():
    try:
        local_env._find_bash()
    except RuntimeError:
        pytest.skip("Git Bash not installed")

    env = local_env.LocalEnvironment(cwd=str(Path.cwd()), timeout=10)
    try:
        result = env.execute("echo WINDOWS_STDOUT_OK")
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert "WINDOWS_STDOUT_OK" in result["output"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only live regression")
def test_local_environment_runs_direct_powershell_cmdlets_on_windows():
    try:
        local_env._find_bash()
        local_env._find_powershell()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = local_env.LocalEnvironment(cwd=str(Path.cwd()), timeout=10)
    try:
        result = env.execute(
            "Get-ChildItem | Select-Object -First 1 -ExpandProperty Name",
            timeout=10,
        )
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert "command not found" not in result["output"]
    assert "#< CLIXML" not in result["output"]
    assert result["output"].strip()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only live regression")
def test_local_environment_prefers_pwsh_on_windows_when_available():
    if not local_env.shutil.which("pwsh"):
        pytest.skip("PowerShell 7+ not installed")
    try:
        local_env._find_bash()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = local_env.LocalEnvironment(cwd=str(Path.cwd()), timeout=10)
    try:
        result = env.execute("$PSVersionTable.PSVersion.Major", timeout=10)
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert "#< CLIXML" not in result["output"]
    assert int(result["output"].strip().splitlines()[-1]) >= 7
