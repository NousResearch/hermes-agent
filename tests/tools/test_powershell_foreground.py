"""Native Windows tests for synchronous PowerShell foreground execution."""

from __future__ import annotations

import os
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.session_context import clear_session_vars, set_session_vars


pytestmark = [
    pytest.mark.windows_only,
    pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is not installed"),
]


def _pwsh_env(cwd: Path, *, timeout: int = 10):
    from tools.environments.local import LocalEnvironment

    return LocalEnvironment(cwd=str(cwd), timeout=timeout)


@contextmanager
def _pwsh_selection():
    from tools.environments.shell_selection import _clear_active_shell_name_cache

    _clear_active_shell_name_cache()
    try:
        with patch.dict(
            os.environ,
            {"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"},
        ):
            yield
    finally:
        _clear_active_shell_name_cache()


def test_pwsh_foreground_executes_utf8_and_returns_zero(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute("Write-Output '中文 café'")
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == "中文 café"


def test_pwsh_selection_is_fixed_for_active_environment(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        os.environ["TERMINAL_SHELL"] = "bash"
        try:
            result = env.execute("Write-Output $PSVersionTable.PSEdition")
        finally:
            env.cleanup()

    assert env.shell_name == "pwsh"
    assert result["returncode"] == 0
    assert result["output"].strip() == "Core"


def test_pwsh_preserves_native_program_exit_code(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute("python -c \"import sys; sys.exit(7)\"")
        finally:
            env.cleanup()

    assert result["returncode"] == 7


def test_pwsh_explicit_exit_code_is_preserved(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute("exit 7")
        finally:
            env.cleanup()

    assert result["returncode"] == 7


def test_pwsh_native_argument_passing_preserves_values(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute(
                "python -c \"import json,sys; "
                "print(json.dumps(sys.argv[1:], ensure_ascii=False))\" "
                "'space value' '中文$literal' 'quote\"value'"
            )
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert json.loads(result["output"]) == [
        "space value",
        "中文$literal",
        'quote"value',
    ]


def test_pwsh_success_after_native_failure_returns_success(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute(
                "python -c \"import sys; sys.exit(7)\"\nWrite-Output 'recovered'"
            )
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == "recovered"


def test_pwsh_here_string_does_not_break_wrapper(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute("$value = @'\n{ literal }\n'@\n$value")
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == "{ literal }"


def test_pwsh_payload_cannot_overwrite_wrapper_status(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute(
                "$__hermesExit = 99\n"
                "$__hermesCommandSucceeded = $false\n"
                "$__hermesCommandNativeExit = 88\n"
                "Write-Output safe"
            )
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == "safe"


def test_pwsh_payload_cannot_hijack_wrapper_cmdlets(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute(
                "function global:ConvertTo-Json { 'not-json' }\n"
                "function global:Get-ChildItem { throw 'hijacked' }\n"
                "function global:Get-Location { throw 'hijacked' }\n"
                "Write-Output safe"
            )
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == "safe"


def test_pwsh_trailing_backtick_cannot_consume_wrapper_trailer(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute("Write-Output safe`")
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == "safe"
    assert "__hermes" not in result["output"]


@pytest.mark.parametrize(
    "command",
    [
        "if ($true) { Write-Output safe",
        "$value = @'\nunterminated",
    ],
)
def test_pwsh_incomplete_payload_is_rejected_before_execution(tmp_path, command):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute(command)
        finally:
            env.cleanup()

    assert result["returncode"] == 1
    assert "__hermes" not in result["output"]


def test_pwsh_cmdlet_failure_returns_one(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute("Write-Error 'expected failure'")
        finally:
            env.cleanup()

    assert result["returncode"] == 1
    assert "expected failure" in result["output"]


def test_pwsh_stdin_round_trip_is_utf8(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            result = env.execute(
                "$s = [Console]::In.ReadToEnd(); [Console]::Out.Write($s)",
                stdin_data="输入 café\n",
            )
        finally:
            env.cleanup()

    assert result["returncode"] == 0
    assert result["output"] == "输入 café\n"


def test_pwsh_cwd_persists_across_commands(tmp_path):
    child = tmp_path / "child with spaces"
    child.mkdir()
    escaped = str(child).replace("'", "''")

    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            changed = env.execute(f"Set-Location -LiteralPath '{escaped}'")
            observed = env.execute("(Get-Location).ProviderPath")
        finally:
            env.cleanup()

    assert changed["returncode"] == 0
    assert Path(env.cwd).resolve() == child.resolve()
    assert Path(observed["output"].strip()).resolve() == child.resolve()


def test_pwsh_environment_set_and_remove_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PWSH_REMOVE_ME", "host-value")

    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        try:
            set_result = env.execute("$env:HERMES_PWSH_STATE = '中文 value'")
            observed = env.execute("$env:HERMES_PWSH_STATE")
            removed = env.execute("Remove-Item Env:HERMES_PWSH_REMOVE_ME")
            observed_removed = env.execute(
                "if (Test-Path Env:HERMES_PWSH_REMOVE_ME) { 'present' } else { 'absent' }"
            )
        finally:
            env.cleanup()

    assert set_result["returncode"] == 0
    assert observed["output"].strip() == "中文 value"
    assert removed["returncode"] == 0
    assert observed_removed["output"].strip() == "absent"


def test_pwsh_session_identity_is_reinjected_not_tombstoned(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        first_tokens = set_session_vars(session_key="session-A")
        try:
            first = env.execute("$env:HERMES_SESSION_KEY")
        finally:
            clear_session_vars(first_tokens)
        second_tokens = set_session_vars(session_key="session-B")
        try:
            second = env.execute("$env:HERMES_SESSION_KEY")
        finally:
            clear_session_vars(second_tokens)
            env.cleanup()

    assert first["output"].strip() == "session-A"
    assert second["output"].strip() == "session-B"


def test_pwsh_timeout_uses_shared_foreground_lifecycle(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path, timeout=1)
        try:
            result = env.execute("Start-Sleep -Seconds 30")
        finally:
            env.cleanup()

    assert result["returncode"] == 124
    assert "timed out" in result["output"]


def test_pwsh_timeout_terminates_descendant_processes(tmp_path):
    marker = tmp_path / "child-survived.txt"
    child_code = (
        "import pathlib,time; time.sleep(3); "
        f"pathlib.Path({str(marker)!r}).write_text('survived')"
    )
    command = (
        "$p = Start-Process python -PassThru -ArgumentList "
        f"@('-c', {json.dumps(child_code)}); Wait-Process -Id $p.Id"
    )

    with _pwsh_selection():
        env = _pwsh_env(tmp_path, timeout=1)
        try:
            result = env.execute(command)
        finally:
            env.cleanup()

    assert result["returncode"] == 124
    time.sleep(3.5)
    assert not marker.exists(), "PowerShell descendant survived timeout cleanup"


def test_pwsh_temp_scripts_are_removed(tmp_path):
    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        temp_root = Path(env.get_temp_dir())
        before = {p.name for p in temp_root.glob("hermes-pwsh-*")}
        try:
            result = env.execute("Write-Output done")
        finally:
            env.cleanup()
        after = {p.name for p in temp_root.glob("hermes-pwsh-*")}

    assert result["returncode"] == 0
    assert after == before


def test_pwsh_spawn_failure_removes_all_temp_material(tmp_path):
    import tools.environments.local as local_module

    with _pwsh_selection():
        env = _pwsh_env(tmp_path)
        temp_root = Path(env.get_temp_dir())
        before = {p.name for p in temp_root.glob("hermes-pwsh-*")}
        try:
            with patch.object(
                local_module.subprocess,
                "Popen",
                side_effect=OSError("expected spawn failure"),
            ):
                with pytest.raises(OSError, match="expected spawn failure"):
                    env.execute("Write-Output never-runs")
        finally:
            env.cleanup()
        after = {p.name for p in temp_root.glob("hermes-pwsh-*")}

    assert after == before
