"""Windows command payload transport tests for ``LocalEnvironment``."""

import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.environments import local as local_mod
from tools.environments.local import LocalEnvironment, _bash_safe_path


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    with patch.object(
        LocalEnvironment, "init_session", autospec=True, return_value=None
    ):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    yield env
    env.cleanup()


@pytest.mark.windows_only
def test_generated_script_body_is_not_in_windows_process_argv(local_env, monkeypatch):
    marker = "HERMES_ARGV_PAYLOAD_MARKER_7f2d"
    script = f"printf '%s' '{marker}'\n" + ("# padding\n" * 1200)
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = list(args)
        argv_text = "\0".join(args)
        if marker not in argv_text:
            script_path = Path(local_mod._msys_to_windows_path(args[-1]))
            captured["bytes"] = script_path.read_bytes()
        return SimpleNamespace(pid=12345, stdin=None, stdout=None)

    monkeypatch.setattr(
        local_mod,
        "_find_bash",
        lambda: r"C:\Program Files\Git\bin\bash.exe",
    )
    monkeypatch.setattr(local_mod.subprocess, "Popen", fake_popen)

    proc = local_env._run_bash(script)

    argv_text = "\0".join(captured["args"])
    assert marker not in argv_text
    assert script not in argv_text
    assert captured["bytes"] == script.encode("utf-8")
    local_env._discard_staged_command_script(
        getattr(proc, "_hermes_staged_command_script")
    )


@pytest.mark.windows_only
def test_staged_script_preserves_surrogateescaped_bytes(local_env):
    script = "printf byte-transparent-\udc80\udcff\n"

    path = local_env._stage_command_script(script)

    assert Path(path).read_bytes() == script.encode("utf-8", errors="surrogateescape")
    local_env._discard_staged_command_script(path)


@pytest.mark.windows_only
def test_real_git_bash_executes_payload_larger_than_cmd_limit(local_env, tmp_path):
    destination = tmp_path / "payload.bin"
    payload = "第一行 Ω\n'quoted' \"double\" C:\\path\\file\n\n末行"
    padding = "\n".join(f"# padding-{i:04d}-{'x' * 32}" for i in range(400))
    command = (
        f"{padding}\n"
        f"printf %s {shlex.quote(payload)} > "
        f"{shlex.quote(_bash_safe_path(str(destination)))}"
    )
    assert len(command) > 8191

    result = local_env.execute(command, timeout=20)

    assert result["returncode"] == 0, result
    assert destination.read_bytes() == payload.encode("utf-8")


@pytest.mark.windows_only
def test_stdin_remains_byte_exact_and_closes_with_large_staged_program(
    local_env, tmp_path
):
    destination = tmp_path / "stdin.bin"
    stdin_data = "αβγ\nquotes ' \"\nC:\\temp\\x\n\nfinal"
    padding = "\n".join(f"# stdin-padding-{i:04d}" for i in range(700))
    command = f"{padding}\ncat > {shlex.quote(_bash_safe_path(str(destination)))}"
    assert len(command) > 8191

    result = local_env.execute(command, timeout=20, stdin_data=stdin_data)

    assert result["returncode"] == 0, result
    assert destination.read_bytes() == stdin_data.encode("utf-8")


def _staged_files(env):
    return set(Path(env.get_temp_dir()).glob("hermes-command-*.sh"))


@pytest.mark.windows_only
def test_process_start_failure_removes_staged_script(local_env, monkeypatch):
    before = _staged_files(local_env)
    monkeypatch.setattr(
        local_mod,
        "_find_bash",
        lambda: r"C:\Program Files\Git\bin\bash.exe",
    )

    def fail_start(*args, **kwargs):
        raise OSError("synthetic process start failure")

    monkeypatch.setattr(local_mod.subprocess, "Popen", fail_start)

    with pytest.raises(OSError, match="synthetic process start failure"):
        local_env._run_bash("printf start")

    assert _staged_files(local_env) == before


@pytest.mark.windows_only
@pytest.mark.parametrize("command", ["printf normal", "exit 7"])
def test_normal_and_nonzero_completion_remove_staged_script(local_env, command):
    before = _staged_files(local_env)

    local_env.execute(command, timeout=10)

    assert _staged_files(local_env) == before


@pytest.mark.windows_only
def test_wait_exception_removes_staged_script(local_env, monkeypatch):
    path = local_env._stage_command_script("printf interrupted")
    proc = SimpleNamespace(_hermes_staged_command_script=path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(local_mod.BaseEnvironment, "_wait_for_process", interrupted)

    with pytest.raises(KeyboardInterrupt):
        local_env._wait_for_process(proc, timeout=1)

    assert not Path(path).exists()


@pytest.mark.windows_only
def test_environment_cleanup_removes_abandoned_staged_script(local_env):
    path = local_env._stage_command_script("printf abandoned")

    local_env.cleanup()

    assert not Path(path).exists()


@pytest.mark.windows_only
def test_failed_unlink_remains_owned_for_cleanup_retry(local_env, monkeypatch):
    path = local_env._stage_command_script("printf retry-cleanup")
    real_unlink = local_mod.os.unlink
    failed_once = False

    def transient_failure(candidate):
        nonlocal failed_once
        if candidate == path and not failed_once:
            failed_once = True
            raise PermissionError("script is still open")
        return real_unlink(candidate)

    monkeypatch.setattr(local_mod.os, "unlink", transient_failure)

    local_env._discard_staged_command_script(path)

    assert Path(path).exists()
    assert path in local_env._staged_command_scripts

    local_env.cleanup()

    assert not Path(path).exists()
    assert path not in local_env._staged_command_scripts


@pytest.mark.windows_only
def test_timeout_removes_staged_script(local_env):
    before = _staged_files(local_env)

    result = local_env.execute("sleep 5", timeout=1)

    assert result.get("timed_out") or result["returncode"] != 0
    assert _staged_files(local_env) == before


@pytest.mark.windows_only
def test_large_remote_script_reaches_fake_ssh_via_stdin_not_argv(local_env, tmp_path):
    argv_file = tmp_path / "ssh-argv.json"
    stdin_file = tmp_path / "ssh-stdin.bin"
    shim = tmp_path / "ssh"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        '"$SSH_TEST_PYTHON" -c \'import json,os,sys; '
        'open(os.environ["SSH_ARGV_FILE"],"w",encoding="utf-8").write('
        "json.dumps(sys.argv[1:])); "
        'open(os.environ["SSH_STDIN_FILE"],"wb").write(sys.stdin.buffer.read())\' '
        '"$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(shim, 0o755)

    remote_script = "Write-Output '开始'\n" + (
        "# remote-padding-xxxxxxxxxxxxxxxx\n" * 400
    )
    assert len(remote_script) > 8191
    local_env.env.update({
        "SSH_ARGV_FILE": str(argv_file),
        "SSH_STDIN_FILE": str(stdin_file),
        "SSH_TEST_PYTHON": sys.executable.replace("\\", "/"),
    })
    command = (
        f"{shlex.quote(_bash_safe_path(str(shim)))} "
        "example.invalid powershell.exe -NoProfile -NonInteractive "
        "-Command - <<'HERMES_REMOTE_SCRIPT'\n"
        f"{remote_script}"
        "HERMES_REMOTE_SCRIPT"
    )

    result = local_env.execute(command, timeout=20)

    assert result["returncode"] == 0, result
    argv = json.loads(argv_file.read_text(encoding="utf-8"))
    argv_text = "\0".join(argv)
    assert len(argv_text) < 8191
    assert "Write-Output" not in argv_text
    assert "EncodedCommand" not in argv
    assert stdin_file.read_bytes() == remote_script.encode("utf-8")
