"""On the local backend a missing path must not exhaust the read budget (fail closed), while every
real lifecycle route stays blocked and remote backends keep their reader and budget."""

import json
import os
import stat
from pathlib import Path

import pytest

from cron import lifecycle_guard
from tools import process_registry
from tools.environments.local import LocalEnvironment
from tools.terminal_tool_guards import gateway_lifecycle_block


class _RemoteEnv:
    """Backend double: answers ``head -c ... < path`` from *files* and records every read."""

    def __init__(self, files=None):
        self.files = files or {}
        self.commands = []

    def execute(self, command, **kwargs):
        self.commands.append(command)
        for path, text in self.files.items():
            if path in command:
                return {"returncode": 0, "output": text}
        return {"returncode": 1, "output": ""}


class _NoReadsEnv(LocalEnvironment):
    def __init__(self):  # no shell session: the guard only needs the type
        self.cwd = None

    def execute(self, command, **kwargs):
        raise AssertionError(f"local backend ran a remote read: {command}")


def _block(command, tmp_path, *, env, env_type="local"):
    return gateway_lifecycle_block(
        command=command, env=env, env_type=env_type, cwd=str(tmp_path), workdir=None,
        session_key="local-backend-test",
    )


def _script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _assert_restart_refused(blocked):
    """A detected lifecycle command, not an unrelated could-not-scan refusal."""
    assert blocked is not None
    assert "cannot restart, stop, or uninstall" in json.loads(blocked)["error"]


def _misses(tmp_path, count):
    return "\n".join(f"bash {tmp_path}/missing-{index}.sh" for index in range(count))


@pytest.fixture(autouse=True)
def _supervised(monkeypatch):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)


def test_local_script_with_many_missing_paths_is_allowed_without_remote_reads(tmp_path):
    count = lifecycle_guard._MAX_LIFECYCLE_SCAN_REMOTE_READS + 1
    script = _script(tmp_path / "notify.sh", f"#!/bin/sh\n{_misses(tmp_path, count)}\n")

    assert _block(f"bash {script}", tmp_path, env=_NoReadsEnv()) is None


def test_local_direct_restart_stays_blocked(tmp_path):
    _assert_restart_refused(_block("hermes gateway restart", tmp_path, env=_NoReadsEnv()))


def test_local_script_with_restart_stays_blocked(tmp_path):
    script = _script(tmp_path / "restart.sh", "#!/bin/sh\nhermes gateway restart\n")

    _assert_restart_refused(_block(f"bash {script}", tmp_path, env=_NoReadsEnv()))


def test_local_wrapper_calling_restart_script_stays_blocked(tmp_path):
    script = _script(tmp_path / "restart.sh", "#!/bin/sh\nhermes gateway restart\n")
    wrapper = _script(tmp_path / "wrapper.sh", f"#!/bin/sh\nbash {script}\n")

    _assert_restart_refused(_block(f"bash {wrapper}", tmp_path, env=_NoReadsEnv()))


def test_local_restart_script_behind_many_missing_paths_stays_blocked(tmp_path):
    """Dropping the reader must not let a real script hide behind the misses."""
    count = lifecycle_guard._MAX_LIFECYCLE_SCAN_REMOTE_READS + 1
    restart = _script(tmp_path / "restart.sh", "#!/bin/sh\nhermes gateway restart\n")
    wrapper = _script(tmp_path / "wrapper.sh", f"#!/bin/sh\n{_misses(tmp_path, count)}\nbash {restart}\n")

    _assert_restart_refused(_block(f"bash {wrapper}", tmp_path, env=_NoReadsEnv()))


def test_local_text_script_with_binary_looking_prefix_stays_blocked(tmp_path):
    """The local read skips a file starting with binary magic, yet bash runs ``MZ\\n...`` as text."""
    script = _script(tmp_path / "pe.sh", "MZ\nhermes gateway restart\n")

    _assert_restart_refused(_block(f"bash {script}", tmp_path, env=_NoReadsEnv()))


def _process_relative_forms(tmp_path):
    (tmp_path / "file-link").symlink_to("/proc/self/cwd/r.sh")
    (tmp_path / "dir-link").symlink_to("/proc/self/cwd")
    return [
        "/proc/self/cwd/r.sh",
        "/proc/thread-self/cwd/r.sh",
        f"{tmp_path}/file-link",
        f"{tmp_path}/dir-link/r.sh",
    ]


@pytest.mark.skipif(not Path("/proc/self/cwd").exists(), reason="needs procfs")
def test_local_process_relative_path_stays_blocked(tmp_path, monkeypatch):
    """Missing for this process, present for the shell: the gateway's cwd is not the terminal's."""
    work = tmp_path / "work"
    elsewhere = tmp_path / "elsewhere"
    work.mkdir()
    elsewhere.mkdir()
    _script(work / "r.sh", "hermes gateway restart\n")
    monkeypatch.chdir(elsewhere)
    env = LocalEnvironment(cwd=str(work), persistent=False)

    for path in _process_relative_forms(tmp_path):
        _assert_restart_refused(
            gateway_lifecycle_block(
                command=f"bash {path}", env=env, env_type="local", cwd=str(work), workdir=None,
                session_key="local-backend-test",
            )
        )


def test_definitely_missing_on_host(tmp_path):
    (tmp_path / "real").write_text("x", encoding="utf-8")
    (tmp_path / "dangling").symlink_to(tmp_path / "gone")
    (tmp_path / "dir-link").symlink_to(tmp_path)

    missing = lifecycle_guard._definitely_missing_on_host
    assert missing(tmp_path / "gone") is True
    assert missing(tmp_path / "gone" / "deeper.sh") is True
    assert missing(tmp_path / "real") is False
    assert missing(tmp_path / "dangling") is False
    assert missing(tmp_path / "dir-link" / "gone") is False
    assert missing(tmp_path / "real" / "under-a-file") is True
    assert missing(Path("relative/gone")) is False
    assert missing(Path("/nul\0byte")) is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_definitely_missing_on_host_treats_an_unsearchable_parent_as_uncertain(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert lifecycle_guard._definitely_missing_on_host(locked / "r.sh") is False
    finally:
        locked.chmod(0o700)


def _flag_passed_to_scan(env, env_type, tmp_path, monkeypatch):
    seen = []

    def _scan(command, **kwargs):
        seen.append(kwargs["backend_shares_host_filesystem"])
        return False, None

    monkeypatch.setattr(lifecycle_guard, "scan_gateway_lifecycle", _scan)
    assert _block("echo ok", tmp_path, env=env, env_type=env_type) is None
    return seen == [True]


def test_host_filesystem_flag_needs_a_local_env_type_and_a_host_env(tmp_path, monkeypatch):
    local = _NoReadsEnv()
    assert _flag_passed_to_scan(local, "local", tmp_path, monkeypatch) is True
    assert _flag_passed_to_scan(local, "ssh", tmp_path, monkeypatch) is False
    assert _flag_passed_to_scan(_RemoteEnv(), "local", tmp_path, monkeypatch) is False


def test_local_existing_unread_files_still_spend_the_read_budget(tmp_path, monkeypatch):
    """Only a path that does not exist is free: a file the local read skipped still counts."""
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_REMOTE_READS", 2)
    paths = [_script(tmp_path / f"pe{index}.sh", "MZ\necho ok\n") for index in range(3)]

    blocked = _block(" && ".join(f"bash {path}" for path in paths), tmp_path, env=_NoReadsEnv())

    assert blocked is not None
    assert "remote reads" in json.loads(blocked)["error"]


def test_local_env_not_on_host_filesystem_keeps_the_reader(tmp_path):
    class _Sandboxed(_RemoteEnv, LocalEnvironment):
        uses_host_filesystem = False

    env = _Sandboxed({"/remote/restart.sh": "hermes gateway restart\n"})

    _assert_restart_refused(_block("bash /remote/restart.sh", tmp_path, env=env))
    assert env.commands


def test_host_filesystem_env_with_remote_env_type_keeps_the_reader(tmp_path):
    class _HostLabelled(_RemoteEnv, LocalEnvironment):
        pass

    env = _HostLabelled({"/remote/restart.sh": "hermes gateway restart\n"})

    _assert_restart_refused(_block("bash /remote/restart.sh", tmp_path, env=env, env_type="ssh"))
    assert env.commands


# "local" here: env_type comes from config, and a reused environment can still be a remote one.
@pytest.mark.parametrize("env_type", ["local", "ssh", "docker", "modal", "singularity"])
def test_remote_backend_script_with_restart_stays_blocked(tmp_path, env_type):
    env = _RemoteEnv({"/remote/restart.sh": "hermes gateway restart\n"})

    assert _block("bash /remote/restart.sh", tmp_path, env=env, env_type=env_type) is not None
    assert env.commands


@pytest.mark.parametrize("env_type", ["local", "ssh", "docker", "modal", "singularity"])
def test_remote_backend_keeps_the_remote_read_budget(tmp_path, env_type, monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_REMOTE_READS", 2)
    env = _RemoteEnv()
    command = " && ".join(f"bash /remote/{index}.sh" for index in range(3))

    blocked = _block(command, tmp_path, env=env, env_type=env_type)

    assert blocked is not None
    assert "remote reads" in json.loads(blocked)["error"]
    assert len(env.commands) == 2
