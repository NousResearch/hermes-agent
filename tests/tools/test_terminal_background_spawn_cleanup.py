"""Transactional cleanup for failures after a background command has spawned."""

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from types import SimpleNamespace

import pytest

from tools import terminal_tool_background as background
from tools.process_registry import ProcessRegistry


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_post_spawn_watcher_setup_failure_reaps_process(monkeypatch, tmp_path):
    """A setup error must not hide a live command behind a handle-less error."""
    import tools.process_registry as process_registry_module

    registry = ProcessRegistry()
    spawned = {}
    original_spawn = background._spawn

    def tracking_spawn(*args, **kwargs):
        session = original_spawn(*args, **kwargs)
        spawned["session"] = session
        return session

    def enable_async_delivery(session, _result, notify, patterns):
        session.watcher_platform = "test"
        return notify, patterns

    def fail_watcher_setup(_registry, session, _session_key):
        assert session.process.poll() is None, (
            "injected failure must happen while the command is live"
        )
        raise RuntimeError("injected watcher setup failure")

    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(process_registry_module, "_find_shell", lambda: "/bin/bash")
    monkeypatch.setattr(registry, "_spawn_env", lambda _env_vars: os.environ.copy())
    monkeypatch.setattr(background, "_spawn", tracking_spawn)
    monkeypatch.setattr(background, "_apply_async_support", enable_async_delivery)
    monkeypatch.setattr(background, "_register_completion_watcher", fail_watcher_setup)

    python = shlex.quote(sys.executable)
    script = shlex.quote("import time; time.sleep(60)")
    try:
        result = json.loads(
            background.spawn_background_process(
                command=f"{python} -c {script}",
                env=SimpleNamespace(env={}),
                env_type="local",
                effective_task_id="spawn-cleanup-test",
                task_id="spawn-cleanup-test",
                session_key="spawn-cleanup-test",
                workdir=None,
                cwd=str(tmp_path),
                effective_pty=False,
                notify_on_complete=True,
                watch_patterns=None,
                approval_note=None,
                pty_disabled_reason=None,
            )
        )
        assert "session" in spawned, result
        session = spawned["session"]

        assert result["exit_code"] == -1
        assert "injected watcher setup failure" in result["error"]
        assert session.process.poll() is not None, result
        assert session.id not in registry._running, result
        assert registry.is_completion_consumed(session.id), result
    finally:
        if session := spawned.get("session"):
            with suppress(Exception):
                registry._reap_untracked(session, session.process)


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_remote_cleanup_without_tree_proof_returns_session_handle(
    monkeypatch, tmp_path
):
    """A sandbox wrapper kill cannot hide a descendant that may still be running."""
    import tools.process_registry as process_registry_module

    class LocalShellSandbox:
        """Run the real ``spawn_via_env`` shell protocol on this host."""

        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, timeout=None, **_kwargs):
            fd, output_path = tempfile.mkstemp(dir=tmp_path)
            os.close(fd)
            try:
                with open(output_path, "w+", encoding="utf-8") as output:
                    completed = subprocess.run(
                        ["/bin/bash", "-c", command],
                        cwd=tmp_path,
                        text=True,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        check=False,
                    )
                    output.seek(0)
                    return {"output": output.read(), "returncode": completed.returncode}
            finally:
                with suppress(OSError):
                    os.unlink(output_path)

    registry = ProcessRegistry()
    spawned = {}
    original_spawn = background._spawn
    parent_pid_path = tmp_path / "remote-parent.pid"
    child_pid_path = tmp_path / "remote-child.pid"

    def tracking_spawn(*args, **kwargs):
        session = original_spawn(*args, **kwargs)
        spawned["session"] = session
        return session

    def fail_setup(session, *_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not child_pid_path.exists():
            time.sleep(0.01)
        assert child_pid_path.exists(), "remote command did not spawn its child"
        spawned["pids"] = {
            session.pid,
            int(parent_pid_path.read_text()),
            int(child_pid_path.read_text()),
            *registry._live_descendants(session.pid),
        }
        raise RuntimeError("injected post-spawn setup failure")

    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(background, "_spawn", tracking_spawn)
    monkeypatch.setattr(background, "_apply_async_support", fail_setup)

    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import os,pathlib,subprocess,sys,time;"
        f"pathlib.Path({str(parent_pid_path)!r}).write_text(str(os.getpid()));"
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        "time.sleep(60)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"

    try:
        result = json.loads(
            background.spawn_background_process(
                command=command,
                env=LocalShellSandbox(),
                env_type="docker",
                effective_task_id="spawn-cleanup-test",
                task_id="spawn-cleanup-test",
                session_key="spawn-cleanup-test",
                workdir=None,
                cwd=str(tmp_path),
                effective_pty=False,
                notify_on_complete=True,
                watch_patterns=None,
                approval_note=None,
                pty_disabled_reason=None,
            )
        )
        session = spawned["session"]
        child_pid = int(child_pid_path.read_text())

        assert ProcessRegistry._is_host_pid_alive(child_pid), (
            "the reproduction requires the remote descendant to outlive its wrapper"
        )
        assert result["exit_code"] == -1
        assert result["session_id"] == session.id
        assert result["pid"] == session.pid
        assert result["cleanup_status"] == "unconfirmed"
        assert "sandbox" in result["cleanup_error"]
        assert registry.get(result["session_id"]) is session
        assert session.id in registry._running
        assert not session.exited
    finally:
        for pid in spawned.get("pids", ()):
            with suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
