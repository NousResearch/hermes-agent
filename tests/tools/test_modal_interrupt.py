"""Modal interrupts must not throw away the sandbox, and a stopped sandbox must be replaced."""

import asyncio
import subprocess
import sys
import threading
import time
import types
from unittest.mock import MagicMock

import psutil
import pytest

from tools.environments import modal as modal_env


class _LocalProcess:
    """Fake Modal ContainerProcess backed by a real local subprocess."""

    def __init__(self, proc):
        self.stdout = types.SimpleNamespace(read=types.SimpleNamespace(aio=self._reader(proc.stdout)))
        self.stderr = types.SimpleNamespace(read=types.SimpleNamespace(aio=self._reader(proc.stderr)))
        self.wait = types.SimpleNamespace(aio=proc.wait)

    @staticmethod
    def _reader(stream):
        async def read():
            return (await stream.read()).decode()
        return read


class _LocalSandbox:
    """Fake Modal Sandbox whose exec runs real bash on this host."""

    def __init__(self):
        self.terminated = False
        self.exec = types.SimpleNamespace(aio=self._exec)
        self.terminate = types.SimpleNamespace(aio=self._terminate)

    async def _exec(self, *argv, timeout=None):
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return _LocalProcess(proc)

    async def _terminate(self):
        self.terminated = True


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _bare_env(sandbox):
    env = object.__new__(modal_env.ModalEnvironment)
    env._sandbox = sandbox
    env._worker = modal_env._AsyncWorker()
    env._worker.start()
    env._task_id = "test"
    env._sandbox_lock = threading.Lock()
    return env


@pytest.mark.platforms("posix")
def test_cancel_kills_the_command_and_keeps_the_sandbox(tmp_path):
    sandbox = _LocalSandbox()
    env = _bare_env(sandbox)
    env.get_temp_dir = lambda: str(tmp_path)
    marker = tmp_path / "child.pid"
    child = None
    try:
        handle = env._run_bash(f"sleep 300 & echo $! > {marker}; wait")
        deadline = time.monotonic() + 5
        while not (marker.exists() and marker.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int(marker.read_text())

        handle.kill()

        assert handle.wait(timeout=10) is not None, "the interrupted command never finished"
        deadline = time.monotonic() + 5
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child)
        assert not sandbox.terminated
        follow_up = env._run_bash("echo still-here")
        assert follow_up.wait(timeout=10) == 0
        assert "still-here" in follow_up.stdout.read()
    finally:
        if child and _alive(child):
            subprocess.run(["kill", "-9", str(child)], check=False)
        env._worker.stop()


def test_before_execute_replaces_a_stopped_sandbox(monkeypatch):
    def fake_sandbox(exit_code):
        async def poll():
            return exit_code
        return types.SimpleNamespace(poll=types.SimpleNamespace(aio=poll))

    created = []

    async def create(*_args, image=None, **_kwargs):
        created.append(image)
        return fake_sandbox(None)

    async def lookup(*_args, **_kwargs):
        return object()

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(
        App=types.SimpleNamespace(lookup=types.SimpleNamespace(aio=lookup)),
        Sandbox=types.SimpleNamespace(create=types.SimpleNamespace(aio=create))))
    monkeypatch.setattr(modal_env, "FileSyncManager", MagicMock())

    stopped = fake_sandbox(0)
    env = _bare_env(stopped)
    env._image_spec = "image-spec"
    env._sandbox_kwargs = {}
    env._cred_mounts = []
    env._sync_manager = MagicMock()
    env.init_session = MagicMock()
    try:
        env._before_execute()
        assert env._sandbox is not stopped
        assert created == ["image-spec"]
        assert env._recreated_notice_pending
        env.init_session.assert_called_once()

        env._before_execute()  # a live sandbox is kept
        assert created == ["image-spec"]
    finally:
        env._worker.stop()
