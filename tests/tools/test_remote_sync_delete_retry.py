"""A rejected remote deletion stays pending until a later sync can remove it."""

import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tools.environments.daytona import DaytonaEnvironment
from tools.environments.file_sync import FileSyncManager
from tools.environments.modal import ModalEnvironment, _AsyncWorker


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses directory write permissions")
@pytest.mark.parametrize("backend", ["modal", "daytona"])
def test_sync_retries_rejected_remote_deletion(tmp_path, monkeypatch, backend):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "script.py"
    source.write_text("print('obsolete')\n", encoding="utf-8")
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    destination = remote_dir / source.name
    worker = _AsyncWorker()
    worker.start()

    def run(command):
        return subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)

    if backend == "modal":
        env = object.__new__(ModalEnvironment)
        env._worker = worker

        async def execute(*args, **kwargs):
            result = run(args[-1])

            async def wait():
                return result.returncode

            async def stderr():
                return result.stderr

            return SimpleNamespace(wait=SimpleNamespace(aio=wait), stderr=SimpleNamespace(read=SimpleNamespace(aio=stderr)))

        env._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=execute))
        delete = env._modal_delete
    else:
        env = object.__new__(DaytonaEnvironment)

        def execute(command):
            result = run(command)
            return SimpleNamespace(result=result.stdout + result.stderr, exit_code=result.returncode)

        env._sandbox = SimpleNamespace(process=SimpleNamespace(exec=execute))
        delete = env._daytona_delete

    def upload(host, target):
        shutil.copy2(host, target)

    manager = FileSyncManager(
        lambda: [(str(source), str(destination))] if source.exists() else [],
        upload, delete, sync_interval=0,
    )
    try:
        manager.sync()
        assert destination.read_bytes() == source.read_bytes()
        source.unlink()
        remote_dir.chmod(0o555)
        manager.sync()
        assert destination.exists()  # real rm failed; this is not an injected Python exception
        remote_dir.chmod(0o755)
        manager.sync()
        assert not destination.exists(), "failed deletion was committed and never retried"
        manager.sync()  # a completed deletion remains a no-op
    finally:
        remote_dir.chmod(0o755)
        worker.stop()
