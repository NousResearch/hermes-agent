"""An expired upload must not resume after a newer sync has committed."""

import asyncio
import base64
import io
import tarfile
import threading
from types import SimpleNamespace

import pytest

from tools.environments.file_sync import FileSyncManager, iter_sync_files
from tools.environments.modal import ModalEnvironment, _AsyncWorker


@pytest.mark.parametrize("delayed", [True, False], ids=["expired-upload", "healthy-upload"])
def test_retry_cannot_be_overwritten_by_expired_upload(tmp_path, monkeypatch, delayed):
    home = tmp_path / "home"
    skill = home / "skills" / "example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("old instructions", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    remote = tmp_path / "remote"
    remote.mkdir()
    target = remote / "root/.hermes/skills/example/SKILL.md"
    worker = _AsyncWorker()
    worker.start()
    loop = worker._loop
    assert loop is not None
    gate = asyncio.Event()
    finished = threading.Event()
    env = object.__new__(ModalEnvironment)
    env._worker, env._persistent, env._sync_manager = worker, False, None
    calls = 0

    async def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1 and delayed:
            await gate.wait()
        payload = bytearray()

        async def drain():
            pass

        def eof():
            # Only the SDK boundary is replaced: consume the real bulk uploader's
            # tar payload into an isolated filesystem, with no cloud credentials.
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload))) as archive:
                archive.extractall(remote, filter="data")

        async def wait():
            return 0

        return SimpleNamespace(
            stdin=SimpleNamespace(write=payload.extend, write_eof=eof, drain=SimpleNamespace(aio=drain)),
            wait=SimpleNamespace(aio=wait),
        )

    env._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=execute))
    real_run = worker.run_coroutine
    submissions = 0

    def run(coro, timeout=600):
        nonlocal submissions
        submissions += 1
        if submissions != 1:
            return real_run(coro, timeout=timeout)

        async def tracked():
            try:
                return await coro
            finally:
                finished.set()

        # Shorten the production 120s deadline, retaining the real Future wait.
        return real_run(tracked(), timeout=2)

    monkeypatch.setattr(worker, "run_coroutine", run)

    def download(destination):
        with tarfile.open(destination, "w") as archive:
            archive.add(remote / "root", arcname="root")

    manager = FileSyncManager(
        iter_sync_files, env._modal_upload, env._modal_delete,
        bulk_upload_fn=env._modal_bulk_upload, bulk_download_fn=download,
    )
    try:
        manager.sync(force=True)
        assert target.exists() is not delayed
        latest = "new instructions from a later edit"
        skill.write_text(latest, encoding="utf-8")
        manager.sync(force=True)
        assert target.read_text(encoding="utf-8") == latest
        loop.call_soon_threadsafe(gate.set)
        assert finished.wait(5), "first upload did not settle"
        # A no-change cycle must not be needed to repair a late, expired upload.
        manager.sync(force=True)
        # Teardown must not mistake the stale upload for a remote edit and pull
        # it back over the user's newer host copy.
        manager.sync_back(hermes_home=home)
        assert skill.read_text(encoding="utf-8") == latest
        assert target.read_text(encoding="utf-8") == latest
        assert calls == 2
    finally:
        loop.call_soon_threadsafe(gate.set)
        finished.wait(5)
        env._sandbox = None
        worker.stop()
        loop.close()


@pytest.mark.parametrize("error", [None, ValueError("operation rejected"), TimeoutError("SDK deadline")])
def test_worker_preserves_results_and_operation_errors(error):
    worker = _AsyncWorker()
    worker.start()
    loop = worker._loop
    assert loop is not None
    value = object()

    async def operation():
        if error is not None:
            raise error
        return value

    async def next_operation():
        return value

    try:
        if error is None:
            assert worker.run_coroutine(operation(), timeout=5) is value
        else:
            with pytest.raises(type(error)) as raised:
                worker.run_coroutine(operation(), timeout=5)
            assert raised.value is error
        assert worker.run_coroutine(next_operation(), timeout=5) is value
    finally:
        worker.stop()
        loop.close()
