"""Modal sync-back must preserve an archive's bytes before extracting any files."""

import shlex
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from tools.environments.file_sync import FileSyncManager, iter_sync_files
from tools.environments.modal import ModalEnvironment, _AsyncWorker


def _environment(tmp_path, reader):
    remote = tmp_path / "remote"
    remote.mkdir()
    worker = _AsyncWorker()
    worker.start()
    env = object.__new__(ModalEnvironment)
    env._worker = worker

    async def execute(*args, text=True):
        # Execute the real tar command in an isolated stand-in for the cloud filesystem.
        command = args[-1].replace("-C / root/.hermes", f"-C {shlex.quote(str(remote))} root/.hermes")
        result = subprocess.run([*args[:-1], command], capture_output=True, check=False)
        if reader == "sdk":
            from modal.io_streams import _StreamReaderThroughServer
            from modal_proto import api_pb2

            async def output(request):
                data = result.stdout if request.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT else result.stderr
                yield SimpleNamespace(batch_index=1, items=[SimpleNamespace(message_bytes=data)],
                                      HasField=lambda name: name == "exit_code")

            client: Any = SimpleNamespace(stub=SimpleNamespace(ContainerExecGetOutput=SimpleNamespace(unary_stream=output)))

            def stream(fd):
                actual = _StreamReaderThroughServer(fd, "ex-test", "container_process", client, text=text)
                return SimpleNamespace(read=SimpleNamespace(aio=actual.read))
        else:
            def stream(fd):
                async def read():
                    data = result.stdout if fd == 1 else result.stderr
                    return data.decode("utf-8") if text else data
                return SimpleNamespace(read=SimpleNamespace(aio=read))

        async def wait():
            return result.returncode

        return SimpleNamespace(stdout=stream(1), stderr=stream(2), wait=SimpleNamespace(aio=wait))

    env._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=execute))
    return env, remote, worker


@pytest.mark.parametrize("reader", ["local", "sdk"])
@pytest.mark.parametrize("payload", [b"ordinary text\n", bytes(range(256)), b"\x89PNG\r\n\x1a\n\x00\xff"],
                         ids=["text", "all-bytes", "png"])
def test_sync_back_preserves_binary_files_and_neighboring_text(tmp_path, monkeypatch, reader, payload):
    if reader == "sdk":
        pytest.importorskip("modal")
    home = tmp_path / "home"
    skill = home / "skills" / "example"
    skill.mkdir(parents=True)
    source = skill / "SKILL.md"
    source.write_text("original instructions", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    env, remote, worker = _environment(tmp_path, reader)
    remote_source = remote / "root/.hermes/skills/example/SKILL.md"

    def upload(host, target):
        destination = remote / target.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(host, destination)

    manager = FileSyncManager(
        iter_sync_files,
        upload, lambda paths: None, bulk_download_fn=env._modal_bulk_download,
    )
    monkeypatch.setattr("tools.environments.file_sync._sleep", lambda delay: None)
    try:
        manager.sync(force=True)
        remote_source.write_text("updated instructions", encoding="utf-8")
        (remote_source.parent / "asset.bin").write_bytes(payload)
        manager.sync_back(hermes_home=home)
        assert source.read_text(encoding="utf-8") == "updated instructions"
        assert (skill / "asset.bin").read_bytes() == payload
    finally:
        worker.stop()


@pytest.mark.parametrize("reader", ["local", "sdk"])
def test_failed_archive_command_does_not_publish_a_partial_download(tmp_path, reader):
    if reader == "sdk":
        pytest.importorskip("modal")
    env, remote, worker = _environment(tmp_path, reader)
    # Missing remote .hermes makes real tar emit an incomplete archive and exit nonzero.
    archive = tmp_path / "download.tar"
    archive.write_bytes(b"previous download")
    try:
        with pytest.raises(RuntimeError, match="bulk download failed"):
            env._modal_bulk_download(archive)
        assert archive.read_bytes() == b"previous download"
    finally:
        worker.stop()
