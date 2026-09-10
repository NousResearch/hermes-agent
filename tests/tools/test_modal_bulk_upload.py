"""Tests for Modal bulk upload via tar/base64 archive."""

import asyncio
import base64
import io
import tarfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.environments import modal as modal_env


def _make_mock_modal_env(monkeypatch, tmp_path):
    """Create a minimal mock ModalEnvironment for testing upload methods.

    Returns a ModalEnvironment-like object with _sandbox and _worker mocked.
    We don't call __init__ because it requires the Modal SDK.
    """
    env = object.__new__(modal_env.ModalEnvironment)
    env._sandbox = MagicMock()
    env._worker = MagicMock()
    env._persistent = False
    env._task_id = "test"
    env._sync_manager = None
    return env


def _make_mock_stdin():
    """Create a mock stdin that captures written data."""
    stdin = MagicMock()
    written_chunks = []

    def mock_write(data):
        written_chunks.append(data)

    stdin.write = mock_write
    stdin.write_eof = MagicMock()
    stdin.drain = MagicMock()
    stdin.drain.aio = AsyncMock()
    stdin._written_chunks = written_chunks
    return stdin


def _wire_async_exec(env, exec_calls=None):
    """Wire mock sandbox.exec.aio and a real run_coroutine on the env.

    Optionally captures exec call args into *exec_calls* list.
    Returns (exec_calls, run_kwargs, stdin_mock).
    """
    if exec_calls is None:
        exec_calls = []
    run_kwargs: dict = {}
    stdin_mock = _make_mock_stdin()

    async def mock_exec_fn(*args, **kwargs):
        exec_calls.append(args)
        proc = MagicMock()
        proc.wait = MagicMock()
        proc.wait.aio = AsyncMock(return_value=0)
        proc.stdin = stdin_mock
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock()
        proc.stderr.read.aio = AsyncMock(return_value="")
        return proc

    env._sandbox.exec = MagicMock()
    env._sandbox.exec.aio = mock_exec_fn

    def real_run_coroutine(coro, **kwargs):
        run_kwargs.update(kwargs)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    env._worker.run_coroutine = real_run_coroutine
    return exec_calls, run_kwargs, stdin_mock


class TestModalBulkUpload:
    """Test _modal_bulk_upload method."""

    def test_empty_files_is_noop(self, monkeypatch, tmp_path):
        """Empty file list should not call worker.run_coroutine."""
        env = _make_mock_modal_env(monkeypatch, tmp_path)
        env._modal_bulk_upload([])
        env._worker.run_coroutine.assert_not_called()

    def test_tar_archive_contains_all_files(self, monkeypatch, tmp_path):
        """The tar archive sent via stdin should contain all files."""
        env = _make_mock_modal_env(monkeypatch, tmp_path)

        src_a = tmp_path / "a.json"
        src_b = tmp_path / "b.py"
        src_a.write_text("cred_content")
        src_b.write_text("skill_content")

        files = [
            (str(src_a), "/root/.hermes/credentials/a.json"),
            (str(src_b), "/root/.hermes/skills/b.py"),
        ]

        exec_calls, _, stdin_mock = _wire_async_exec(env)
        env._modal_bulk_upload(files)

        # Verify the command reads from stdin (no echo with embedded payload)
        assert len(exec_calls) == 1
        args = exec_calls[0]
        assert args[0] == "bash"
        assert args[1] == "-c"
        cmd = args[2]
        assert "mkdir -p" in cmd
        assert "base64 -d" in cmd
        assert "tar xzf" in cmd
        assert "-C /" in cmd

        # Reassemble the base64 payload from stdin chunks and verify tar contents
        payload = "".join(stdin_mock._written_chunks)
        tar_data = base64.b64decode(payload)
        buf = io.BytesIO(tar_data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = sorted(tar.getnames())
            assert "root/.hermes/credentials/a.json" in names
            assert "root/.hermes/skills/b.py" in names

            # Verify content
            a_content = tar.extractfile("root/.hermes/credentials/a.json").read()
            assert a_content == b"cred_content"
            b_content = tar.extractfile("root/.hermes/skills/b.py").read()
            assert b_content == b"skill_content"

        # Verify stdin was closed
        stdin_mock.write_eof.assert_called_once()


    def test_stdin_chunked_for_large_payloads(self, monkeypatch, tmp_path):
        """Payloads larger than _STDIN_CHUNK_SIZE should be split into multiple writes."""
        env = _make_mock_modal_env(monkeypatch, tmp_path)

        # Use random bytes so gzip cannot compress them -- ensures the
        # base64 payload exceeds one 1 MB chunk.
        import os as _os
        src = tmp_path / "large.bin"
        src.write_bytes(_os.urandom(1024 * 1024 + 512 * 1024))
        files = [(str(src), "/root/.hermes/large.bin")]

        exec_calls, _, stdin_mock = _wire_async_exec(env)
        env._modal_bulk_upload(files)

        # Should have multiple stdin write chunks
        assert len(stdin_mock._written_chunks) >= 2

        # Reassembled payload should still decode to valid tar
        payload = "".join(stdin_mock._written_chunks)
        tar_data = base64.b64decode(payload)
        buf = io.BytesIO(tar_data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
            assert "root/.hermes/large.bin" in names
