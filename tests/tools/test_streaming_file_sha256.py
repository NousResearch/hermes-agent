"""Streaming digest matches raw bytes over the shell transport."""
import hashlib

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


def test_streaming_hash_shell_transport(tmp_path, monkeypatch):
    path = tmp_path / "large file.txt"
    data = bytes([239, 187, 191]) + b"abc\r\n" * 10000
    path.write_bytes(data)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    assert ops.file_sha256(str(path)) == hashlib.sha256(data).hexdigest()


def test_rejects_digest_with_unfenced_extra_output(tmp_path, monkeypatch):
    from tools.file_operations_common import ExecuteResult
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    monkeypatch.setattr(ops, "_exec", lambda *a, **kw: ExecuteResult(stdout="a" * 64 + " banner", exit_code=0))
    assert ops.file_sha256(str(tmp_path / "data")) is None


def test_native_hash_exception_does_not_escape(tmp_path, monkeypatch):
    from tools import file_operations
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: True)

    def fail(*args):
        raise RuntimeError("synthetic_private_value")

    monkeypatch.setattr(file_operations._native_io, "sha256_of", fail)
    assert ops.file_sha256(str(tmp_path / 'x')) is None


def test_backend_exception_does_not_escape_digest_or_delete(tmp_path, monkeypatch):
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic_private_value")

    monkeypatch.setattr(ops, "_exec_python_snippet", fail)
    assert ops.file_sha256(str(tmp_path / 'x')) is None
    result = ops.delete_file(str(tmp_path / 'x'), expected_sha256='a' * 64)
    assert result.error and "synthetic_private_value" not in repr(result)


def test_ignores_banner_around_fenced_digest(tmp_path, monkeypatch):
    from tools.file_operations_common import ExecuteResult
    path = tmp_path / "data"
    path.write_bytes(b"hello\r\n")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    original = ops._exec_python_snippet

    def noisy(*args, **kwargs):
        result = original(*args, **kwargs)
        return ExecuteResult(stdout="banner\n" + result.stdout + "\nnoise", exit_code=result.exit_code)

    monkeypatch.setattr(ops, "_exec_python_snippet", noisy)
    assert ops.file_sha256(str(path)) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_rejects_nonregular_and_missing_interpreter(tmp_path, monkeypatch):
    import os
    import pytest
    from tools.file_operations_common import ExecuteResult

    if not hasattr(os, 'mkfifo'):
        pytest.skip('FIFO unsupported')
    path = tmp_path / 'pipe'
    os.mkfifo(path)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, '_native_read_enabled', lambda: False)
    assert ops.file_sha256(str(path)) is None
    monkeypatch.setattr(ops, '_exec_python_snippet', lambda *a, **kw: ExecuteResult(stdout='', exit_code=127))
    assert ops.file_sha256(str(path)) is None


def test_streaming_hash_does_not_require_sha256sum(tmp_path, monkeypatch):
    path = tmp_path / "sample.bin"
    data = b"some raw bytes" * 10000
    path.write_bytes(data)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    real_exec = ops._exec

    def no_sha(command, *args, **kwargs):
        if "sha256sum" in command:
            from tools.file_operations_common import ExecuteResult
            return ExecuteResult(stdout="", exit_code=127)
        return real_exec(command, *args, **kwargs)

    monkeypatch.setattr(ops, "_exec", no_sha)
    assert ops.file_sha256(str(path)) == hashlib.sha256(data).hexdigest()
