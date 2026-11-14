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


def test_python_hash_fallback_when_sha256sum_missing(tmp_path, monkeypatch):
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
