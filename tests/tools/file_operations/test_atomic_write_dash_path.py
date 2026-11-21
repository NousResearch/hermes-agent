"""Atomic write on a shell backend to a relative path that starts with ``-``.

``_atomic_write`` hands the path to ``mkdir -p``, ``readlink``, ``stat`` and ``mv``. Quoting does
not stop a leading ``-`` from being read as an option, so ``write_file("-notes.txt", ...)`` failed
with a usage error from ``mv`` (and a parent like ``-dir/`` broke ``mkdir``). Nothing here mocks
the write: the backend is a local bash subprocess with remote-backend semantics, so the native
host-file shortcut stays off and the shell script is what runs.
"""
import hashlib
import os
import shutil
import stat
import subprocess
import sys

import pytest

from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell file operations")

CONTENT = "atomic bytes: caf\u00e9\\n\nsecond line\n"
NAMES = ["-leading.txt", "--help", "-dir/child.txt", "-dir/-child.txt", "-quote' space.txt",
         "ordinary.txt", "./-explicit.txt"]


class SubprocessBackend:
    """Runs each command in a real bash; no ``uses_host_filesystem``, so no native I/O."""

    env_type = "ssh"

    def __init__(self, cwd):
        self.cwd = str(cwd)
        self.calls = []

    def execute(self, command, cwd=None, timeout=None, stdin_data=None):
        result = subprocess.run(
            [shutil.which("bash") or "bash", "--noprofile", "--norc", "-c", command],
            cwd=cwd or self.cwd, input=stdin_data, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=min(timeout or 10, 10), check=False,
        )
        self.calls.append((command, result.returncode, result.stdout))
        return {"output": result.stdout, "returncode": result.returncode}


@pytest.fixture
def ops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # file safety resolves relative paths on the host
    result = ShellFileOperations(SubprocessBackend(tmp_path))
    assert result._native_read_enabled() is False
    return result


@pytest.mark.parametrize("name", NAMES)
def test_bare_relative_path_is_written(ops, tmp_path, name):
    result = ops.write_file(name, CONTENT)  # deliberately no caller-added ./
    assert result.error is None, result.error
    actual = (tmp_path / name).read_bytes()
    assert actual == CONTENT.encode()
    assert result.verified is True
    assert result.bytes_written == len(actual)
    assert result._content_sha256 == hashlib.sha256(actual).hexdigest()
    assert not list(tmp_path.rglob(".hermes-tmp.*"))


@pytest.mark.parametrize("name", ["-leading.txt", "--help", "-dir/-child.txt"])
def test_overwrite_keeps_the_existing_mode(ops, tmp_path, name):
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old bytes\n")
    target.chmod(0o640)
    result = ops.write_file(name, CONTENT)
    assert result.error is None, result.error
    assert target.read_bytes() == CONTENT.encode()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_a_dash_named_symlink_updates_its_target(ops, tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"old target\n")
    target.chmod(0o640)
    link = tmp_path / "-link.txt"
    link.symlink_to("target.txt")
    result = ops.write_file("-link.txt", CONTENT)
    assert result.error is None, result.error
    assert link.is_symlink()
    assert os.readlink(link) == "target.txt"
    assert target.read_bytes() == CONTENT.encode()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
