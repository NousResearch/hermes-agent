"""Post-write hash verification on a shell backend, with real GNU ``sha256sum``.

GNU ``sha256sum FILE`` escapes a file name that holds a backslash or newline and marks the line
with a leading ``\\``, so the first token is ``\\<digest>`` and never equals the real digest: the
write landed but was reported as a hash mismatch, which is a hard error. A name starting with
``-`` was read as an option, so no hash was taken at all. Nothing here mocks checksum output:
the backend is a local bash subprocess with remote-backend semantics, so the native host-file
shortcut stays off and the shell path is what runs.
"""
import hashlib
import shutil
import subprocess
import sys

import pytest

from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file names and GNU coreutils")

NAMES = ["ordinary.txt", "back\\slash.txt", "new\nline.txt", "-leading.txt"]
IDS = ["ordinary", "backslash", "newline", "dash-prefixed"]
CONTENT = "persisted bytes: caf\u00e9\\n\nsecond line\n"


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


def _gnu_sha256sum() -> bool:
    try:
        version = subprocess.run(["sha256sum", "--version"], text=True, capture_output=True,
                                 timeout=10, check=False).stdout
    except OSError:
        return False
    return "GNU coreutils" in version


@pytest.fixture
def ops(tmp_path, monkeypatch):
    if not _gnu_sha256sum():
        pytest.skip("needs GNU sha256sum")
    monkeypatch.chdir(tmp_path)  # file safety resolves relative paths on the host
    result = ShellFileOperations(SubprocessBackend(tmp_path))
    assert result._native_read_enabled() is False
    return result


def _hash_calls(ops):
    return [call for call in ops.env.calls if call[0].startswith("sha256sum")]


@pytest.mark.parametrize("name", NAMES[:3], ids=IDS[:3])
def test_write_reports_verified_for_the_bytes_on_disk(ops, tmp_path, name):
    target = tmp_path / name
    result = ops.write_file(str(target), CONTENT)
    # Persistence first: the red run shows a false mismatch, not a failed write.
    assert target.read_bytes() == CONTENT.encode()
    assert result.error is None, result.error
    assert result.verified is True
    assert result.bytes_written == len(CONTENT.encode())


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_verify_takes_a_hash_for_every_name(ops, tmp_path, name):
    target = tmp_path / name
    target.write_bytes(CONTENT.encode())
    assert ops._verify_written_hash(str(target), CONTENT.encode()) == (True, None)
    assert _hash_calls(ops)[-1][1] == 0


def test_a_bare_dash_prefixed_relative_name_is_hashed(ops, tmp_path):
    (tmp_path / "-leading.txt").write_bytes(CONTENT.encode())
    assert ops._verify_written_hash("-leading.txt", CONTENT.encode()) == (True, None)


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_a_genuine_mismatch_stays_a_hard_error(ops, tmp_path, name):
    target = tmp_path / name
    target.write_bytes(b"different persisted bytes\n")
    verified, error = ops._verify_written_hash(str(target), CONTENT.encode())
    assert verified is False
    assert "on-disk content hash differs" in error.error


def test_a_missing_file_is_unverified_not_a_mismatch(ops, tmp_path):
    assert ops._verify_written_hash(str(tmp_path / "absent.txt"), b"") == (None, None)


def test_an_empty_file_with_a_backslash_name_verifies(ops, tmp_path):
    target = tmp_path / "empty\\file.txt"
    target.touch()
    assert ops._verify_written_hash(str(target), b"") == (True, None)


@pytest.mark.parametrize("name", NAMES[1:3], ids=IDS[1:3])
def test_gnu_filename_output_really_carries_the_escape_marker(tmp_path, name):
    """Pins the premise: GNU's per-file line starts with a backslash for these names."""
    if not _gnu_sha256sum():
        pytest.skip("needs GNU sha256sum")
    target = tmp_path / name
    target.write_bytes(CONTENT.encode())
    output = subprocess.run(["sha256sum", "--", str(target)], text=True, capture_output=True,
                            timeout=10, check=True).stdout
    assert output.startswith("\\" + hashlib.sha256(CONTENT.encode()).hexdigest())
