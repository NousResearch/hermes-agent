"""Permission-denied writes must explain themselves.

The atomic-write path streams into a ``.hermes-tmp.XXXXXX`` file beside the
target, so a bare permission failure used to surface as::

    Failed to write file: /usr/local/sbin/.hermes-tmp.38704: Permission denied

That names a temp path the caller never asked for, points at the wrong thing
(the *directory* is unwritable, not that temp file), and reads like an internal
Hermes bug -- so the agent retries the same doomed write instead of reaching for
sudo. These tests pin the actionable message instead.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.file_operations import ShellFileOperations  # noqa: E402


class _LocalEnv:
    """Minimal terminal backend: execute() -> {output, returncode}."""

    def __init__(self, cwd):
        self.cwd = str(cwd)

    def execute(self, command, cwd=None, stdin_data=None, **kwargs):
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            cwd=cwd or self.cwd,
            input=stdin_data if stdin_data is not None else None,
        )
        return {
            "output": (proc.stdout or "") + (proc.stderr or ""),
            "returncode": proc.returncode,
        }


@pytest.fixture
def ops(tmp_path):
    return ShellFileOperations(_LocalEnv(tmp_path))


@pytest.fixture
def readonly_dir(tmp_path):
    """A directory the current user cannot write into.

    Uses mode 0o500 rather than a real system path so the test is
    hermetic and does not depend on the host layout.
    """
    d = tmp_path / "locked"
    d.mkdir()
    (d / "existing.txt").write_text("original\n")
    d.chmod(0o500)
    yield d
    d.chmod(0o700)  # let pytest clean up


@pytest.mark.skipif(
    __import__("os").geteuid() == 0,
    reason="root bypasses directory permissions",
)
class TestPermissionDeniedMessage:
    def test_existing_file_in_unwritable_dir(self, ops, readonly_dir):
        target = readonly_dir / "existing.txt"
        err = ops.write_file(str(target), "new content\n").error

        assert err, "write into an unwritable directory must fail"
        assert "hermes-tmp" not in err, "internal temp path must not leak"
        assert str(target) in err, "the real target must be named"
        assert "permission denied" in err.lower()
        assert "sudo install" in err, "must suggest the elevated-write remedy"
        # original untouched -- the atomic write never swapped anything in
        assert target.read_text() == "original\n"

    def test_new_file_in_unwritable_dir(self, ops, readonly_dir):
        target = readonly_dir / "brand-new.txt"
        err = ops.write_file(str(target), "hello\n").error

        assert err
        assert "hermes-tmp" not in err
        assert str(target) in err
        assert "sudo install" in err

    def test_patch_inherits_the_explanation(self, ops, readonly_dir):
        """patch_replace routes through write_file, so it must not leak either."""
        target = readonly_dir / "existing.txt"
        err = ops.patch_replace(str(target), "original", "patched").error

        assert err
        assert "hermes-tmp" not in err, "patch must not leak the temp path either"

    def test_non_permission_errors_pass_through(self, ops, tmp_path):
        """Only permission failures get rewritten; other errors stay verbatim."""
        # A path whose parent is a regular file -> ENOTDIR, not EACCES.
        blocker = tmp_path / "iam_a_file"
        blocker.write_text("x")
        err = ops.write_file(str(blocker / "child.txt"), "data").error

        assert err
        assert "sudo install" not in err, "non-permission errors must not be rewritten"


@pytest.mark.skipif(
    __import__("os").geteuid() == 0,
    reason="root bypasses directory permissions",
)
def test_writable_paths_still_work(ops, tmp_path):
    """Non-regression: the happy path is untouched."""
    target = tmp_path / "fine.txt"
    result = ops.write_file(str(target), "hello\n")

    assert result.error is None
    assert target.read_text() == "hello\n"
