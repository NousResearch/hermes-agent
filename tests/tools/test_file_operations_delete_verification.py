"""Regression tests for the V4A delete verification gap.

Issue #57788's fix (#57789) added post-write verification for `write_file`
and `patch_replace`. The issue explicitly noted V4A delete was out of scope:

> V4A **delete** operations are not covered by this fix (delete success is
> "file no longer exists," a different semantic than "content matches" —
> verifying that correctly needs its own check, and was out of scope for
> this fix to keep it minimal and low-risk).

`ShellFileOperations.delete_file` and `delete_path` call `_python_delete`,
which runs a Python snippet via the executor and trusts the exit code. If
the executor returns `exit_code == 0` but the file is still on disk (race
condition, backend FS oddity, sandbox quirk), the tool reports success for
a non-existent delete — the user thinks the file is gone but it isn't.

This test suite verifies that delete operations correctly report failure
when the underlying executor reports success but the file is still on
disk. The fix should re-check that the path doesn't exist after the
executor returns 0.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.file_operations import ShellFileOperations, WriteResult


def _make_env(*, exit_code: int = 0, stdout: str = "", stderr: str = ""):
    """Build a minimal env stub that returns the given exit/output for _exec."""
    env = MagicMock()
    env.execute.return_value = {
        "output": stdout,
        "error": stderr,
        "returncode": exit_code,
    }
    return env


def _make_verifying_env(*, delete_success: bool, verify_state: str):
    """Build an env that handles both the delete AND the verify command.

    delete_success: True if the delete itself should report exit_code=0
    verify_state: "GONE" if file is actually gone, "STILL_THERE" if not
    """
    env = MagicMock()

    def execute(command, *args, **kwargs):
        if command.startswith("test ! -e"):
            # The post-delete verification check
            if verify_state == "GONE":
                return {"output": "GONE\n", "error": "", "returncode": 0}
            else:
                return {"output": "STILL_THERE\n", "error": "", "returncode": 0}
        # The actual delete command
        if delete_success:
            return {"output": "", "error": "", "returncode": 0}
        return {"output": "", "error": "permission denied", "returncode": 1}

    env.execute.side_effect = execute
    return env


def _make_ops(env):
    """Build a ShellFileOperations with the given env, isolated file path root."""
    # Use a temp dir as a fake sandbox so _expand_tilde doesn't try to write
    # to the real filesystem
    return ShellFileOperations(env)


class TestDeleteFileReportsFailureWhenExecutorLies:
    """delete_file must verify post-delete that the file is actually gone."""

    def test_delete_file_succeeds_when_file_is_gone(self, tmp_path):
        """When the executor succeeds AND the file is actually gone,
        delete_file must return a success result."""
        path = str(tmp_path / "to_delete.txt")
        # Delete succeeds, verify confirms file is gone
        env = _make_verifying_env(delete_success=True, verify_state="GONE")
        ops = ShellFileOperations(env)

        result = ops.delete_file(path)
        assert result.error is None, f"unexpected error: {result.error}"

    def test_delete_file_fails_when_file_still_present(self, tmp_path):
        """When the executor reports success but the file is still on disk,
        delete_file must surface this as an error, not a silent success."""
        path = str(tmp_path / "ghost.txt")
        # Delete reports success BUT verify says file is still there
        env = _make_verifying_env(delete_success=True, verify_state="STILL_THERE")
        ops = ShellFileOperations(env)

        result = ops.delete_file(path)
        # After fix: this should be an error result
        assert result.error is not None, (
            f"delete_file reported success but file still exists! "
            f"This is the same trust gap as #57788 for write_file."
        )
        assert "still exists" in result.error.lower() or "delete" in result.error.lower()

    def test_delete_path_recursive_verifies_files_deleted(self, tmp_path):
        """Recursive delete must verify the path is actually gone."""
        path = str(tmp_path / "ghost_dir")
        # Delete succeeds, verify says it's still there
        env = _make_verifying_env(delete_success=True, verify_state="STILL_THERE")
        ops = ShellFileOperations(env)

        result = ops.delete_path(path, recursive=True)
        assert result.error is not None


class TestDeleteVerificationDoesNotBreakExistingBehavior:
    """Regression: the existing delete tests must still pass."""

    def test_delete_file_preserves_existing_return_shape(self, tmp_path):
        """WriteResult shape must not change — fix must only fill the error field."""
        path = str(tmp_path / "x.txt")
        env = _make_verifying_env(delete_success=True, verify_state="GONE")
        ops = ShellFileOperations(env)

        result = ops.delete_file(path)
        # Must be a WriteResult, not None, not an exception
        assert isinstance(result, WriteResult)
        # On success, error must be None and bytes_written can be anything
        assert result.error is None

    def test_delete_file_with_executor_failure_returns_error(self, tmp_path):
        """When the executor itself fails (exit_code != 0), the existing
        error path must still work. The new verification should not
        regress this case."""
        path = str(tmp_path / "x.txt")
        env = _make_verifying_env(delete_success=False, verify_state="STILL_THERE")
        ops = ShellFileOperations(env)

        result = ops.delete_file(path)
        assert result.error is not None
        assert "permission" in result.error.lower() or "delete" in result.error.lower()
