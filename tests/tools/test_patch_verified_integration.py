"""Integration tests for patch tool mode='verified'."""

import os
import subprocess
import sys
import tempfile

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tools.file_operations import ShellFileOperations


class LocalEnv:
    def __init__(self, cwd):
        self.cwd = cwd

    def execute(self, command, cwd=None, timeout=None, stdin_data=None, **kw):
        p = subprocess.run(
            command,
            shell=True,
            cwd=cwd or self.cwd,
            capture_output=True,
            text=True,
            input=stdin_data,
            timeout=timeout,
        )
        return {"output": p.stdout + p.stderr, "returncode": p.returncode}


def _ops(tmpdir):
    return ShellFileOperations(LocalEnv(tmpdir), cwd=tmpdir)


def test_verified_preserves_stale_context_and_updates_target():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.py")
        open(path, "w").write('header = "changed"\nvalue = 3\ntail = 1\n')
        ops = _ops(d)
        patch = f"""*** Begin Patch
*** Update File: {path}
@@ 2 @@
 header = "old"
-value = 3
+value = 10
 tail = 1
*** End Patch
"""
        res = ops.patch_verified(patch)
        assert res.success, res.error
        assert open(path).read() == 'header = "changed"\nvalue = 10\ntail = 1\n'


def test_verified_rejects_semantic_whitespace_stale_target():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "messages.py")
        actual = 'msg = "ab"\n'
        open(path, "w").write(actual)
        ops = _ops(d)
        patch = f"""*** Begin Patch
*** Update File: {path}
@@ 1 @@
-msg = "a b"
+msg = "new"
*** End Patch
"""
        res = ops.patch_verified(patch)
        assert not res.success
        assert "precondition" in (res.error or "")
        assert open(path).read() == actual


def test_verified_rejects_non_numeric_range_hint_with_digits():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.py")
        actual = "x = 1\n"
        open(path, "w").write(actual)
        ops = _ops(d)
        patch = f"""*** Begin Patch
*** Update File: {path}
@@ issue 123 @@
-x = 1
+x = 2
*** End Patch
"""
        res = ops.patch_verified(patch)
        assert not res.success
        assert "numeric snapshot range" in (res.error or "")
        assert open(path).read() == actual


def test_replace_mode_still_works():
    """Regression: fuzzy replace is unaffected by verified mode addition."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.py")
        open(path, "w").write("x = 1\ny = 2\n")
        ops = _ops(d)
        res = ops.patch_replace(path, old_string="x = 1", new_string="x = 42")
        assert res.success, res.error
        assert "x = 42" in open(path).read()


def test_verified_rolls_back_on_second_write_failure():
    """Multi-file verified patch must roll back the first file if the
    second write fails, preserving all-or-nothing atomicity.

    Addresses review feedback from teknium1 on PR #45627: the sequential
    write loop previously left earlier files committed when a later
    write_file returned an error.
    """

    class FailSecondWrite(ShellFileOperations):
        def __init__(self, env, cwd):
            super().__init__(env, cwd=cwd)
            self._call_count = 0
            self._fail_path = None

        def write_file(self, path, content):
            self._call_count += 1
            if self._call_count == 2:
                from tools.file_operations import WriteResult
                self._fail_path = path
                return WriteResult(error="simulated write failure")
            return super().write_file(path, content)

    with tempfile.TemporaryDirectory() as d:
        path_a = os.path.join(d, "a.py")
        path_b = os.path.join(d, "b.py")
        orig_a = "x = 1\n"
        orig_b = "y = 2\n"
        open(path_a, "w").write(orig_a)
        open(path_b, "w").write(orig_b)

        ops = FailSecondWrite(LocalEnv(d), cwd=d)

        patch = f"""*** Begin Patch
*** Update File: {path_a}
@@ 1 @@
-x = 1
+x = 42
*** Update File: {path_b}
@@ 1 @@
-y = 2
+y = 99
*** End Patch
"""
        res = ops.patch_verified(patch)

        # The patch must fail, not partially succeed.
        assert not res.success, "expected failure on second write"
        assert "simulated write failure" in (res.error or "")

        # File A must be restored to its original content (rolled back).
        assert open(path_a).read() == orig_a, "first file was not rolled back"
        # File B must be untouched (write failed).
        assert open(path_b).read() == orig_b, "second file was modified"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
