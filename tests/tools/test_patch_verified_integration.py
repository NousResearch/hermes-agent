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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
