"""A failed atomic write must not leave its staged temp file behind.

``ShellFileOperations._atomic_write`` stages the new bytes in a temp file next to
the target and renames it over the target. When the target is read-only, the
perms copy that runs before ``cat`` hands the temp that same read-only mode, so
the write itself fails with ``EACCES`` and the EXIT trap is the *only* cleanup
path left. That trap used to quote the temp path with stray backslashes, so it
removed ``\\<path>\\`` instead of ``<path>``: the write failed, the original file
was correctly left intact, and a 0-byte read-only temp file was stranded in the
target's directory — every retry leaving one more.

The invariant these tests pin is about the FAILURE path only; the success path
is already covered by ``test_file_write_surrogate_roundtrip``. They run the real
``ShellFileOperations`` against a real ``LocalEnvironment`` and inspect the real
directory, because the defect is entirely in the emitted shell script.
"""
import os

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

# Running as root makes ``cat >`` succeed on a 0o444 file, so the failure path
# under test cannot be reached; skip rather than assert something meaningless.
requires_unprivileged_user = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores the read-only file mode, so the write-failure path is unreachable",
)


@pytest.fixture
def env(tmp_path):
    """A real LocalEnvironment rooted in a temp directory."""
    return LocalEnvironment(cwd=str(tmp_path), timeout=15)


@pytest.fixture
def ops(env, tmp_path):
    """ShellFileOperations wired to the real local environment."""
    return ShellFileOperations(env, cwd=str(tmp_path))


class TestAtomicWriteTempCleanup:
    @requires_unprivileged_user
    def test_failed_write_on_readonly_target_strands_no_temp(self, ops, tmp_path):
        target = tmp_path / "readonly.py"
        target.write_text("# original\n")
        target.chmod(0o444)

        result = ops.write_file(str(target), "# attempted rewrite\n")

        assert result.error is not None, "writing a read-only file must fail"
        assert target.read_text() == "# original\n", "the original must survive untouched"
        assert not list(tmp_path.glob(".hermes-tmp*")), "the staged temp must be cleaned up"

    @requires_unprivileged_user
    def test_repeated_failed_writes_do_not_accumulate_temps(self, ops, tmp_path):
        target = tmp_path / "readonly2.py"
        target.write_text("# original\n")
        target.chmod(0o444)

        for _ in range(3):
            result = ops.write_file(str(target), "# attempted rewrite\n")
            assert result.error is not None

        assert target.read_text() == "# original\n"
        assert not list(tmp_path.glob(".hermes-tmp*"))
