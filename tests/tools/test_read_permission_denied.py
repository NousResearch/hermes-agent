"""read_file must tell an inaccessible path from a missing one (#138090).

A stat denied with EACCES/EPERM (search-less parent directory, sandbox
read-deny) made every shell test ([ -f ], [ -e ], [ -L ]) merely false, so
``_probe_regular_file`` answered "missing" and every read path reported
"File not found" — the model then went hunting for "similar files". The probe
now carries ``ls -d``'s errno and the read paths map "access_denied" to an
explicit permission error with ``not_found`` False.
"""

import json
import os

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture()
def ops(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path))
    return ShellFileOperations(env, cwd=str(tmp_path))


@pytest.fixture()
def locked(tmp_path):
    """A file whose parent directory denies search (chmod 000), so stat on the
    child fails with EACCES for any non-root user. Root sees through chmod, and
    Windows ignores it — both skip rather than lie."""
    parent = tmp_path / "locked"
    parent.mkdir()
    target = parent / "notes.txt"
    target.write_text("x", encoding="utf-8")
    os.chmod(parent, 0)
    try:
        os.stat(target)
    except PermissionError:
        yield target
        os.chmod(parent, 0o755)
        return
    os.chmod(parent, 0o755)
    pytest.skip("chmod 000 does not block this user (root?)")


@pytest.fixture()
def unreadable(tmp_path):
    """A regular file that exists in a searchable directory but denies the read
    (chmod 000 on the FILE, not the parent): stat succeeds, only open does not.
    Before the ``[ -r ]`` guard every read path silenced the open failure and
    answered empty content (review of #138096). Root reads through chmod 0 — skip
    rather than lie."""
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    os.chmod(target, 0)
    if os.access(target, os.R_OK):
        pytest.skip("chmod 000 leaves the file readable (root?)")
    yield target
    os.chmod(target, 0o644)


class TestProbeAccessDenied:
    @pytest.mark.platforms("posix")
    def test_probe_reports_access_denied_not_missing(self, ops, locked):
        assert ops._probe_regular_file(str(locked)) == (0, "access_denied")

    @pytest.mark.platforms("posix")
    def test_probe_still_reports_missing_for_absent_path(self, ops, tmp_path):
        assert ops._probe_regular_file(str(tmp_path / "nope.txt")) == (0, "missing")

    @pytest.mark.platforms("posix")
    def test_probe_reports_access_denied_for_unreadable_regular_file(
        self, ops, unreadable
    ):
        # Stat succeeds (searchable parent); only the read is denied. Without the
        # [ -r ] guard the wc redirect failed with stderr silenced and the file
        # read as empty.
        assert ops._probe_regular_file(str(unreadable)) == (0, "access_denied")

    @pytest.mark.platforms("posix")
    def test_missing_file_named_like_the_errno_stays_missing(self, ops, tmp_path):
        # ls's message embeds the path, so an unanchored "ermission denied"
        # substring match would flag this absent file as inaccessible.
        trick = tmp_path / "notes - permission denied.txt"
        assert ops._probe_regular_file(str(trick)) == (0, "missing")


class TestReadPathsSurfacePermissionError:
    @pytest.mark.platforms("posix")
    def test_read_file_native_handoff(self, ops, locked):
        # Default config: native stat raises PermissionError (an OSError) and
        # hands over to the shell path, whose probe must say access_denied.
        result = ops.read_file(str(locked), offset=1, limit=10)
        assert result.error and "Permission denied" in result.error
        assert not result.not_found

    @pytest.mark.platforms("posix")
    def test_read_file_compound_probe(self, ops, locked, monkeypatch):
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        result = ops.read_file(str(locked), offset=1, limit=10)
        assert result.error and "Permission denied" in result.error
        assert not result.not_found

    @pytest.mark.platforms("posix")
    def test_read_file_raw(self, ops, locked):
        result = ops.read_file_raw(str(locked))
        assert result.error and "Permission denied" in result.error
        assert not result.not_found

    @pytest.mark.platforms("posix")
    def test_read_file_bytes(self, ops, locked):
        result = ops.read_file_bytes(str(locked))
        assert result.error and "Permission denied" in result.error
        assert not result.not_found


class TestUnreadableRegularFileSurfacesPermissionError:
    """The second shape of #138090 review: the file exists, the parent is
    searchable, only the read itself is denied. Every path must say so — an
    empty-content success would teach the model the file is empty."""

    @pytest.mark.platforms("posix")
    def test_read_file_native_handoff(self, ops, unreadable):
        # Native stat succeeds; open raises PermissionError and hands over to
        # the shell path, whose [ -r ] guard must route to access_denied.
        result = ops.read_file(str(unreadable), offset=1, limit=10)
        assert result.error and "Permission denied" in result.error
        assert not result.not_found

    @pytest.mark.platforms("posix")
    def test_read_file_compound_probe(self, ops, unreadable, monkeypatch):
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        result = ops.read_file(str(unreadable), offset=1, limit=10)
        assert result.error and "Permission denied" in result.error
        assert not result.not_found

    @pytest.mark.platforms("posix")
    def test_read_file_raw(self, ops, unreadable):
        result = ops.read_file_raw(str(unreadable))
        assert result.error and "Permission denied" in result.error
        assert not result.not_found

    @pytest.mark.platforms("posix")
    def test_read_file_bytes(self, ops, unreadable):
        result = ops.read_file_bytes(str(unreadable))
        assert result.error and "Permission denied" in result.error
        assert not result.not_found


@pytest.mark.platforms("posix")
def test_read_file_tool_json_reports_permission_denied(locked, monkeypatch):
    from tools import file_tools

    monkeypatch.setenv("TERMINAL_CWD", str(locked.parent.parent))
    result = json.loads(file_tools.read_file_tool(str(locked)))
    assert result.get("error") and "Permission denied" in result["error"]
    assert not result.get("not_found")
    assert not result.get("similar_files")
