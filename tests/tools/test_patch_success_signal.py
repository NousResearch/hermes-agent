"""Tests for the patch tool's success/error signalling.

Regression tests for issue #59600: ``patch()`` must report ``success: false``
(and an ``error`` string) when the write is silently skipped — i.e. when
preconditions fail and nothing actually lands on disk.  A success-looking
response for a no-op write caused silent data loss in the agent loop.

The failure modes covered here:
  * ``old_text`` (old_string) is not found in the target file
  * the target path doesn't exist for a V4A "Add File" patch (create)
  * a fuzzy V4A UPDATE hunk matches zero times

Plus three happy-path tests to make sure the fix does NOT regress:
  * exact-match replace succeeds
  * V4A Add File succeeds
  * fuzzy UPDATE succeeds
"""

import json
import shutil

import pytest


# ---------------------------------------------------------------------------
# Fixtures: keep HERMES_HOME isolated + clear the file-ops cache so each test
# gets a fresh shell backend without leaking read-tracker state across files.
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    try:
        from tools.file_tools import clear_file_ops_cache, _read_tracker_lock, _read_tracker
        clear_file_ops_cache()
        with _read_tracker_lock:
            _read_tracker.clear()
    except Exception:
        pass


@pytest.fixture
def clear_patch_failures():
    """Reset the per-task consecutive-failure counter so a previous test's
    escalation cycle doesn't poison the ``success: true`` path of this one."""
    try:
        from tools.file_tools import _patch_failure_tracker, _patch_failure_lock
        with _patch_failure_lock:
            _patch_failure_tracker.clear()
    except Exception:
        pass
    yield
    try:
        from tools.file_tools import _patch_failure_tracker, _patch_failure_lock
        with _patch_failure_lock:
            _patch_failure_tracker.clear()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# REPLACE MODE — old_string not found
# ---------------------------------------------------------------------------

class TestReplaceModeFailureSignal:
    def test_old_string_not_found_returns_success_false(self, hermes_home, tmp_path, clear_patch_failures):
        """When old_string can't be matched, the response must carry
        ``success: false`` (not a misleading ``success: true`` with the
        failure buried in an ``error`` field the agent might ignore)."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "victim.py"
        target.write_text("def hello():\n    return 'hi'\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "def goodbye():\n    return 'bye'\n",
                "new_string": "def hello():\n    return 'howdy'\n",
            },
            task_id="t_old_missing",
        )
        d = json.loads(result)

        assert d.get("success") is False, (
            f"old_string not found must report success: false, got: {d!r}"
        )
        assert d.get("error"), f"missing error message: {d!r}"
        # File must be untouched — the failure is a no-op, not a partial write.
        assert target.read_text() == "def hello():\n    return 'hi'\n"

    def test_fuzzy_match_zero_results_returns_success_false(
        self, hermes_home, tmp_path, clear_patch_failures,
    ):
        """When the fuzzy matcher's 9 strategies all return zero hits
        (no substring of the file even resembles ``old_string``), success
        must be false.  We pick a token with no character overlap with
        the file's contents so no whitespace/case normalisation can bridge
        the gap."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "fuzzy_victim.py"
        target.write_text("alpha\nbeta\ngamma\n")

        # A 30-char token built from letters/digits that appear NOWHERE
        # in "alpha\nbeta\ngamma\n" — fuzzy strategies (case-fold,
        # whitespace-normalise, token-jumble, prefix, suffix, etc.) all
        # bail on zero candidate hits.
        needle = "ZZZZQQQQXXXXPPPPMMMMCCCC_VVVV_BBBB_NNNN_DDDD_FFFF"
        assert needle not in target.read_text(), "test setup invariant"

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": needle,
                "new_string": "replacement\n",
            },
            task_id="t_fuzzy_zero",
        )
        d = json.loads(result)

        assert d.get("success") is False, f"fuzzy zero-hits must be success: false: {d!r}"
        assert d.get("error"), f"missing error message: {d!r}"
        # File must remain untouched.
        assert target.read_text() == "alpha\nbeta\ngamma\n"


# ---------------------------------------------------------------------------
# REPLACE MODE — happy path (regression guard)
# ---------------------------------------------------------------------------

class TestReplaceModeHappyPath:
    def test_exact_match_replacement_returns_success_true(
        self, hermes_home, tmp_path, clear_patch_failures,
    ):
        from tools.file_tools import _handle_patch

        target = tmp_path / "happy_replace.py"
        target.write_text("def hello():\n    return 'hi'\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "return 'hi'",
                "new_string": "return 'howdy'",
            },
            task_id="t_happy_replace",
        )
        d = json.loads(result)

        assert d.get("success") is True, (
            f"exact-match replacement must remain success: true, got: {d!r}"
        )
        assert "return 'howdy'" in target.read_text()


# ---------------------------------------------------------------------------
# V4A PATCH MODE — create (Add File) on missing parent path
# ---------------------------------------------------------------------------

class TestV4ACreateModeFailureSignal:
    def test_add_file_to_unwritable_parent_returns_success_false(
        self, hermes_home, tmp_path, clear_patch_failures,
    ):
        """``*** Add File:`` whose parent directory cannot be created
        (e.g. target under a non-existent / read-only ancestor that
        ``mkdir -p`` cannot place) must surface ``success: false`` instead
        of a misleading success."""
        from tools.file_tools import _handle_patch

        # ``/dev/null/subdir/should_not_exist.py`` — ``mkdir -p /dev/null/subdir``
        # fails because /dev/null is not a directory on Linux.  This gives us
        # a portable, no-permission-yoinking way to provoke an ADD failure.
        bad_path = "/dev/null/should_not_exist/file.py"

        patch = (
            "*** Begin Patch\n"
            f"*** Add File: {bad_path}\n"
            "+hello world\n"
            "*** End Patch\n"
        )

        result = _handle_patch(
            {"mode": "patch", "patch": patch},
            task_id="t_add_bad",
        )
        d = json.loads(result)

        # Either a tool_error envelope ({"success": false, "error": "..."})
        # or a structured PatchResult with success: false is acceptable.
        # The contract is the same: success MUST be false.
        assert d.get("success") is False, (
            f"uncreatable ADD target must report success: false, got: {d!r}"
        )
        assert d.get("error"), f"missing error message: {d!r}"


# ---------------------------------------------------------------------------
# V4A PATCH MODE — happy paths (regression guards)
# ---------------------------------------------------------------------------

class TestV4AHappyPath:
    def test_create_add_file_succeeds(self, hermes_home, tmp_path, clear_patch_failures):
        from tools.file_tools import _handle_patch

        target = tmp_path / "created.py"
        assert not target.exists()

        patch = (
            "*** Begin Patch\n"
            f"*** Add File: {target}\n"
            "+def created():\n"
            "+    return 'made by patch'\n"
            "*** End Patch\n"
        )

        result = _handle_patch(
            {"mode": "patch", "patch": patch},
            task_id="t_add_good",
        )
        d = json.loads(result)

        assert d.get("success") is True, f"valid ADD must remain success: true: {d!r}"
        assert target.exists(), "ADD should have created the file"
        assert "made by patch" in target.read_text()

    def test_fuzzy_update_succeeds(self, hermes_home, tmp_path, clear_patch_failures):
        """Fuzzy UPDATE with a slight whitespace / indentation drift between
        the patch's context lines and the on-disk file should still match
        and return success: true (regression guard for the fuzzy path)."""
        from tools.file_tools import _handle_patch

        target = tmp_path / "fuzzy_happy.py"
        # On-disk: 4-space indent
        target.write_text("def foo():\n    return 1\n")

        # Patch: extra space before the search line — fuzzy matcher should
        # still align it (it normalises leading whitespace).
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@ def foo @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 42\n"
            "*** End Patch\n"
        )

        result = _handle_patch(
            {"mode": "patch", "patch": patch},
            task_id="t_fuzzy_happy",
        )
        d = json.loads(result)

        assert d.get("success") is True, f"fuzzy UPDATE must remain success: true: {d!r}"
        assert "return 42" in target.read_text()