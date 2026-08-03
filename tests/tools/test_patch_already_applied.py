"""Tests for already-applied patch detection (success-shaped no-op).

Production mining showed the #1 patch failure class is a re-send of an
edit that already landed: `old_string and new_string are identical` (299
occurrences in a 250k-window) plus a share of hunk-not-found errors where
new text is already present. These previously errored, sending the model
into re-read/re-patch loops; they now return success with no_change=True.
"""

import json
import os
import tempfile

import pytest

from tools.fuzzy_match import is_already_applied


class TestIsAlreadyApplied:
    def test_identical_strings_present_in_content(self):
        assert is_already_applied("x = compute_value(1)\n", "compute_value(1)", "compute_value(1)")

    def test_identical_strings_absent_from_content(self):
        assert not is_already_applied("y = 2\n", "compute_value(1)", "compute_value(1)")

    def test_old_gone_new_present(self):
        content = "def new_name(x):\n    return x\n"
        assert is_already_applied(content, "def old_name(x):", "def new_name(x):")

    def test_old_still_present_means_half_applied(self):
        content = "def old_name(x):\n    pass\n\ndef new_name(x):\n    pass\n"
        assert not is_already_applied(content, "def old_name(x):", "def new_name(x):")

    def test_new_absent_not_applied(self):
        assert not is_already_applied("def old_name(x):\n", "def old_name(x):", "def new_name(x):")

    def test_trivial_new_string_never_matches(self):
        # A short target ("x = 1") appearing by coincidence must not mask a
        # genuinely broken edit.
        assert not is_already_applied("x = 1\n", "y = 2", "x = 1")

    def test_exact_presence_required(self):
        content = "def new_name( x ):\n"  # whitespace differs
        assert not is_already_applied(content, "def old_name(x):", "def new_name(x):")


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path


def _patch_tool(**kwargs):
    from tools.file_tools import patch_tool
    return json.loads(patch_tool(**kwargs))


class TestPatchReplaceAlreadyApplied:
    def test_identical_old_new_present_is_success_noop(self, workdir):
        f = workdir / "a.py"
        f.write_text("value = compute_total(items)\n")
        r = _patch_tool(path=str(f), old_string="value = compute_total(items)",
                        new_string="value = compute_total(items)", task_id="t-applied")
        assert r["success"] is True
        assert r.get("no_change") is True
        assert "already" in r["note"]
        assert f.read_text() == "value = compute_total(items)\n"

    def test_replay_of_landed_edit_is_success_noop(self, workdir):
        # old_string is entirely gone (no approximate remnant for the fuzzy
        # chain to latch onto) while new_string is present verbatim.
        f = workdir / "b.py"
        f.write_text("import os\n\nRETRY_LIMIT_SECONDS = 30\n")
        r = _patch_tool(path=str(f), old_string="TIMEOUT_WINDOW_MS = 9000",
                        new_string="RETRY_LIMIT_SECONDS = 30", task_id="t-applied")
        assert r["success"] is True
        assert r.get("no_change") is True

    def test_genuine_no_match_still_errors(self, workdir):
        f = workdir / "c.py"
        f.write_text("something_else = 1\n")
        r = _patch_tool(path=str(f), old_string="def missing_function():",
                        new_string="def replacement_function():", task_id="t-applied")
        assert "error" in r

    def test_identical_but_absent_still_errors(self, workdir):
        f = workdir / "d.py"
        f.write_text("unrelated = True\n")
        r = _patch_tool(path=str(f), old_string="def not_here_function():",
                        new_string="def not_here_function():", task_id="t-applied")
        assert "error" in r

    def test_half_applied_rename_still_errors(self, workdir):
        # Both old and new text present: NOT already-applied. The identical
        # old/new strings short-circuit before any fuzzy matching, and the
        # old text still being present must block the no-op path.
        f = workdir / "e.py"
        f.write_text("def old_fn_name():\n    pass\n\ndef new_fn_variant():\n    pass\n")
        from tools.fuzzy_match import is_already_applied
        assert not is_already_applied(f.read_text(), "def old_fn_name():", "def new_fn_variant():")


class TestV4AAlreadyApplied:
    def test_already_applied_hunk_skipped_in_multi_hunk_patch(self, workdir):
        f = workdir / "mod.py"
        f.write_text(
            "def already_renamed_helper(x):\n"
            "    return x * 2\n"
            "\n"
            "def second_helper(y):\n"
            "    return y + 1\n"
        )
        patch_content = (
            "*** Begin Patch\n"
            f"*** Update File: {f}\n"
            "@@ def already_renamed_helper @@\n"
            "-def old_helper_name(x):\n"
            "+def already_renamed_helper(x):\n"
            "@@ def second_helper @@\n"
            "-    return y + 1\n"
            "+    return y + 2\n"
            "*** End Patch\n"
        )
        r = _patch_tool(mode="patch", patch=patch_content, task_id="t-v4a")
        assert r["success"] is True, r
        text = f.read_text()
        assert "return y + 2" in text            # live hunk applied
        assert "already_renamed_helper" in text  # no-op hunk left intact

    def test_fully_applied_patch_is_noop_success(self, workdir):
        f = workdir / "done.py"
        f.write_text("STATUS = 'migrated_to_v2_schema'\n")
        patch_content = (
            "*** Begin Patch\n"
            f"*** Update File: {f}\n"
            "-STATUS = 'legacy_v1_schema'\n"
            "+STATUS = 'migrated_to_v2_schema'\n"
            "*** End Patch\n"
        )
        r = _patch_tool(mode="patch", patch=patch_content, task_id="t-v4a")
        assert r["success"] is True, r
        assert f.read_text() == "STATUS = 'migrated_to_v2_schema'\n"
