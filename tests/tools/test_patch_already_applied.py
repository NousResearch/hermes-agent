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

from tools.fuzzy_match import (
    _json_candidate_has_applied_string_value,
    is_already_applied,
)


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


class TestJsonCandidateHasAppliedStringValue:
    def test_exact_string_property_matches(self):
        content = '{"url": "https://correct.example/item", "state": "ready"}\n'
        candidate = '"url": "https://correct.example/item",'
        assert _json_candidate_has_applied_string_value(
            content, candidate, "https://correct.example/item"
        )

    @pytest.mark.parametrize(
        ("content", "candidate", "new_string"),
        [
            ('{"url": "correct"}\n', '"url": "correct"', "other"),
            ('{"count": 7}\n', '"count": 7', "7"),
            (
                '{"url": "correct"}\n',
                '"url": "wrong", "url": "correct"',
                "correct",
            ),
            ('{"url": "correct"}\n', '"url": "correct"\n"state": "ready"', "correct"),
            ('{"url": "correct"', '"url": "correct"', "correct"),
        ],
    )
    def test_non_proof_candidates_are_rejected(self, content, candidate, new_string):
        assert not _json_candidate_has_applied_string_value(
            content, candidate, new_string
        )


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

    def test_json_replay_is_noop_before_similarity_matching(self, workdir):
        f = workdir / "record.json"
        original = (
            '{\n'
            '  "service": {\n'
            '    "hashid": "synthetic-123",\n'
            '    "url": "https://correct.example/work-order/synthetic-123",\n'
            '    "status": "ready"\n'
            '  }\n'
            '}\n'
        )
        f.write_text(original, encoding="utf-8")

        r = _patch_tool(
            path=str(f),
            old_string="https://wrong.example/work-order/synthetic-123",
            new_string="https://correct.example/work-order/synthetic-123",
            task_id="t-applied-json",
        )

        assert r["success"] is True, r
        assert r.get("no_change") is True
        assert f.read_text(encoding="utf-8") == original
        assert json.loads(f.read_text(encoding="utf-8"))["service"]["url"].startswith(
            "https://correct.example/"
        )

    def test_unrelated_target_text_does_not_block_similarity_edit(self, workdir):
        f = workdir / "scoped.txt"
        original = (
            "BEGIN target\n"
            "current marker\n"
            "END target\n"
            "\n"
            "COMPLETELY REPLACED TARGET BLOCK\n"
        )
        f.write_text(original, encoding="utf-8")

        r = _patch_tool(
            path=str(f),
            old_string="BEGIN target\nold marker\nEND target",
            new_string="COMPLETELY REPLACED TARGET BLOCK",
            task_id="t-applied-scoped",
        )

        assert r["success"] is True, r
        assert r.get("no_change") is not True
        assert f.read_text(encoding="utf-8").count(
            "COMPLETELY REPLACED TARGET BLOCK"
        ) == 2

    @pytest.mark.parametrize("replace_all", [False, True])
    def test_target_retained_as_candidate_context_does_not_skip_edit(
        self, workdir, replace_all
    ):
        f = workdir / "retained-context.txt"
        original = (
            "BEGIN target\n"
            "RETAINED TARGET PHRASE\n"
            "current marker\n"
            "END target\n"
        )
        f.write_text(original, encoding="utf-8")

        r = _patch_tool(
            path=str(f),
            old_string=(
                "BEGIN target\n"
                "RETAINED TARGET PHRASE\n"
                "stale marker\n"
                "END target"
            ),
            new_string="RETAINED TARGET PHRASE",
            replace_all=replace_all,
            task_id=f"t-retained-context-{replace_all}",
        )

        assert r["success"] is True, r
        assert r.get("no_change") is not True
        assert f.read_text(encoding="utf-8") == "RETAINED TARGET PHRASE\n"

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

    def test_degenerate_identical_hunk_skipped_in_validation(self, workdir):
        """A hunk whose -/+ lines are identical is a no-op: the apply phase
        skips it, so validation must not fail the patch (previously it
        reached fuzzy_find_and_replace, whose identical-strings error names
        old_string/new_string — parameters that don't exist in patch mode).
        The short text also dodges is_already_applied's >=8-char rescue."""
        f = workdir / "degen.py"
        f.write_text("A = 1\nB = 2\n")
        patch_content = (
            "*** Begin Patch\n"
            f"*** Update File: {f}\n"
            "-A = 1\n"
            "+A = 1\n"
            "@@ B @@\n"
            "-B = 2\n"
            "+B = 3\n"
            "*** End Patch\n"
        )
        r = _patch_tool(mode="patch", patch=patch_content, task_id="t-v4a")
        assert r["success"] is True, r
        text = f.read_text()
        assert "A = 1" in text  # degenerate hunk left intact
        assert "B = 3" in text  # live hunk applied
