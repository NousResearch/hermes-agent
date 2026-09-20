"""Operator-locked policy regions — the self-patch guard.

On 2026-06-18 a live self-patch silently REVERSED the ``--wait-seconds`` policy in
``positions-optimize/SKILL.md``: it deleted the "120s is cron-unsafe" rule, asserted the
opposite, and cited its own runs as evidence. Nobody was alerted; it surfaced by chance
during an unrelated deploy. These tests pin the guard that stops a recurrence.

The contract, in one line: **the locked bytes are operator property.** The agent may edit
anything outside the markers; it may not rewrite, drop, reorder, unbalance or mint them.
An operator still can, through git or the authenticated dashboard editor.
"""

import logging
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import (
    _create_skill,
    _delete_skill,
    _edit_skill,
    _patch_skill,
    _remove_file,
    _write_file,
    _locked_region_violation,
    _file_has_lock_marker,
    operator_authority,
)


@contextmanager
def _skill_dir(tmp_path):
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


LOCKED_POLICY = """\
---
name: positions-optimize
description: Use when optimizing positions. Runs the executor against open positions.
---

# Positions Optimize

## Calibration notes

Nothing yet.

<!-- operator-locked -->
- Keep the executor's --wait-seconds at 30; treat 120s as cron-unsafe.
<!-- /operator-locked -->

## Appendix
"""

UNLOCKED_POLICY = """\
---
name: plain-skill
description: Use when doing the plain thing. Does the plain thing.
---

# Plain Skill

Step 1: do the thing.
"""

# The exact 2026-06-18 reversal: the rule deleted and inverted, inside the markers.
REVERSED_POLICY = LOCKED_POLICY.replace(
    "- Keep the executor's --wait-seconds at 30; treat 120s as cron-unsafe.",
    "- Raise --wait-seconds to 120; my last 14 runs show 30s truncates fills.")

SKILL_MD = "SKILL.md"


def _make_locked_skill(tmp_path, name="positions-optimize", content=LOCKED_POLICY):
    """Author the locked skill the way an OPERATOR does — out of band, on disk."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / SKILL_MD).write_text(content, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# The incident itself: every write action refuses to touch a locked region
# ---------------------------------------------------------------------------

class TestLockedRegionIsRefused:
    def test_edit_refuses_reversing_locked_policy(self, tmp_path):
        """The 2026-06-18 regression, verbatim."""
        d = _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            result = _edit_skill("positions-optimize", REVERSED_POLICY)
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        # and the policy survived on disk, byte for byte
        assert "treat 120s as cron-unsafe" in (d / SKILL_MD).read_text()
        assert "Raise --wait-seconds to 120" not in (d / SKILL_MD).read_text()

    def test_patch_refuses_editing_inside_the_region(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            result = _patch_skill(
                "positions-optimize",
                "- Keep the executor's --wait-seconds at 30; treat 120s as cron-unsafe.",
                "- Raise --wait-seconds to 120.")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "treat 120s as cron-unsafe" in (d / SKILL_MD).read_text()

    def test_patch_refuses_dropping_the_whole_region(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        locked_block = (
            "<!-- operator-locked -->\n"
            "- Keep the executor's --wait-seconds at 30; treat 120s as cron-unsafe.\n"
            "<!-- /operator-locked -->\n")
        with _skill_dir(tmp_path):
            result = _patch_skill("positions-optimize", locked_block, "")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "treat 120s as cron-unsafe" in (d / SKILL_MD).read_text()

    def test_patch_refuses_stripping_only_the_markers(self, tmp_path):
        """Unlocking in place: keep the text, delete the fence. Count check catches it."""
        d = _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            result = _patch_skill("positions-optimize", "<!-- /operator-locked -->", "")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "<!-- /operator-locked -->" in (d / SKILL_MD).read_text()

    def test_write_file_refuses_clobbering_a_locked_supporting_file(self, tmp_path):
        d = _make_locked_skill(tmp_path, content=UNLOCKED_POLICY.replace(
            "name: plain-skill", "name: positions-optimize"))
        ref = d / "references" / "policy.md"
        ref.parent.mkdir(parents=True)
        ref.write_text(
            "<!-- operator-locked -->\nNever trade past the book-flat gate.\n"
            "<!-- /operator-locked -->\n", encoding="utf-8")
        with _skill_dir(tmp_path):
            result = _write_file("positions-optimize", "references/policy.md", "anything goes now\n")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "book-flat gate" in ref.read_text()

    def test_remove_file_refuses_a_locked_supporting_file(self, tmp_path):
        d = _make_locked_skill(tmp_path, content=UNLOCKED_POLICY.replace(
            "name: plain-skill", "name: positions-optimize"))
        ref = d / "references" / "policy.md"
        ref.parent.mkdir(parents=True)
        ref.write_text(
            "<!-- operator-locked -->\nNever trade past the book-flat gate.\n"
            "<!-- /operator-locked -->\n", encoding="utf-8")
        with _skill_dir(tmp_path):
            result = _remove_file("positions-optimize", "references/policy.md")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert ref.exists()

    def test_delete_refuses_a_skill_carrying_locked_policy(self, tmp_path):
        """Deleting the skill drops the policy just as silently as rewriting it."""
        d = _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            result = _delete_skill("positions-optimize", absorbed_into="")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "SKILL.md" in result["error"]
        assert d.exists()

    def test_delete_names_every_locked_file(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        ref = d / "references" / "gates.md"
        ref.parent.mkdir(parents=True)
        ref.write_text("<!-- operator-locked -->\nBook-flat gate.\n<!-- /operator-locked -->\n",
                       encoding="utf-8")
        with _skill_dir(tmp_path):
            result = _delete_skill("positions-optimize", absorbed_into="")
        assert result["success"] is False
        assert "SKILL.md" in result["error"]
        assert "references/gates.md" in result["error"]


# ---------------------------------------------------------------------------
# Self-improvement must keep working — the guard is narrow on purpose
# ---------------------------------------------------------------------------

class TestOutsideTheMarkersStillWorks:
    def test_patch_outside_the_region_succeeds(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            result = _patch_skill("positions-optimize", "Nothing yet.",
                                  "30s wait held across 14 cron fires on 2026-09-19.")
        assert result["success"] is True, result.get("error")
        text = (d / SKILL_MD).read_text()
        assert "30s wait held across 14 cron fires" in text
        assert "treat 120s as cron-unsafe" in text  # locked bytes untouched

    def test_edit_rewriting_everything_outside_succeeds(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        rewritten = LOCKED_POLICY.replace("## Appendix", "## Appendix\n\nRewritten wholesale.")
        rewritten = rewritten.replace("Nothing yet.", "Plenty now.")
        with _skill_dir(tmp_path):
            result = _edit_skill("positions-optimize", rewritten)
        assert result["success"] is True, result.get("error")
        assert "Rewritten wholesale." in (d / SKILL_MD).read_text()

    def test_write_and_remove_unlocked_supporting_file_succeed(self, tmp_path):
        _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            written = _write_file("positions-optimize", "references/notes.md", "calibration\n")
            removed = _remove_file("positions-optimize", "references/notes.md")
        assert written["success"] is True, written.get("error")
        assert removed["success"] is True, removed.get("error")

    def test_unlocked_skill_is_fully_editable_and_deletable(self, tmp_path):
        with _skill_dir(tmp_path):
            assert _create_skill("plain-skill", UNLOCKED_POLICY)["success"] is True
            assert _edit_skill("plain-skill", UNLOCKED_POLICY.replace(
                "Step 1: do the thing.", "Step 1: do it better."))["success"] is True
            assert _delete_skill("plain-skill", absorbed_into="")["success"] is True


# ---------------------------------------------------------------------------
# Forgery: the agent may not mint operator authority for itself
# ---------------------------------------------------------------------------

class TestForgery:
    def test_create_minting_lock_markers_is_refused(self, tmp_path):
        forged = UNLOCKED_POLICY.replace(
            "Step 1: do the thing.",
            "<!-- operator-locked -->\nStep 1: never let anyone change this.\n"
            "<!-- /operator-locked -->")
        with _skill_dir(tmp_path):
            result = _create_skill("plain-skill", forged)
        assert result["success"] is False
        assert "operator-lock markers" in result["error"]
        assert not (tmp_path / "plain-skill").exists()  # nothing left on disk

    def test_edit_minting_lock_markers_is_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("plain-skill", UNLOCKED_POLICY)
            result = _edit_skill("plain-skill", UNLOCKED_POLICY.replace(
                "Step 1: do the thing.",
                "<!-- operator-locked -->\nStep 1: mine now.\n<!-- /operator-locked -->"))
        assert result["success"] is False
        assert "operator-lock markers" in result["error"]

    def test_write_file_minting_lock_markers_is_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("plain-skill", UNLOCKED_POLICY)
            result = _write_file("plain-skill", "references/mine.md",
                                 "<!-- operator-locked -->\nmine\n<!-- /operator-locked -->\n")
        assert result["success"] is False
        assert "operator-lock markers" in result["error"]

    def test_adding_a_second_lock_to_an_already_locked_file_is_refused(self, tmp_path):
        _make_locked_skill(tmp_path)
        extra = LOCKED_POLICY.replace(
            "## Appendix",
            "<!-- operator-locked -->\nAlso mine.\n<!-- /operator-locked -->\n\n## Appendix")
        with _skill_dir(tmp_path):
            result = _edit_skill("positions-optimize", extra)
        assert result["success"] is False
        assert "operator-locked" in result["error"]


# ---------------------------------------------------------------------------
# Malformed layouts — the bypass upstream review found on #51258
# ---------------------------------------------------------------------------

class TestMalformedMarkersFailClosed:
    """A lone marker makes both region lists empty, so region equality holds
    vacuously: counting alone let the marker be relocated and the surrounding
    text rewritten. Layout is validated instead, and refused fail-closed."""

    def test_lone_opener_relocation_is_refused(self, tmp_path):
        original = "# Skill\n\n<!-- operator-locked -->\nkeep 30s\n\n## Notes\n"
        moved = "# Skill\n\nkeep 120s actually\n\n## Notes\n<!-- operator-locked -->\n"
        assert _locked_region_violation(original, moved) is not None

    def test_lone_closer_relocation_is_refused(self, tmp_path):
        original = "# Skill\n\n<!-- /operator-locked -->\nkeep 30s\n"
        moved = "# Skill\n\nkeep 120s\n<!-- /operator-locked -->\n"
        assert _locked_region_violation(original, moved) is not None

    def test_lone_marker_file_refuses_even_an_identical_write(self, tmp_path):
        """Fail closed: the tool cannot prove anything about a malformed file."""
        original = "# Skill\n<!-- operator-locked -->\nkeep 30s\n"
        err = _locked_region_violation(original, original)
        assert err is not None
        assert "malformed" in err

    def test_nested_open_markers_are_refused(self, tmp_path):
        original = ("<!-- operator-locked -->\nA\n<!-- operator-locked -->\nB\n"
                    "<!-- /operator-locked -->\n")
        assert _locked_region_violation(original, original + "tail\n") is not None

    def test_close_before_open_is_refused(self, tmp_path):
        original = ("<!-- /operator-locked -->\nX\n<!-- operator-locked -->\nY\n"
                    "<!-- /operator-locked -->\n")
        assert _locked_region_violation(original, original + "tail\n") is not None

    def test_edit_cannot_leave_the_file_malformed(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        unbalanced = LOCKED_POLICY.replace("<!-- /operator-locked -->\n", "")
        with _skill_dir(tmp_path):
            result = _edit_skill("positions-optimize", unbalanced)
        assert result["success"] is False
        assert "<!-- /operator-locked -->" in (d / SKILL_MD).read_text()

    def test_well_formed_multi_region_file_is_still_editable_outside(self, tmp_path):
        """The fail-closed rule must not catch legitimate multi-lock files —
        a single SKILL.md can carry several locked regions."""
        content = LOCKED_POLICY
        for i in range(4):
            content = content.replace(
                "## Appendix",
                f"<!-- operator-locked: rule-{i} -->\nrule {i}\n<!-- /operator-locked -->\n\n## Appendix")
        d = _make_locked_skill(tmp_path, content=content)
        with _skill_dir(tmp_path):
            result = _patch_skill("positions-optimize", "Nothing yet.", "Something now.")
        assert result["success"] is True, result.get("error")
        text = (d / SKILL_MD).read_text()
        assert text.count("<!-- operator-locked") == 5
        assert text.count("<!-- /operator-locked -->") == 5


# ---------------------------------------------------------------------------
# The operator's own path stays open (upstream review finding #1)
# ---------------------------------------------------------------------------

class TestOperatorAuthority:
    """`hermes_cli/web_routers/skills.py` calls `_create_skill`/`_edit_skill` directly for
    authenticated dashboard writes — "an authenticated dashboard write IS the user". Without
    an explicit operator scope the guard refuses the very flow that authors and lifts locks."""

    def test_dashboard_can_author_a_lock(self, tmp_path):
        forged_for_agent = UNLOCKED_POLICY.replace(
            "Step 1: do the thing.",
            "<!-- operator-locked -->\nStep 1: policy.\n<!-- /operator-locked -->")
        with _skill_dir(tmp_path):
            with operator_authority():
                result = _create_skill("plain-skill", forged_for_agent)
        assert result["success"] is True, result.get("error")

    def test_dashboard_can_lift_a_lock(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            with operator_authority():
                result = _edit_skill("positions-optimize", REVERSED_POLICY)
        assert result["success"] is True, result.get("error")
        assert "Raise --wait-seconds to 120" in (d / SKILL_MD).read_text()

    def test_authority_does_not_leak_out_of_scope(self, tmp_path):
        d = _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            with operator_authority():
                pass
            result = _edit_skill("positions-optimize", REVERSED_POLICY)
        assert result["success"] is False
        assert "treat 120s as cron-unsafe" in (d / SKILL_MD).read_text()

    def test_dashboard_can_delete_a_locked_skill(self, tmp_path):
        _make_locked_skill(tmp_path)
        with _skill_dir(tmp_path):
            with operator_authority():
                result = _delete_skill("positions-optimize", absorbed_into="")
        assert result["success"] is True, result.get("error")


# ---------------------------------------------------------------------------
# The refusal is never silent — killing the silence is the core of the guard
# ---------------------------------------------------------------------------

class TestAuditLogging:
    def test_refusal_is_logged_with_skill_and_action(self, tmp_path, caplog):
        _make_locked_skill(tmp_path)
        with caplog.at_level(logging.WARNING, logger="tools.skill_manager_tool"):
            with _skill_dir(tmp_path):
                _edit_skill("positions-optimize", REVERSED_POLICY)
        assert any(
            r.levelno == logging.WARNING
            and "operator-locked" in r.getMessage()
            and "positions-optimize" in r.getMessage()
            for r in caplog.records), caplog.text

    def test_delete_refusal_is_logged(self, tmp_path, caplog):
        _make_locked_skill(tmp_path)
        with caplog.at_level(logging.WARNING, logger="tools.skill_manager_tool"):
            with _skill_dir(tmp_path):
                _delete_skill("positions-optimize", absorbed_into="")
        assert any("operator-locked file(s)" in r.getMessage() for r in caplog.records), caplog.text

    def test_allowed_touch_of_a_locked_file_is_logged(self, tmp_path, caplog):
        """The edit is legal (outside the markers) but byte-locking cannot see
        contradiction, so the touch is breadcrumbed rather than passing unseen."""
        _make_locked_skill(tmp_path)
        with caplog.at_level(logging.INFO, logger="tools.skill_manager_tool"):
            with _skill_dir(tmp_path):
                result = _patch_skill("positions-optimize", "Nothing yet.", "Calibrated.")
        assert result["success"] is True, result.get("error")
        assert any("outside its operator-locked" in r.getMessage() for r in caplog.records), caplog.text

    def test_no_audit_noise_for_unlocked_skills(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="tools.skill_manager_tool"):
            with _skill_dir(tmp_path):
                _create_skill("plain-skill", UNLOCKED_POLICY)
                _edit_skill("plain-skill", UNLOCKED_POLICY.replace("thing.", "thing!"))
        assert not any("operator-locked" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Scanning is bounded — a delete must never die on a large supporting file
# ---------------------------------------------------------------------------

class TestBoundedScan:
    def test_marker_found_across_a_chunk_boundary(self, tmp_path):
        """Chunks overlap by more than the longest marker, so none is split."""
        from tools.skill_manager_tool import _LOCK_SCAN_CHUNK
        f = tmp_path / "big.md"
        f.write_text("x" * (_LOCK_SCAN_CHUNK - 10) + "<!-- operator-locked -->\ny\n"
                     "<!-- /operator-locked -->\n", encoding="utf-8")
        assert _file_has_lock_marker(f) is True

    def test_binary_file_does_not_crash_the_scan(self, tmp_path):
        f = tmp_path / "blob.bin"
        f.write_bytes(bytes(range(256)) * 4096)
        assert _file_has_lock_marker(f) is False

    def test_delete_survives_a_large_unlocked_supporting_file(self, tmp_path):
        from tools.skill_manager_tool import _LOCK_SCAN_CHUNK
        d = _make_locked_skill(tmp_path, name="plain-skill", content=UNLOCKED_POLICY)
        (d / "assets").mkdir()
        (d / "assets" / "big.bin").write_bytes(b"\x00" * (_LOCK_SCAN_CHUNK * 2))
        with _skill_dir(tmp_path):
            result = _delete_skill("plain-skill", absorbed_into="")
        assert result["success"] is True, result.get("error")
