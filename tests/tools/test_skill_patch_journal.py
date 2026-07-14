"""Tests for the skill write journal + background-review cooldown (local patch 10, F008).

Every successful mutating skill_manage action must land in the per-profile
.patch-journal.jsonl with enough content to revert it, and autonomous
background-review content writes are rate-limited per skill.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import (
    _background_review_cooldown_guard,
    _background_review_write_guard,
    skill_manage,
)


@contextmanager
def _skill_env(tmp_path):
    """Sandbox the skills dir AND the journal path to tmp_path."""
    journal = tmp_path / ".patch-journal.jsonl"
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
         patch("tools.skill_manager_tool._skill_journal_path", return_value=journal):
        yield journal


VALID_SKILL_CONTENT = """\
---
name: journal-test-skill
description: A test skill for journal testing.
---

# Journal Test Skill

Step 1: Do the thing.
"""

REWRITTEN_CONTENT = """\
---
name: journal-test-skill
description: Updated description.
---

# Journal Test Skill v2

Step 1: Do the new thing.
"""


def _journal_entries(journal):
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]


def _write_journal_entry(journal, skill, action="patch", origin="background_review", hours_ago=0.0):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    entry = {"ts": ts, "origin": origin, "action": action, "skill": skill, "file": "SKILL.md"}
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


class TestJournalWrites:
    def test_patch_journals_old_and_new_strings(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            skill_manage("create", "journal-test-skill", content=VALID_SKILL_CONTENT)
            skill_manage(
                "patch", "journal-test-skill",
                old_string="Do the thing.", new_string="Do the better thing.",
            )
            entries = _journal_entries(journal)
            actions = [e["action"] for e in entries]
            assert "create" in actions and "patch" in actions
            patch_entry = [e for e in entries if e["action"] == "patch"][0]
            assert patch_entry["skill"] == "journal-test-skill"
            assert patch_entry["old_string"] == "Do the thing."
            assert patch_entry["new_string"] == "Do the better thing."
            assert patch_entry["origin"] == "foreground"
            assert patch_entry["file"] == "SKILL.md"
            assert patch_entry["ts"]

    def test_edit_journals_prior_content(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            skill_manage("create", "journal-test-skill", content=VALID_SKILL_CONTENT)
            skill_manage("edit", "journal-test-skill", content=REWRITTEN_CONTENT)
            edit_entry = [e for e in _journal_entries(journal) if e["action"] == "edit"][0]
            assert edit_entry["prior_content"] == VALID_SKILL_CONTENT

    def test_failed_patch_not_journaled(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            skill_manage("create", "journal-test-skill", content=VALID_SKILL_CONTENT)
            skill_manage(
                "patch", "journal-test-skill",
                old_string="text that does not exist anywhere", new_string="x",
            )
            assert all(e["action"] != "patch" for e in _journal_entries(journal))

    def test_journal_failure_never_breaks_the_write(self, tmp_path):
        bad_journal = tmp_path / "no-such-dir-parent-is-a-file" / "j.jsonl"
        (tmp_path / "no-such-dir-parent-is-a-file").write_text("a file, not a dir")
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
             patch("tools.skill_manager_tool._skill_journal_path", return_value=bad_journal):
            result = json.loads(skill_manage("create", "journal-test-skill", content=VALID_SKILL_CONTENT))
            assert result["success"] is True
            result = json.loads(skill_manage(
                "patch", "journal-test-skill",
                old_string="Do the thing.", new_string="Do the better thing.",
            ))
            assert result["success"] is True


class TestBackgroundCooldown:
    def test_recent_bg_write_blocks(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            _write_journal_entry(journal, "journal-test-skill", hours_ago=1)
            with patch("tools.skill_provenance.is_background_review", return_value=True):
                guard = _background_review_cooldown_guard("journal-test-skill", "patch")
            assert guard is not None
            assert guard["success"] is False
            assert "cooldown" in guard["error"]

    def test_old_bg_write_allows(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            _write_journal_entry(journal, "journal-test-skill", hours_ago=48)
            guard = _background_review_cooldown_guard("journal-test-skill", "patch")
            assert guard is None

    def test_foreground_entries_do_not_count(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            _write_journal_entry(journal, "journal-test-skill", origin="foreground", hours_ago=1)
            guard = _background_review_cooldown_guard("journal-test-skill", "patch")
            assert guard is None

    def test_other_skill_entries_do_not_count(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            _write_journal_entry(journal, "some-other-skill", hours_ago=1)
            guard = _background_review_cooldown_guard("journal-test-skill", "patch")
            assert guard is None

    def test_delete_not_cooldown_limited(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            _write_journal_entry(journal, "journal-test-skill", hours_ago=1)
            guard = _background_review_cooldown_guard("journal-test-skill", "delete")
            assert guard is None

    def test_env_zero_disables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_BG_SKILL_WRITE_COOLDOWN_HOURS", "0")
        with _skill_env(tmp_path) as journal:
            _write_journal_entry(journal, "journal-test-skill", hours_ago=1)
            guard = _background_review_cooldown_guard("journal-test-skill", "patch")
            assert guard is None

    def test_write_guard_end_to_end_blocks_bg_repatch(self, tmp_path):
        """Full guard path: an unpinned local skill, fresh bg journal entry -> refusal."""
        with _skill_env(tmp_path) as journal:
            skill_manage("create", "journal-test-skill", content=VALID_SKILL_CONTENT)
            _write_journal_entry(journal, "journal-test-skill", hours_ago=1)
            skill_dir = tmp_path / "journal-test-skill"
            with patch("tools.skill_provenance.is_background_review", return_value=True):
                guard = _background_review_write_guard("journal-test-skill", skill_dir, "patch")
            assert guard is not None
            assert "cooldown" in guard["error"]

    def test_write_guard_foreground_unaffected(self, tmp_path):
        with _skill_env(tmp_path) as journal:
            skill_manage("create", "journal-test-skill", content=VALID_SKILL_CONTENT)
            _write_journal_entry(journal, "journal-test-skill", hours_ago=1)
            skill_dir = tmp_path / "journal-test-skill"
            guard = _background_review_write_guard("journal-test-skill", skill_dir, "patch")
            assert guard is None
