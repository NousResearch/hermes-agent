"""Tests for _log_patch() — skill patch/edit history logging."""

import json
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import _log_patch


SKILL_NAME = "test-skill"


def _make_skill_dir(tmp_path):
    skill_dir = tmp_path / "skills" / SKILL_NAME
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Test", encoding="utf-8")
    return skill_dir


def _read_history(skill_dir):
    history = skill_dir / "patch_history.jsonl"
    if not history.exists():
        return []
    return [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]


class TestLogPatch:
    def test_patch_creates_record(self, tmp_path):
        skill_dir = _make_skill_dir(tmp_path)
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "patch", None, "old text", "new text")
        records = _read_history(skill_dir)
        assert len(records) == 1
        r = records[0]
        assert r["schema_version"] == 1
        assert r["skill"] == SKILL_NAME
        assert r["action"] == "patch"
        assert r["file"] == "SKILL.md"
        assert r["old_text_preview"] == "old text"
        assert r["new_text_preview"] == "new text"
        assert "timestamp" in r

    def test_edit_creates_record_with_content(self, tmp_path):
        skill_dir = _make_skill_dir(tmp_path)
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "edit", old_text="old content", new_text="new content")
        records = _read_history(skill_dir)
        assert len(records) == 1
        assert records[0]["action"] == "edit"
        assert records[0]["old_text_preview"] == "old content"
        assert records[0]["new_text_preview"] == "new content"

    def test_empty_string_replacement_is_logged(self, tmp_path):
        """Replacing text with empty string should still log previews."""
        skill_dir = _make_skill_dir(tmp_path)
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "patch", None, "delete me", "")
        records = _read_history(skill_dir)
        assert len(records) == 1
        assert records[0]["old_text_preview"] == "delete me"
        assert records[0]["new_text_preview"] == ""

    def test_none_text_omits_preview_fields(self, tmp_path):
        skill_dir = _make_skill_dir(tmp_path)
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "edit")
        records = _read_history(skill_dir)
        assert "old_text_preview" not in records[0]
        assert "new_text_preview" not in records[0]

    def test_truncation_at_200_chars(self, tmp_path):
        skill_dir = _make_skill_dir(tmp_path)
        long_text = "x" * 500
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "patch", None, long_text, long_text)
        records = _read_history(skill_dir)
        assert len(records[0]["old_text_preview"]) == 200
        assert len(records[0]["new_text_preview"]) == 200

    def test_multiple_appends_preserve_prior(self, tmp_path):
        skill_dir = _make_skill_dir(tmp_path)
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "patch", None, "a", "b")
            _log_patch(SKILL_NAME, "edit", old_text="c", new_text="d")
            _log_patch(SKILL_NAME, "patch", "refs/notes.md", "e", "f")
        records = _read_history(skill_dir)
        assert len(records) == 3
        assert records[0]["action"] == "patch"
        assert records[1]["action"] == "edit"
        assert records[2]["file"] == "refs/notes.md"

    def test_missing_skill_does_not_crash(self):
        with patch("tools.skill_manager_tool._find_skill", return_value=None):
            _log_patch("nonexistent", "patch", None, "a", "b")  # should not raise

    def test_write_failure_does_not_crash(self, tmp_path):
        """Logging failure must not break the skill edit/patch operation."""
        skill_dir = _make_skill_dir(tmp_path)
        # Make history file a directory to force write error
        (skill_dir / "patch_history.jsonl").mkdir()
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "patch", None, "a", "b")  # should not raise

    def test_unicode_content(self, tmp_path):
        skill_dir = _make_skill_dir(tmp_path)
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "patch", None, "stary tekst ąęół", "nowy tekst 日本語")
        records = _read_history(skill_dir)
        assert "ąęół" in records[0]["old_text_preview"]
        assert "日本語" in records[0]["new_text_preview"]

    def test_custom_file_path(self, tmp_path):
        skill_dir = _make_skill_dir(tmp_path)
        with patch("tools.skill_manager_tool._find_skill", return_value={"path": skill_dir}):
            _log_patch(SKILL_NAME, "patch", "scripts/helper.py", "old", "new")
        records = _read_history(skill_dir)
        assert records[0]["file"] == "scripts/helper.py"
