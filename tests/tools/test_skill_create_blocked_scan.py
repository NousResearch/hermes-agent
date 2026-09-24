"""_create_skill must not destroy pre-existing directories when a security scan blocks.

Regression coverage for issue #119534: ``_create_skill`` used to ``rmtree`` the whole
skill directory on a blocked scan. A name colliding with a pre-existing non-skill
directory (no SKILL.md, so ``_find_skill`` misses it) — user data under ``skills/``,
a partially-deleted skill, or a dir under ``skills.create_dir`` — then lost its
entire contents. The fix snapshots the pre-state and, on a blocked scan, removes
only what the create call itself added: the same contract ``_guarded_write``
applies to edit/patch/write_file.

Arms that fail against the unfixed source:
- pre-existing dir with a data file -> data file must survive (unfixed: rmtree'd)
- pre-existing SKILL.md that _find_skill missed -> restored byte-for-byte (unfixed: rmtree'd)
"""

import json
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import _create_skill

BLOCKED_SCAN_ERROR = "Security scan blocked this skill (test): blocked"


def _blocked_scan(skill_dir):
    """Stub scanner returning the same error shape the real gate produces."""
    return BLOCKED_SCAN_ERROR


VALID_SKILL_CONTENT = """\
---
name: colliding-name
description: A skill used by the collision regression tests.
---

# Colliding Name

Step 1: Do the thing.
"""


@pytest.fixture
def scan_blocks(monkeypatch):
    """Force the opt-in security scan to block, regardless of guard config."""
    monkeypatch.setattr(
        "tools.skill_manager_tool._security_scan_skill", _blocked_scan)


class TestCreateSkillBlockedScanPreservesPreExisting:
    def test_pre_existing_data_file_survives_blocked_scan(self, tmp_path, scan_blocks):
        """A name colliding with a pre-existing non-skill directory: the data file
        inside must survive the blocked scan (unfixed code rmtree'd the whole dir)."""
        skill_dir = tmp_path / "colliding-name"
        skill_dir.mkdir()
        data_file = skill_dir / "user-notes.txt"
        data_file.write_text("irreplaceable user data", encoding="utf-8")

        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            result = _create_skill("colliding-name", VALID_SKILL_CONTENT)

        assert result["success"] is False
        assert "Security scan blocked" in result["error"]
        assert data_file.exists(), "pre-existing data file must survive a blocked scan"
        assert data_file.read_text(encoding="utf-8") == "irreplaceable user data"
        assert not (skill_dir / "SKILL.md").exists(), "the written SKILL.md must be unlinked"

    def test_pre_existing_skill_md_restored_byte_for_byte(self, tmp_path, scan_blocks):
        """A pre-existing SKILL.md that _find_skill misses (simulating an excluded path
        or a race): on a blocked scan it is restored byte-for-byte, not deleted with
        the directory."""
        skill_dir = tmp_path / "colliding-name"
        skill_dir.mkdir()
        pre_md = skill_dir / "SKILL.md"
        original = "---\nname: colliding-name\ndescription: Original.\n---\n\n# Original\n"
        pre_md.write_text(original, encoding="utf-8", newline="")
        original_bytes = pre_md.read_bytes()

        # Simulate the miss the issue describes (excluded path / race): the discovery
        # layer reports no skill, while the dir + SKILL.md exist on disk.
        with patch("tools.skill_manager_tool._find_skill", return_value=None), \
             patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            result = _create_skill("colliding-name", VALID_SKILL_CONTENT)

        assert result["success"] is False
        assert "Security scan blocked" in result["error"]
        assert pre_md.read_bytes() == original_bytes, \
            "pre-existing SKILL.md must be restored byte-for-byte"

    def test_fresh_dir_still_removed_on_blocked_scan(self, tmp_path, scan_blocks):
        """The non-collision path is unchanged: a directory this call created is fully
        removed when the scan blocks (no half-created skill left behind)."""
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            result = _create_skill("fresh-skill", VALID_SKILL_CONTENT)

        assert result["success"] is False
        assert "Security scan blocked" in result["error"]
        assert not (tmp_path / "fresh-skill").exists(), "fresh dir must still be fully removed"

    def test_end_to_end_create_through_apply_skill_pending(self, tmp_path, scan_blocks):
        """End-to-end arm through skill_manage(action='create') via the approval-replay
        bypass (apply_skill_pending), mirroring the real /skills approve path."""
        from tools.skill_manager_tool import apply_skill_pending

        skill_dir = tmp_path / "colliding-name"
        skill_dir.mkdir()
        data_file = skill_dir / "evidence.bin"
        data_file.write_bytes(b"\x00\x01\x02keep me")

        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            payload = {"action": "create", "name": "colliding-name",
                       "content": VALID_SKILL_CONTENT}
            result = json.loads(apply_skill_pending(payload))

        assert result["success"] is False
        assert "Security scan blocked" in result["error"]
        assert data_file.exists()
        assert data_file.read_bytes() == b"\x00\x01\x02keep me"
        assert not (skill_dir / "SKILL.md").exists()

    def test_collision_with_unrelated_scan_target(self, tmp_path, scan_blocks):
        """The stray-flagged-file arm: the scanner inspects pre-existing dir contents,
        so a colliding dir with its own flagged file triggers the same wipe. The
        pre-existing file must survive regardless of what the scanner thought of it."""
        skill_dir = tmp_path / "colliding-name"
        skill_dir.mkdir()
        pre_file = skill_dir / "suspicious-helper.sh"
        pre_file.write_text("# pre-existing\n", encoding="utf-8")

        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            result = _create_skill("colliding-name", VALID_SKILL_CONTENT)

        assert result["success"] is False
        assert pre_file.exists()
        assert pre_file.read_text(encoding="utf-8") == "# pre-existing\n"
        assert not (skill_dir / "SKILL.md").exists()

    def test_unreadable_pre_existing_skill_md_fails_closed_before_write(self, tmp_path, scan_blocks,
                                                                        monkeypatch):
        """An unreadable pre-existing SKILL.md fails closed BEFORE any write: the caller
        gets an error and the pre-existing file is never overwritten by the new skill."""
        skill_dir = tmp_path / "colliding-name"
        skill_dir.mkdir()
        pre_md = skill_dir / "SKILL.md"
        original = "---\nname: colliding-name\ndescription: Original.\n---\n\n# Original\n"
        pre_md.write_text(original, encoding="utf-8", newline="")
        original_bytes = pre_md.read_bytes()

        def raise_permission_error(self, *a, **kw):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(type(pre_md), "read_bytes", raise_permission_error)

        with patch("tools.skill_manager_tool._find_skill", return_value=None), \
             patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            result = _create_skill("colliding-name", VALID_SKILL_CONTENT)

        monkeypatch.undo()
        assert result["success"] is False
        assert "Cannot read existing SKILL.md" in result["error"]
        assert pre_md.read_bytes() == original_bytes
