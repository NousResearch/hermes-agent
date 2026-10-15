"""Tests for tools/skills_sync_optional.py — official optional-skill restore."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from tools.skills_sync_optional import restore_official_optional_skill

SKILL_MD_OFFICIAL = "---\nname: pytorch-fsdp\n---\n# official source\n"
SKILL_MD_LOCAL = "---\nname: pytorch-fsdp\n---\n# locally modified copy\n"


def _write_skill(skill_dir: Path, text: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")


class TestRestoreOfficialOptional:
    def _patches(self, optional_dir: Path, skills_dir: Path) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch("tools.skills_sync._get_optional_dir", return_value=optional_dir)
        )
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        return stack

    def test_restore_backup_is_not_rediscovered_as_active_skill(self, tmp_path):
        """#126612: repairing a modified official optional skill must leave the restore
        backup discoverable on disk for rollback but invisible to skill discovery —
        skill_view's candidate scan sees exactly one copy instead of a name collision."""
        from agent.skill_utils import is_excluded_skill_path, iter_skill_index_files
        from tools.skills_tool import _collect_skill_candidates

        optional_dir = tmp_path / "optional-skills"
        _write_skill(optional_dir / "ml" / "pytorch-fsdp", SKILL_MD_OFFICIAL)
        skills_dir = tmp_path / "user_skills"
        _write_skill(skills_dir / "ml" / "pytorch-fsdp", SKILL_MD_LOCAL)

        with self._patches(optional_dir, skills_dir):
            result = restore_official_optional_skill("pytorch-fsdp", restore=True)

        assert result["ok"] is True
        assert result["restored"] == ["pytorch-fsdp"]
        assert result["backed_up"] == ["ml/pytorch-fsdp"]
        backup_md = Path(result["backup_dir"]) / "ml" / "pytorch-fsdp" / "SKILL.md"
        assert backup_md.exists()  # rollback copy survives on disk...
        assert SKILL_MD_LOCAL in backup_md.read_text(encoding="utf-8")
        assert is_excluded_skill_path(backup_md)  # ...but never counts as a skill path
        active_md = skills_dir / "ml" / "pytorch-fsdp" / "SKILL.md"
        assert SKILL_MD_OFFICIAL in active_md.read_text(encoding="utf-8")

        # skill_view's collector: one unique candidate, no backup shadowing it.
        assert [
            md
            for _, md in _collect_skill_candidates("pytorch-fsdp", None, [skills_dir])
        ] == [active_md]
        # The shared index walker behind skills_list / prompt building agrees.
        assert list(iter_skill_index_files(skills_dir, "SKILL.md")) == [active_md]

    def test_second_restore_does_not_relocate_earlier_backup(self, tmp_path):
        """The restore scan itself must skip ``.restore-backups``: repairing the same
        skill twice must leave the first backup where it was instead of moving it
        (name-matching would otherwise "repair" the backup into a nested backup)."""
        optional_dir = tmp_path / "optional-skills"
        _write_skill(optional_dir / "ml" / "pytorch-fsdp", SKILL_MD_OFFICIAL)
        skills_dir = tmp_path / "user_skills"
        _write_skill(skills_dir / "ml" / "pytorch-fsdp", SKILL_MD_LOCAL)

        with self._patches(optional_dir, skills_dir):
            first = restore_official_optional_skill("pytorch-fsdp", restore=True)
            _write_skill(
                skills_dir / "ml" / "pytorch-fsdp", SKILL_MD_LOCAL
            )  # modify again
            second = restore_official_optional_skill("pytorch-fsdp", restore=True)

        assert second["ok"] is True and second["backed_up"] == ["ml/pytorch-fsdp"]
        first_backup_md = Path(first["backup_dir"]) / "ml" / "pytorch-fsdp" / "SKILL.md"
        assert first_backup_md.exists()  # the earlier backup was never re-parented away
        assert SKILL_MD_LOCAL in first_backup_md.read_text(encoding="utf-8")
