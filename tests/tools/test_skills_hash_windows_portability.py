"""Skill bundle hashes use the same portable path order on every host."""

from tools.skills_hub import OptionalSkillSource, bundle_content_hash
from tools.skills_guard import content_hash


def _write_demo_skill(root):
    skill_dir = root / "demo-skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("same content", encoding="utf-8")
    (skill_dir / "references" / "checklist.md").write_text(
        "- [ ] security\n", encoding="utf-8"
    )
    return skill_dir


def test_optional_bundle_uses_posix_keys_and_matches_installed_hash(tmp_path):
    skill_dir = _write_demo_skill(tmp_path)
    source = OptionalSkillSource()
    source._optional_dir = tmp_path

    bundle = source.fetch("official/demo-skill")

    assert bundle is not None
    assert set(bundle.files) == {"SKILL.md", "references/checklist.md"}
    assert bundle_content_hash(bundle) == content_hash(skill_dir)
