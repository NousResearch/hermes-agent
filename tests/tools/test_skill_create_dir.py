"""Tests for ``skills.create_dir`` — config-driven skill creation directory.

When configured, agent-created skills (skill_manage action=create) land in
``skills.create_dir`` instead of the profile-local skills dir, the directory
is scanned for discovery like the local dir, and the instruction text that
names the creation path renders the configured directory.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with an empty local skills dir."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_hermes_home_cache", None, raising=False)

    from agent import skill_utils as su
    su._external_dirs_cache_clear()

    import tools.skills_tool as skills_tool
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(smt, "SKILLS_DIR", home / "skills")
    yield home
    su._external_dirs_cache_clear()


def _write_config(home: Path, body: str):
    (home / "config.yaml").write_text(body, encoding="utf-8")
    from agent import skill_utils as su
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


def _skill_md(name: str) -> str:
    return (
        f"---\nname: {name}\n"
        f"description: Use when testing create dir routing. One-line behavior.\n"
        f"---\n\n# {name}\n\nBody.\n"
    )


class TestGetSkillCreateDir:
    def test_unset_returns_none(self, isolated_home):
        from agent.skill_utils import get_skill_create_dir
        _write_config(isolated_home, "skills:\n  external_dirs: []\n")
        assert get_skill_create_dir() is None

    def test_absolute_path(self, isolated_home, tmp_path):
        from agent.skill_utils import get_skill_create_dir
        brain = tmp_path / "brain-skills"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        assert get_skill_create_dir() == brain.resolve()

    def test_relative_path_resolves_against_home(self, isolated_home):
        from agent.skill_utils import get_skill_create_dir
        _write_config(isolated_home, "skills:\n  create_dir: brain\n")
        assert get_skill_create_dir() == (isolated_home / "brain").resolve()

    def test_tilde_expansion(self, isolated_home):
        from agent.skill_utils import get_skill_create_dir
        _write_config(isolated_home, "skills:\n  create_dir: ~/brain-skills\n")
        assert get_skill_create_dir() == (Path.home() / "brain-skills").resolve()

    def test_local_skills_dir_treated_as_unset(self, isolated_home):
        from agent.skill_utils import get_skill_create_dir
        _write_config(
            isolated_home, f"skills:\n  create_dir: {isolated_home / 'skills'}\n"
        )
        assert get_skill_create_dir() is None

    def test_empty_string_treated_as_unset(self, isolated_home):
        from agent.skill_utils import get_skill_create_dir
        _write_config(isolated_home, "skills:\n  create_dir: ''\n")
        assert get_skill_create_dir() is None


class TestDisplaySkillCreateDir:
    def test_default_renders_local_skills_path(self, isolated_home):
        from agent.skill_utils import display_skill_create_dir
        _write_config(isolated_home, "skills: {}\n")
        assert display_skill_create_dir().endswith("/skills/")

    def test_configured_renders_configured_path(self, isolated_home, tmp_path):
        from agent.skill_utils import display_skill_create_dir
        brain = tmp_path / "opt-brain"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        assert "opt-brain" in display_skill_create_dir()

    def test_schema_helper_follows_config(self, isolated_home, tmp_path):
        from tools.skill_manager_tool import _display_create_dir
        brain = tmp_path / "opt-brain"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        assert "opt-brain" in _display_create_dir()


class TestDiscovery:
    def test_create_dir_in_all_skills_dirs(self, isolated_home, tmp_path):
        from agent.skill_utils import get_all_skills_dirs
        brain = tmp_path / "brain-skills"
        brain.mkdir()
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        dirs = [d.resolve() for d in get_all_skills_dirs()]
        assert dirs[0] == (isolated_home / "skills").resolve()
        assert brain.resolve() in dirs

    def test_missing_create_dir_not_scanned(self, isolated_home, tmp_path):
        from agent.skill_utils import get_all_skills_dirs
        brain = tmp_path / "does-not-exist"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        assert brain.resolve() not in [d.resolve() for d in get_all_skills_dirs()]

    def test_no_duplicate_when_also_in_external_dirs(self, isolated_home, tmp_path):
        from agent.skill_utils import get_all_skills_dirs
        brain = tmp_path / "brain-skills"
        brain.mkdir()
        _write_config(
            isolated_home,
            f"skills:\n  create_dir: {brain}\n  external_dirs:\n    - {brain}\n",
        )
        dirs = [d.resolve() for d in get_all_skills_dirs()]
        assert dirs.count(brain.resolve()) == 1


class TestCreateRouting:
    def test_create_lands_in_create_dir(self, isolated_home, tmp_path):
        from tools.skill_manager_tool import skill_manage
        brain = tmp_path / "brain-skills"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        res = json.loads(skill_manage("", "", operations=[{
            "action": "create", "name": "routed-skill",
            "content": _skill_md("routed-skill"),
        }]))
        assert res.get("success"), res
        assert (brain / "routed-skill" / "SKILL.md").exists()
        assert not (isolated_home / "skills" / "routed-skill").exists()
        # Out-of-root creation reports an absolute path, not a relative_to
        # crash (single-op legacy shape surfaces the path field).
        res_flat = json.loads(skill_manage(
            "create", "routed-skill-flat", content=_skill_md("routed-skill-flat"),
        ))
        assert res_flat.get("success"), res_flat
        assert str(brain / "routed-skill-flat") == res_flat["path"]

    def test_create_with_category(self, isolated_home, tmp_path):
        from tools.skill_manager_tool import skill_manage
        brain = tmp_path / "brain-skills"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        res = json.loads(skill_manage("", "", operations=[{
            "action": "create", "name": "cat-skill", "category": "devops",
            "content": _skill_md("cat-skill"),
        }]))
        assert res.get("success"), res
        assert (brain / "devops" / "cat-skill" / "SKILL.md").exists()

    def test_default_create_still_local(self, isolated_home):
        from tools.skill_manager_tool import skill_manage
        _write_config(isolated_home, "skills: {}\n")
        res = json.loads(skill_manage("", "", operations=[{
            "action": "create", "name": "local-skill",
            "content": _skill_md("local-skill"),
        }]))
        assert res.get("success"), res
        assert (isolated_home / "skills" / "local-skill" / "SKILL.md").exists()

    def test_created_skill_is_findable_and_patchable(self, isolated_home, tmp_path):
        from tools.skill_manager_tool import skill_manage, _find_skill
        brain = tmp_path / "brain-skills"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        json.loads(skill_manage("", "", operations=[{
            "action": "create", "name": "patchable-skill",
            "content": _skill_md("patchable-skill"),
        }]))
        found = _find_skill("patchable-skill")
        assert found is not None
        res = json.loads(skill_manage("", "", operations=[{
            "action": "patch", "name": "patchable-skill",
            "old_string": "Body.", "new_string": "Patched.",
        }]))
        assert res.get("success"), res
        assert "Patched." in (brain / "patchable-skill" / "SKILL.md").read_text()
