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


class TestReadPathDiscovery:
    """The system-prompt skill index is built from ``get_all_skills_dirs()``,
    which folds ``create_dir`` in — so a create_dir skill is advertised to the
    model. Every path that then resolves an advertised name (skill_view,
    skills_list, /slash dispatch, gateway menus) must scan the same dirs, or
    the model is offered a skill it cannot load."""

    @staticmethod
    def _seed(brain: Path, name: str = "brain-skill", category: str = "devops") -> Path:
        skill_dir = brain / category / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(_skill_md(name), encoding="utf-8")
        return skill_dir

    def test_skill_view_finds_create_dir_skill(self, isolated_home, tmp_path):
        from tools.skills_tool import skill_view
        brain = tmp_path / "brain-skills"
        skill_dir = self._seed(brain)
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        res = json.loads(skill_view("brain-skill"))
        assert res.get("success"), res
        assert Path(res["skill_dir"]).resolve() == skill_dir.resolve()

    def test_skill_created_via_skill_manage_is_viewable(self, isolated_home, tmp_path):
        from tools.skill_manager_tool import skill_manage
        from tools.skills_tool import skill_view
        brain = tmp_path / "brain-skills"
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        created = json.loads(skill_manage("", "", operations=[{
            "action": "create", "name": "fresh-skill",
            "content": _skill_md("fresh-skill"),
        }]))
        assert created.get("success"), created
        res = json.loads(skill_view("fresh-skill"))
        assert res.get("success"), res

    def test_skills_list_includes_create_dir_skill_with_category(self, isolated_home, tmp_path):
        from tools.skills_tool import _find_all_skills
        brain = tmp_path / "brain-skills"
        self._seed(brain)
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        by_name = {s["name"]: s for s in _find_all_skills()}
        assert "brain-skill" in by_name
        assert by_name["brain-skill"]["category"] == "devops"

    def test_create_dir_also_listed_in_external_dirs_is_not_ambiguous(self, isolated_home, tmp_path):
        """Listing the same dir under both keys must not surface one skill twice."""
        from tools.skills_tool import skill_view
        brain = tmp_path / "brain-skills"
        self._seed(brain)
        _write_config(
            isolated_home,
            f"skills:\n  create_dir: {brain}\n  external_dirs:\n    - {brain}\n",
        )
        res = json.loads(skill_view("brain-skill"))
        assert res.get("success"), res

    def test_slash_command_registered_for_create_dir_skill(self, isolated_home, tmp_path, monkeypatch):
        import agent.skill_commands as sc_mod
        brain = tmp_path / "brain-skills"
        self._seed(brain)
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        monkeypatch.setattr(sc_mod, "_skill_commands", {})
        monkeypatch.setattr(sc_mod, "_skill_commands_platform", None)
        monkeypatch.setattr(sc_mod, "_skill_commands_home", None)
        assert "/brain-skill" in sc_mod.get_skill_commands()

    def test_gateway_menu_admits_create_dir_skill(self, isolated_home, tmp_path):
        from unittest.mock import patch
        from hermes_cli.commands_platforms import telegram_menu_commands
        brain = (tmp_path / "brain-skills")
        skill_dir = self._seed(brain).resolve()
        _write_config(isolated_home, f"skills:\n  create_dir: {brain}\n")
        fake_cmds = {"/brain-skill": {
            "name": "brain-skill", "description": "create_dir skill",
            "skill_md_path": str(skill_dir / "SKILL.md"), "skill_dir": str(skill_dir),
        }}
        with patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds):
            menu, _ = telegram_menu_commands(max_commands=100)
        assert "brain_skill" in {n for n, _ in menu}
