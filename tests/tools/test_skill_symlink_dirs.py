"""``skill_manage`` must resolve a skill the way ``skill_view``/``skills_list`` do — including a
skill directory that is a symlink (a profile whose ``skills/<name>`` points into a checkout).

Discovery walks skill roots with ``iter_skill_index_files`` (``os.walk(followlinks=True)``), so a
symlinked skill is listed and loadable. ``skill_manage`` scanned with ``Path.rglob``, which does not
descend through directory symlinks, so the same skill was "not found in active profile" for
patch/edit — and ``create(name, category)`` then nested a duplicate ``SKILL.md`` inside the real
skill directory (#141050).
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with an empty local skills dir and no external roots."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_hermes_home_cache", None, raising=False)

    from agent import skill_utils as su
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()

    from tools import skills_tool
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(smt, "SKILLS_DIR", home / "skills")
    yield home
    su._external_dirs_cache_clear()


def _skill_md(name: str) -> str:
    return (
        f"---\nname: {name}\n"
        f"description: Use when testing a symlinked skill dir.\n"
        f"---\n\n# {name}\n\nBody.\n"
    )


def _symlinked_skill(home: Path, checkout: Path, name: str = "heal") -> tuple[Path, Path]:
    """``<home>/skills/<name>`` as a directory symlink to a real skill outside the skills root."""
    real = checkout / name
    real.mkdir(parents=True)
    (real / "SKILL.md").write_text(_skill_md(name), encoding="utf-8")
    link = home / "skills" / name
    link.symlink_to(real, target_is_directory=True)
    return real, link


def test_symlinked_skill_is_visible_to_skill_manage(isolated_home, tmp_path):
    """The resolver ``skill_manage`` uses must see what discovery sees."""
    import tools.skill_manager_tool as smt

    _real, link = _symlinked_skill(isolated_home, tmp_path / "checkout")

    found = smt._find_skill("heal")
    assert found is not None, "a symlinked skill dir is invisible to skill_manage"
    assert Path(found["path"]) == link


def test_patch_edits_through_a_symlinked_skill_dir(isolated_home, tmp_path):
    """An edit of a symlinked skill writes the real file instead of answering 'not found'."""
    from tools.skill_manager_tool import skill_manage

    real, _link = _symlinked_skill(isolated_home, tmp_path / "checkout")

    res = json.loads(skill_manage("patch", "heal", old_string="Body.", new_string="Patched body."))
    assert res.get("success"), res
    assert "Patched body." in (real / "SKILL.md").read_text(encoding="utf-8")


def test_create_refuses_name_that_already_exists_behind_a_symlink(isolated_home, tmp_path):
    """``create(name, category=name)`` must not nest a second skill inside the existing one."""
    from tools.skill_manager_tool import skill_manage

    real, _link = _symlinked_skill(isolated_home, tmp_path / "checkout")

    res = json.loads(skill_manage("create", "heal", category="heal", content=_skill_md("heal")))
    assert not res.get("success"), res
    assert not (real / "heal").exists(), "a duplicate skill was created inside the symlinked skill"


def test_create_refuses_target_inside_an_existing_skill_dir(isolated_home, tmp_path):
    """Any new skill dir under an existing skill directory would scan as a nested duplicate."""
    from tools.skill_manager_tool import skill_manage

    real, _link = _symlinked_skill(isolated_home, tmp_path / "checkout")

    res = json.loads(skill_manage("create", "nested", category="heal", content=_skill_md("nested")))
    assert not res.get("success"), res
    assert not (real / "nested").exists()
