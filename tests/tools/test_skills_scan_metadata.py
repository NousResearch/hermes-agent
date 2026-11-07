"""Category discovery stays profile-scoped without repeated per-skill configuration reads."""

import pytest

from agent import skill_utils
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import skills_tool


def _write_skill(root, category, name):
    directory = root / category / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill\n---\n# {name}\n", encoding="utf-8"
    )


def test_catalog_category_roots_are_bounded_and_refresh_between_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})
    monkeypatch.setattr(skill_utils, "get_project_skills_dirs", list)
    original = skill_utils.get_external_skills_dirs
    calls = []

    def external_dirs():
        calls.append(True)
        return original()

    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", external_dirs)
    for profile in ("first", "second"):
        home, external = tmp_path / profile, tmp_path / f"external-{profile}"
        _write_skill(home / "skills", f"local-{profile}", f"local-one-{profile}")
        _write_skill(home / "skills", f"local-{profile}", f"local-two-{profile}")
        _write_skill(external, f"external-{profile}", f"external-one-{profile}")
        (home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external.as_posix()}\n", encoding="utf-8"
        )
        token = set_hermes_home_override(home)
        calls.clear()
        try:
            found = skills_tool.find_all_skills()
        finally:
            reset_hermes_home_override(token)
        assert {skill["name"]: skill["category"] for skill in found} == {
            f"local-one-{profile}": f"local-{profile}",
            f"local-two-{profile}": f"local-{profile}",
            f"external-one-{profile}": f"external-{profile}",
        }
        # One discovery lookup plus at most one category snapshot, regardless
        # of the number of skills. A later profile must obtain its own roots.
        assert len(calls) <= 2


@pytest.mark.parametrize("mode, expected", [
    ("overlapping", "category"), ("external-unavailable", "category"), ("empty", None),
])
def test_category_root_snapshot_preserves_precedence_and_empty_roots(tmp_path, monkeypatch, mode, expected):
    local = tmp_path / "skills"
    skill = local / "category" / "nested" / "skill" / "SKILL.md"
    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: local)

    def external_dirs():
        if mode == "external-unavailable":
            raise OSError("external skill roots unavailable")
        return [local / "category"]

    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", external_dirs)
    roots = [] if mode == "empty" else skills_tool._category_search_dirs()
    assert skills_tool._get_category_from_path(skill, roots=roots) == expected
    if mode != "empty":
        assert skills_tool._get_category_from_path(skill) == expected
