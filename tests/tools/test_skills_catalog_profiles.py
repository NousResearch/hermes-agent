"""Regression tests for profile- and context-scoped skill catalog results."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def isolated_catalog(monkeypatch, tmp_path):
    from agent import skill_utils
    from hermes_cli import plugins
    from tools import skills_tool

    task_home = tmp_path / "task-hermes-home"
    task_home.mkdir()
    os_home = tmp_path / "os-home"
    os_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(task_home))

    monkeypatch.setattr(skill_utils, "get_project_skills_dirs", list)
    monkeypatch.setattr(skill_utils, "get_skill_create_dir", lambda: None)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", list)
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda: set())
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(list_plugin_skill_metadata=list),
    )
    skills_tool.clear_skills_cache()
    try:
        yield skills_tool, skill_utils, tmp_path
    finally:
        skills_tool.clear_skills_cache()


def _write_skill(profile_home, relative_path, name, description, *, environments=None):
    skill_dir = profile_home / "skills" / relative_path
    skill_dir.mkdir(parents=True, exist_ok=True)
    metadata = [f"name: {name}", f"description: {description}"]
    if environments:
        metadata.append(f"environments: [{', '.join(environments)}]")
    frontmatter = "\n".join(metadata)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\nInstructions.\n", encoding="utf-8"
    )
    return skill_dir


def _dispatch_skills_list(home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.registry import registry

    token = set_hermes_home_override(home)
    try:
        result = registry.dispatch("skills_list", {})
    finally:
        reset_hermes_home_override(token)
    assert isinstance(result, str)
    payload = json.loads(result)
    assert payload["success"] is True
    return payload


def _listed_skills(payload):
    return {item["name"]: item["description"] for item in payload["skills"]}


def _find_all_for(skills_tool, home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    try:
        return skills_tool.find_all_skills()
    finally:
        reset_hermes_home_override(token)


def test_catalog_results_are_cached_per_profile_and_clear_reloads_each(
    isolated_catalog, monkeypatch
):
    """Alternating profile lookups reuse each parse without mixing roots or caller mutations."""
    skills_tool, skill_utils, tmp_path = isolated_catalog
    from hermes_constants import get_hermes_home

    profile_a = tmp_path / "profiles" / "a"
    profile_b = tmp_path / "profiles" / "b"
    _write_skill(profile_a, "mlops/training/a-one", "a-one", "profile A one")
    _write_skill(profile_a, "mlops/training/a-two", "a-two", "profile A two")
    _write_skill(profile_b, "research/b-one", "b-one", "profile B one")
    _write_skill(profile_b, "research/b-two", "b-two", "profile B two")

    parse_counts = {profile_a.resolve(): 0, profile_b.resolve(): 0}
    real_parse = skill_utils.parse_frontmatter

    def count_real_parse(text):
        parse_counts[get_hermes_home().resolve()] += 1
        return real_parse(text)

    monkeypatch.setattr(skill_utils, "parse_frontmatter", count_real_parse)

    results = [
        _dispatch_skills_list(profile)
        for profile in (profile_a, profile_b, profile_a, profile_b)
    ]
    assert _listed_skills(results[0]) == {
        "a-one": "profile A one",
        "a-two": "profile A two",
    }
    assert _listed_skills(results[1]) == {
        "b-one": "profile B one",
        "b-two": "profile B two",
    }
    assert _listed_skills(results[2]) == _listed_skills(results[0])
    assert _listed_skills(results[3]) == _listed_skills(results[1])
    assert parse_counts == {profile_a.resolve(): 2, profile_b.resolve(): 2}

    direct_results = _find_all_for(skills_tool, profile_a)
    direct_results[0]["name"] = "caller mutation"
    direct_results.append({"name": "injected"})
    assert {item["name"] for item in _find_all_for(skills_tool, profile_a)} == {
        "a-one",
        "a-two",
    }
    assert parse_counts == {profile_a.resolve(): 2, profile_b.resolve(): 2}

    skills_tool.clear_skills_cache()
    assert _listed_skills(_dispatch_skills_list(profile_a)) == {
        "a-one": "profile A one",
        "a-two": "profile A two",
    }
    assert _listed_skills(_dispatch_skills_list(profile_b)) == {
        "b-one": "profile B one",
        "b-two": "profile B two",
    }
    assert parse_counts == {profile_a.resolve(): 4, profile_b.resolve(): 4}

    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE_MAX_ENTRIES", 2)
    profile_c = tmp_path / "profiles" / "c"
    _write_skill(profile_c, "research/c-one", "c-one", "profile C")
    parse_counts[profile_c.resolve()] = 0
    assert _listed_skills(_dispatch_skills_list(profile_c)) == {"c-one": "profile C"}
    assert len(skills_tool._SKILLS_CACHE) <= 2
    assert _listed_skills(_dispatch_skills_list(profile_a)) == _listed_skills(results[0])
    assert len(skills_tool._SKILLS_CACHE) <= 2
    assert _listed_skills(_dispatch_skills_list(profile_b)) == _listed_skills(results[1])
    assert len(skills_tool._SKILLS_CACHE) <= 2


def test_catalog_cache_tracks_context_dependent_kanban_visibility(
    isolated_catalog, monkeypatch
):
    """The same home and directory tree can have different offer visibility per execution."""
    _skills_tool, skill_utils, tmp_path = isolated_catalog
    profile = tmp_path / "profiles" / "shared"
    _write_skill(
        profile,
        "workflow/kanban-helper",
        "kanban-helper",
        "Only relevant inside a Kanban worker",
        environments=["kanban"],
    )
    context = {"kanban": False}
    monkeypatch.setitem(skill_utils._ENV_DETECTORS, "kanban", lambda: context["kanban"])

    assert _listed_skills(_dispatch_skills_list(profile)) == {}
    context["kanban"] = True
    assert _listed_skills(_dispatch_skills_list(profile)) == {
        "kanban-helper": "Only relevant inside a Kanban worker"
    }
    context["kanban"] = False
    assert _listed_skills(_dispatch_skills_list(profile)) == {}
