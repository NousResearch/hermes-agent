"""Exact paths and fail-closed ambiguity use real config, dispatch and pending storage."""

import json

import pytest

from tools import skill_manager_tool as smt, skill_usage, write_approval as wa
from tools.registry import registry
from tools.skill_provenance import set_current_write_origin, reset_current_write_origin


BODY = "---\nname: {name}\ndescription: Target resolution.\n---\nOriginal body.\n"


def skill(directory, name="research"):
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "SKILL.md"
    file.write_text(BODY.format(name=name), encoding="utf-8")
    return file


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    configure(home)
    return home


def configure(home, mode="create", enabled=True, extra=""):
    (home / "config.yaml").write_text(
        f"skills:\n  write_approval: {str(enabled).lower()}\n  write_approval_mode: {mode}\n" + extra,
        encoding="utf-8")


def patch(name, old="Original body.", new="Updated body."):
    return dict(action="patch", name=name, old_string=old, new_string=new)


def dispatch(*ops):
    return json.loads(registry.dispatch("skill_manage", {"operations": list(ops)}))


@pytest.mark.parametrize("flat", [False, True])
def test_ambiguous_short_name_returns_exact_paths_without_mutating_or_staging(home, flat):
    first = skill(home / "skills/product/research")
    second = skill(home / "skills/marketing/research")
    before = {p: p.read_bytes() for p in (first, second)}
    op = patch("research")
    result = json.loads(smt.skill_manage(**op)) if flat else dispatch(op)
    assert not result["success"]
    assert result["error_type"] == "ambiguous_skill_target"
    assert result["candidates"] == sorted([str(first.parent), str(second.parent)])
    assert all(p.read_bytes() == data for p, data in before.items())
    assert wa.list_pending(wa.SKILLS) == []
    retry = dispatch(patch(result["candidates"][0]))
    assert retry["success"] and not retry.get("staged")
    assert "Updated body." in (first if str(first.parent) == result["candidates"][0] else second).read_text()


def test_category_path_selects_only_that_skill_and_keeps_usage_key(home):
    first = skill(home / "skills/product/research")
    second = skill(home / "skills/marketing/research")
    result = dispatch(patch("product/research"))
    assert result["success"] and not result.get("staged")
    assert "Updated body." in first.read_text()
    assert "Original body." in second.read_text()
    assert "product/research" not in skill_usage.load_usage()


def test_same_category_across_catalogs_requires_absolute_path(home, tmp_path):
    external = tmp_path / "external"
    first = skill(home / "skills/product/research")
    second = skill(external / "product/research")
    configure(home, extra=f"  external_dirs: [{json.dumps(str(external))}]\n")
    result = dispatch(patch("product/research"))
    assert result["error_type"] == "ambiguous_skill_target"
    assert set(result["candidates"]) == {str(first.parent), str(second.parent)}
    assert dispatch(patch(str(second.parent)))["success"]
    assert "Updated body." in second.read_text() and "Original body." in first.read_text()


@pytest.mark.parametrize("target", ["outside", "traversal", "support", "excluded"])
def test_explicit_path_cannot_escape_catalog_or_target_hidden_support(home, tmp_path, target):
    parent = skill(home / "skills/existing", "existing")
    files = {
        "outside": skill(tmp_path / "outside"),
        "traversal": skill(tmp_path / "escape"),
        "support": skill(parent.parent / "references/sample"),
        "excluded": skill(home / "skills/.git/sample"),
    }
    name = str(files[target].parent)
    if target == "traversal":
        name = str(home / "skills/../../escape")
    before = files[target].read_bytes()
    result = dispatch(patch(name))
    assert not result["success"]
    assert files[target].read_bytes() == before


def test_absolute_target_batch_rollback_restores_exact_directory(home):
    target = skill(home / "skills/product/research")
    before = target.read_bytes()
    result = dispatch(patch(str(target.parent)), patch(str(target.parent), old="not present"))
    assert not result["success"]
    assert target.read_bytes() == before


@pytest.mark.parametrize("mode,enabled", [("all", True), ("all", False), ("create", False)])
def test_legacy_lookup_preserves_first_directory_match(home, mode, enabled):
    first = skill(home / "skills/product/research")
    second = skill(home / "skills/marketing/research")
    configure(home, mode=mode, enabled=enabled)
    before = smt._find_skill("research")
    result = dispatch(patch("research"))
    assert result["success"]
    assert smt._find_skill("research") == before
    if enabled:
        assert result["staged"]
        assert all("Original body." in p.read_text() for p in (first, second))
    else:
        assert "Updated body." in (before["path"] / "SKILL.md").read_text()


def test_display_alias_collision_is_reported_but_directory_still_wins(home):
    first = skill(home / "skills/first", "shared")
    second = skill(home / "skills/second", "shared")
    result = dispatch(patch("shared"))
    assert result["error_type"] == "ambiguous_skill_target"
    assert set(result["candidates"]) == {str(first.parent), str(second.parent)}
    directory = skill(home / "skills/shared", "different")
    assert dispatch(patch("shared"))["success"]
    assert "Updated body." in directory.read_text()
    assert all("Original body." in p.read_text() for p in (first, second))


@pytest.mark.parametrize("essential", [False, True])
def test_exact_path_cannot_bypass_pinned_or_essential_deletion_guard(home, essential):
    name = "hermes-agent" if essential else "research"
    file = skill(home / f"skills/product/{name}", name)
    if not essential:
        skill_usage.set_pinned(name, True)
    result = dispatch(dict(action="delete", name=str(file.parent)))
    assert not result["success"]
    assert "essential" in result["error"] if essential else "pinned" in result["error"]
    assert file.exists()


def test_background_archive_uses_selected_directory_not_duplicate_basename(home):
    first = skill(home / "skills/product/research")
    second = skill(home / "skills/marketing/research")
    umbrella = skill(home / "skills/umbrella", "umbrella")
    skill_usage.record_created("research", agent_created=True)
    token = set_current_write_origin("background_review")
    try:
        result = dispatch(dict(action="delete", name=str(first.parent), absorbed_into=str(umbrella.parent)))
    finally:
        reset_current_write_origin(token)
    assert result["success"], result
    assert not first.exists() and second.exists()
    assert (home / "skills/.archive/research/SKILL.md").exists()


def test_exact_path_preserves_pin_on_different_frontmatter_name(home):
    file = skill(home / "skills/product/research", "product-research")
    skill_usage.set_pinned("product-research", True)
    result = dispatch(dict(action="delete", name=str(file.parent)))
    assert not result["success"] and "pinned" in result["error"]
    assert file.exists()


def test_ordered_frontmatter_collision_is_rejected_before_any_write(home):
    first = skill(home / "skills/first", "before")
    second = skill(home / "skills/second", "shared")
    before = first.read_bytes()
    result = dispatch(dict(action="patch", name="first", content=BODY.format(name="shared")), patch("shared"))
    assert result["error_type"] == "ambiguous_skill_target"
    assert first.read_bytes() == before and "Original body." in second.read_text()
    assert wa.list_pending(wa.SKILLS) == []


@pytest.mark.parametrize("preexisting", [False, True])
def test_failed_creation_replay_owns_exact_target_despite_duplicate_names(home, preexisting):
    first = skill(home / "skills/product/research")
    second = skill(home / "skills/marketing/research")
    target = home / "skills/third/fresh"
    if preexisting:
        target.mkdir(parents=True)
    before = {p: p.read_bytes() for p in (first, second)}
    ops = [dict(action="create", name="fresh", category="third", content=BODY.format(name="fresh")),
           dict(action="write_file", name=str(target), file_path="references/example.md", file_content="Temporary."),
           patch("marketing/research", old="definitely-not-present-XYZ")]
    staged = dispatch(*ops)
    assert staged["staged"]
    record = wa.get_pending(wa.SKILLS, staged["pending_id"])
    assert record is not None
    result = json.loads(smt.apply_skill_pending(record["payload"]))
    assert not result["success"]
    assert result["failed_index"] == 2 and result["completed_before_failure"] == 2
    assert all(p.read_bytes() == data for p, data in before.items())
    assert target.exists() == preexisting
    assert not (target / "SKILL.md").exists()


def test_creation_with_existing_duplicate_names_returns_exact_followup_path(home):
    skill(home / "skills/product/research")
    skill(home / "skills/marketing/research")
    staged = json.loads(smt.skill_manage(action="create", name="fresh", category="third", content=BODY.format(name="fresh")))
    record = wa.get_pending(wa.SKILLS, staged["pending_id"])
    assert record is not None
    result = json.loads(smt.apply_skill_pending(record["payload"]))
    assert result["success"], result
    assert str(home / "skills/third/fresh") in result["hint"]
