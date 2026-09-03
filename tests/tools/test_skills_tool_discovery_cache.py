"""Regression tests for the _find_all_skills discovery cache.

The cache signature must cover the same recursive, exclusion-aware SKILL.md
path set as discovery. These tests protect immediate and deep additions,
directory-before-file materialization, followed category symlinks, caller-copy
isolation, and separate disabled/full cache views. In-place content edits remain
bounded by the short TTL.
"""

import time

import pytest

import tools.skills_tool as st


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch, tmp_path):
    """Isolate every test: clear the module cache and point the scan at
    an empty external-dirs list + a tmp skills root."""
    st._SKILLS_CACHE.clear()
    monkeypatch.setattr(st, "_skills_dir", lambda: tmp_path / "skills")
    monkeypatch.setattr(
        "agent.skill_utils.get_external_skills_dirs", lambda: []
    )
    monkeypatch.setattr(st, "_get_disabled_skill_names", lambda: set())
    yield
    st._SKILLS_CACHE.clear()


def _write_skill(root, category, name, description="a skill"):
    d = root / "skills" / category / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n",
        encoding="utf-8",
    )
    return d


def test_cache_hit_serves_copies_not_cache_objects(tmp_path):
    """Callers mutate the returned dicts (web_server annotates
    s['enabled']/s['usage']) — the cache must hand out per-call copies."""
    _write_skill(tmp_path, "cat-a", "skill-one")
    first = st._find_all_skills()
    assert [s["name"] for s in first] == ["skill-one"]

    # Mutate what the first caller got; the next (cached) call must be clean.
    first[0]["enabled"] = False
    first.append({"name": "junk"})

    second = st._find_all_skills()
    assert [s["name"] for s in second] == ["skill-one"]
    assert "enabled" not in second[0], "cache poisoned by caller mutation"
    assert second is not first


def test_nested_skill_add_invalidates_even_when_directory_mtimes_are_frozen(tmp_path):
    """Child-name changes must invalidate independently of filesystem mtimes."""
    import os

    _write_skill(tmp_path, "cat-a", "skill-one")
    assert [s["name"] for s in st._find_all_skills()] == ["skill-one"]

    root = tmp_path / "skills"
    category = root / "cat-a"
    root_stat = root.stat()
    category_stat = category.stat()
    _write_skill(tmp_path, "cat-a", "skill-two")
    os.utime(root, ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns))
    os.utime(category, ns=(category_stat.st_atime_ns, category_stat.st_mtime_ns))

    names = sorted(s["name"] for s in st._find_all_skills())
    assert names == ["skill-one", "skill-two"]


def test_deep_nested_skill_add_invalidates_cached_discovery(tmp_path):
    """Discovery and its signature must cover the same recursive layout."""
    _write_skill(tmp_path, "mlops/training", "skill-one")
    assert [s["name"] for s in st._find_all_skills()] == ["skill-one"]

    _write_skill(tmp_path, "mlops/training", "skill-two")

    names = sorted(s["name"] for s in st._find_all_skills())
    assert names == ["skill-one", "skill-two"]


def test_skill_md_materialization_invalidates_cached_missing_directory(tmp_path):
    """A directory created before its SKILL.md must not poison the cache."""
    _write_skill(tmp_path, "cat-a", "skill-one")
    pending = tmp_path / "skills" / "cat-a" / "skill-two"
    pending.mkdir(parents=True)
    assert [s["name"] for s in st._find_all_skills()] == ["skill-one"]

    (pending / "SKILL.md").write_text(
        "---\nname: skill-two\ndescription: second skill\n---\n# skill-two\n",
        encoding="utf-8",
    )

    names = sorted(s["name"] for s in st._find_all_skills())
    assert names == ["skill-one", "skill-two"]


def test_symlinked_category_add_invalidates_cached_discovery(tmp_path):
    """The signature must follow category symlinks just like discovery does."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    target = tmp_path / "shared-category"
    target.mkdir()
    linked = skills_root / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    def write_linked_skill(name):
        skill_dir = target / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: linked skill\n---\n# {name}\n",
            encoding="utf-8",
        )

    write_linked_skill("skill-one")
    assert [s["name"] for s in st._find_all_skills()] == ["skill-one"]

    write_linked_skill("skill-two")

    names = sorted(s["name"] for s in st._find_all_skills())
    assert names == ["skill-one", "skill-two"]


def test_disabled_and_full_views_cached_separately(tmp_path, monkeypatch):
    _write_skill(tmp_path, "cat-a", "skill-one")
    _write_skill(tmp_path, "cat-a", "skill-two")
    monkeypatch.setattr(st, "_get_disabled_skill_names", lambda: {"skill-two"})

    filtered = sorted(s["name"] for s in st._find_all_skills())
    everything = sorted(s["name"] for s in st._find_all_skills(skip_disabled=True))
    assert filtered == ["skill-one"]
    assert everything == ["skill-one", "skill-two"]
