"""Regression tests for the _find_all_skills discovery cache (#58985 salvage).

Covers the cache-signature fix layered on the cherry-picked contributor
commit: the original keyed the cache on the max mtime of only the TOP-LEVEL
scan dirs, so adding/removing a skill inside a category subdir (which bumps
the category dir's mtime, not the root's) served a stale list indefinitely.
The signature now covers roots + immediate children (mirroring
hermes_cli/profiles.py::_count_skills) plus the disabled-set, with a short
TTL bounding in-place SKILL.md edit staleness.
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


def test_nested_category_skill_add_invalidates(tmp_path):
    """THE bug in the original PR: a new skill inside an existing category
    bumps the category dir's mtime only — the root-mtime key missed it."""
    _write_skill(tmp_path, "cat-a", "skill-one")
    first = st._find_all_skills()
    assert [s["name"] for s in first] == ["skill-one"]

    # Freeze the ROOT dir's mtime so only the category-child signature moves
    # (guards against filesystems bumping the parent too).
    root = tmp_path / "skills"
    root_stat = root.stat()
    _write_skill(tmp_path, "cat-a", "skill-two")
    import os
    os.utime(root, (root_stat.st_atime, root_stat.st_mtime))

    names = sorted(s["name"] for s in st._find_all_skills())
    assert names == ["skill-one", "skill-two"], (
        "category-nested skill add must invalidate the cache"
    )


def test_disabled_set_change_invalidates(tmp_path, monkeypatch):
    """Disabling a skill is a config change with NO filesystem mtime bump —
    it must still invalidate."""
    _write_skill(tmp_path, "cat-a", "skill-one")
    _write_skill(tmp_path, "cat-a", "skill-two")
    names = sorted(s["name"] for s in st._find_all_skills())
    assert names == ["skill-one", "skill-two"]

    monkeypatch.setattr(st, "_get_disabled_skill_names", lambda: {"skill-two"})
    names = sorted(s["name"] for s in st._find_all_skills())
    assert names == ["skill-one"], "disabled-set change must invalidate the cache"


def test_ttl_expiry_forces_rescan(tmp_path, monkeypatch):
    """In-place SKILL.md edits are invisible to any directory signature;
    the TTL bounds that staleness."""
    skill_dir = _write_skill(tmp_path, "cat-a", "skill-one", "old description")
    first = st._find_all_skills()
    assert first[0]["description"] == "old description"

    # Edit the file in place; keep every directory mtime identical.
    import os
    cat = tmp_path / "skills" / "cat-a"
    root = tmp_path / "skills"
    stats = {p: p.stat() for p in (root, cat, skill_dir)}
    (skill_dir / "SKILL.md").write_text(
        "---\nname: skill-one\ndescription: new description\n---\n# skill-one\n",
        encoding="utf-8",
    )
    for p, s in stats.items():
        os.utime(p, (s.st_atime, s.st_mtime))

    # Within TTL: stale (documented trade-off).
    assert st._find_all_skills()[0]["description"] == "old description"

    # Past TTL: fresh.
    monkeypatch.setattr(st, "_SKILLS_CACHE_TTL_SECONDS", 0.0)
    assert st._find_all_skills()[0]["description"] == "new description"


def test_disabled_and_full_views_cached_separately(tmp_path, monkeypatch):
    _write_skill(tmp_path, "cat-a", "skill-one")
    _write_skill(tmp_path, "cat-a", "skill-two")
    monkeypatch.setattr(st, "_get_disabled_skill_names", lambda: {"skill-two"})

    filtered = sorted(s["name"] for s in st._find_all_skills())
    everything = sorted(s["name"] for s in st._find_all_skills(skip_disabled=True))
    assert filtered == ["skill-one"]
    assert everything == ["skill-one", "skill-two"]
