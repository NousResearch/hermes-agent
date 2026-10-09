"""Public names for the skills tool's discovery and lookup helpers.

Plugins that list or inspect skills need the same walk, ordering, search roots
and lookup gate the ``skills_list`` / ``skill_view`` tools use. Each public
name below is the SAME object as the private spelling it replaces, so internal
callers and tests that patch the private spelling behave exactly as before.
"""

import pytest

from tools import skill_usage, skills_sync, skills_tool


@pytest.mark.parametrize(
    ("module", "public", "private"),
    [
        (skills_tool, "find_all_skills", "_find_all_skills"),
        (skills_tool, "sort_skills", "_sort_skills"),
        (skills_tool, "skill_search_dirs", "_skill_search_dirs"),
        (skills_tool, "locate_skill", "_locate_skill"),
        (skills_tool, "skill_lookup_path_error", "_skill_lookup_path_error"),
        (skills_sync, "dir_hash", "_dir_hash"),
        (skill_usage, "read_skill_name", "_read_skill_name"),
    ],
)
def test_public_name_is_the_private_helper(module, public, private):
    assert getattr(module, public) is getattr(module, private)


def test_sort_skills_orders_by_category_then_name():
    rows = [{"name": "b", "category": "x"}, {"name": "a", "category": "x"}, {"name": "z"}]
    assert [r["name"] for r in skills_tool.sort_skills(rows)] == ["z", "a", "b"]


def test_skill_lookup_path_error_refuses_traversal_and_accepts_relative_names():
    assert skills_tool.skill_lookup_path_error("../escape")
    assert skills_tool.skill_lookup_path_error("category/name") is None


def test_dir_hash_tracks_content(tmp_path):
    (tmp_path / "SKILL.md").write_text("one", encoding="utf-8")
    first = skills_sync.dir_hash(tmp_path)
    assert skills_sync.dir_hash(tmp_path) == first
    (tmp_path / "SKILL.md").write_text("two", encoding="utf-8")
    assert skills_sync.dir_hash(tmp_path) != first


def test_read_skill_name_prefers_frontmatter_then_fallback(tmp_path):
    named = tmp_path / "named.md"
    named.write_text("---\nname: from-frontmatter\n---\nbody\n", encoding="utf-8")
    bare = tmp_path / "bare.md"
    bare.write_text("no frontmatter\n", encoding="utf-8")
    assert skill_usage.read_skill_name(named, "fallback") == "from-frontmatter"
    assert skill_usage.read_skill_name(bare, "fallback") == "fallback"


# Overriding a public name must reach the module's own callers: they call the
# public spelling, so a plugin that swaps one in changes the tool's behaviour.


def test_overriding_find_all_skills_and_sort_skills_reaches_skills_list(monkeypatch):
    import json

    monkeypatch.setattr(skills_tool, "find_all_skills",
                        lambda **_kw: [{"name": "from-override", "description": "d", "category": None}])
    assert "from-override" in [s["name"] for s in json.loads(skills_tool.skills_list())["skills"]]
    monkeypatch.setattr(skills_tool, "sort_skills", lambda _skills: [{"name": "sorted-by-override"}])
    assert json.loads(skills_tool.skills_list())["skills"] == [{"name": "sorted-by-override"}]


def test_overriding_skill_lookup_path_error_reaches_skill_view(monkeypatch):
    import json

    monkeypatch.setattr(skills_tool, "skill_lookup_path_error", lambda _name: "refused-by-override")
    result = json.loads(skills_tool.skill_view("plain-name"))
    assert result["success"] is False and "refused-by-override" in result["error"]


def test_overriding_skill_search_dirs_and_locate_skill_reaches_skill_view(monkeypatch, tmp_path):
    seen = {}

    def fake_locate(name, local_category_name, roots):
        seen["roots"] = roots
        return '{"success": false, "error": "located-by-override"}', None, None

    monkeypatch.setattr(skills_tool, "skill_search_dirs", lambda: ([(0, tmp_path)], tmp_path))
    monkeypatch.setattr(skills_tool, "locate_skill", fake_locate)
    assert "located-by-override" in skills_tool.skill_view("plain-name")
    assert seen["roots"] == [(0, tmp_path)]


def test_overriding_dir_hash_reaches_origin_hash_check(monkeypatch, tmp_path):
    monkeypatch.setattr(skills_sync, "dir_hash", lambda _d, **_kw: "hash-from-override")
    assert skills_sync._matches_origin_hash(tmp_path, "hash-from-override")


def test_overriding_read_skill_name_reaches_skill_dir_match(monkeypatch, tmp_path):
    skill_md = tmp_path / "dir-name" / "SKILL.md"
    skill_md.parent.mkdir()
    skill_md.write_text("---\nname: declared\n---\n", encoding="utf-8")
    monkeypatch.setattr(skill_usage, "read_skill_name", lambda _md, fallback: "name-from-override")
    assert skill_usage._match_skill_dir([skill_md], "name-from-override") == skill_md.parent
