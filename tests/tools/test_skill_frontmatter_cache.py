"""skill_view frontmatter memoization: hits are reused, edits and parser patches invalidate."""
import os

import pytest

from tools import skills_tool as st
from tools import skills_tool_plugin as plugin


@pytest.fixture(autouse=True)
def _clear_cache():
    plugin._FRONTMATTER_CACHE.clear()
    yield
    plugin._FRONTMATTER_CACHE.clear()


def _write(path, name, desc="d"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\nbody\n", encoding="utf-8")


def test_repeat_lookup_parses_once(tmp_path, monkeypatch):
    md = tmp_path / "a" / "SKILL.md"
    _write(md, "alpha")
    calls = []
    real = st._parse_frontmatter
    monkeypatch.setattr(st, "_parse_frontmatter", lambda text: (calls.append(1), real(text))[1])
    assert plugin._safe_frontmatter(md)["name"] == "alpha"
    assert plugin._safe_frontmatter(md)["name"] == "alpha"
    assert len(calls) == 1


def test_edit_invalidates(tmp_path):
    md = tmp_path / "a" / "SKILL.md"
    _write(md, "alpha")
    assert plugin._safe_frontmatter(md)["name"] == "alpha"
    _write(md, "renamed-skill")
    st_ = md.stat()
    os.utime(md, ns=(st_.st_atime_ns, st_.st_mtime_ns + 1_000_000))
    assert plugin._safe_frontmatter(md)["name"] == "renamed-skill"


def test_returned_dict_is_isolated_from_cache(tmp_path):
    md = tmp_path / "a" / "SKILL.md"
    _write(md, "alpha")
    plugin._safe_frontmatter(md)["name"] = "mutated"
    assert plugin._safe_frontmatter(md)["name"] == "alpha"


def test_patched_parser_bypasses_stale_entries(tmp_path, monkeypatch):
    md = tmp_path / "a" / "SKILL.md"
    _write(md, "alpha")
    assert plugin._safe_frontmatter(md)["name"] == "alpha"
    monkeypatch.setattr(st, "_parse_frontmatter", lambda text: ({"name": "patched"}, text))
    assert plugin._safe_frontmatter(md)["name"] == "patched"


def test_content_and_missing_paths_are_not_cached(tmp_path):
    assert plugin._safe_frontmatter(content="---\nname: x\ndescription: y\n---\nb\n")["name"] == "x"
    assert plugin._safe_frontmatter(tmp_path / "missing" / "SKILL.md") == {}
    assert plugin._FRONTMATTER_CACHE == {}
