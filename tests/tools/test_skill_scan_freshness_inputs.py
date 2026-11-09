"""Every input the scan serves from must be fingerprinted or invalidated.

Review coverage for the catalog-perf series: the freshness signature must see the legacy flat
``<name>.md`` files a lookup can serve, exclusion names runtime state BELOW a configured root
(never an unrelated ancestor above it), and offer-time visibility is decided at read time —
the tree signature proves the files unchanged, never the execution context.
"""
import json
from unittest.mock import patch

import agent.skill_utils as su
import tools.skills_tool as st


def _skill_md(parent, name, frontmatter_extra=""):
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    extra = f"\n{frontmatter_extra}" if frontmatter_extra else ""
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill.{extra}\n---\n\nBody.\n", encoding="utf-8")
    return d


def test_a_flat_markdown_added_to_a_walked_dir_is_found(tmp_path):
    """A legacy flat skill written into an unchanged directory must appear on the next
    skill_view — the signature fingerprint has to cover the flat ``<name>.md`` filenames, or a
    warmed locate keeps answering "not found" forever."""
    loose = tmp_path / "loose"
    loose.mkdir()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        st.clear_skills_cache()
        assert not json.loads(st.skill_view("new-review-flat")).get("success")
        (loose / "new-review-flat.md").write_text(
            "---\nname: new-review-flat\ndescription: flat skill.\n---\n\nBody.\n", encoding="utf-8")
        result = json.loads(st.skill_view("new-review-flat"))
    assert result.get("success"), "a flat <name>.md added to a walked dir must be located"
    assert result.get("name") == "new-review-flat"


def test_a_root_under_a_cache_named_ancestor_still_scans(tmp_path):
    """``cache`` exclusion names runtime state inside a root; a home laid out as
    ``<fixture>/cache/profile/skills`` is a configured root and must still scan (parity
    with main: judging absolute path parts drops the entire tree)."""
    skills = tmp_path / "cache" / "profile" / "skills"
    _skill_md(skills, "review-anchor")
    with patch("tools.skills_tool.SKILLS_DIR", skills):
        st.clear_skills_cache()
        names = [s["name"] for s in st.find_all_skills()]
    assert "review-anchor" in names, "an excluded-named ANCESTOR of the root dropped the root"


def test_offer_gates_are_decided_at_read_time(tmp_path, monkeypatch):
    """The catalog persists rows, not verdicts: a kanban-gated skill must appear in the listing
    when the environment turns on, with no file change between the two reads (per-lookup
    visibility selection, reconciling with #146644)."""
    _skill_md(tmp_path, "gated", frontmatter_extra="environments: [kanban]")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        st.clear_skills_cache()
        monkeypatch.setitem(su._ENV_DETECTORS, "kanban", lambda: False)
        assert "gated" not in [s["name"] for s in st.find_all_skills()]
        monkeypatch.setitem(su._ENV_DETECTORS, "kanban", lambda: True)
        names = [s["name"] for s in st.find_all_skills()]
    assert "gated" in names, "scan-time eligibility was persisted: the flipped context never applied"


def test_kanban_gate_runs_the_real_detector(tmp_path):
    """The kanban gate must survive its own call-out: an explicitly selected kanban toolset
    surfaces a kanban-gated skill through the real catalog (stubbing ``_ENV_DETECTORS`` hides
    a TypeError in ``_detect_kanban``, whose silent ``except`` would hide every kanban skill)."""
    _skill_md(tmp_path, "kanban-play", frontmatter_extra="environments: [kanban]")
    from tools.kanban_toolset_context import scoped_kanban_toolset_selection

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), scoped_kanban_toolset_selection(["kanban"]):
        st.clear_skills_cache()
        names = [s["name"] for s in st.find_all_skills()]
    assert "kanban-play" in names, "an explicitly selected kanban toolset hid its own skills"
