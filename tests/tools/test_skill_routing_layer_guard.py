"""Tests for the routing-layer size guard (local patch, fryccer — re-port on upgrade).

A SKILL.md at/over ``skills.routing_cap_chars`` (default 20,000) is a routing
document: skill_manage refuses body growth beyond ROUTING_POINTER_ALLOWANCE
(+400 chars — one-line pointer entries) on edit / patch / write_file('SKILL.md')
and steers the new knowledge into references/ instead. Shrinks and sub-cap
writes are never affected; ``create`` keeps its plain MAX_SKILL_CONTENT_CHARS
semantics (it is deliberately NOT wired to this guard).
"""

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import (
    ROUTING_POINTER_ALLOWANCE,
    _routing_layer_size_guard,
    skill_manage,
)


@pytest.fixture(autouse=True)
def isolate_skills(tmp_path, monkeypatch):
    """Redirect SKILLS_DIR to a temp directory."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return skills_dir


@contextmanager
def _routing_cap(cap):
    """Pin ``skills.routing_cap_chars`` by mocking the config READ (the guard itself runs)."""
    with patch("hermes_cli.config.load_config",
               return_value={"skills": {"routing_cap_chars": cap}}):
        yield


def _skill_content(name: str, total_chars: int) -> str:
    """Valid SKILL.md whose TOTAL length is exactly ``total_chars`` (boundaries matter here)."""
    frontmatter = f"---\nname: {name}\ndescription: A test skill\n---\n"
    heading = "# Test Skill\n\n"
    body = heading + "x" * max(0, total_chars - len(frontmatter) - len(heading))
    return frontmatter + body


def _create(name: str, total_chars: int) -> None:
    with _routing_cap(20000):  # create is never guarded; any cap value works
        result = json.loads(skill_manage(
            action="create", name=name, content=_skill_content(name, total_chars)))
    assert result["success"] is True, result


# ---------------------------------------------------------------------------
# Unit level
# ---------------------------------------------------------------------------


class TestRoutingCapConfig:
    def test_cap_reads_skills_routing_cap_chars(self):
        with _routing_cap(1234):
            from tools.skill_manager_tool import _routing_cap_chars
            assert _routing_cap_chars() == 1234

    def test_config_failure_falls_back_to_default(self):
        """If load_config raises, the documented default cap applies (not 'disabled')."""
        from tools.skill_manager_tool import _routing_cap_chars
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert _routing_cap_chars() == 20000

    def test_zero_cap_disables_guard(self):
        with _routing_cap(0):
            err = _routing_layer_size_guard("fat", "x" * 50_000, "y" * 99_000)
        assert err is None


# ---------------------------------------------------------------------------
# 1+2. Disabled cap and small skills
# ---------------------------------------------------------------------------


class TestCapDisabled:
    def test_any_growth_passes_when_cap_is_zero(self):
        _create("fat", 25_000)
        with _routing_cap(0):
            result = json.loads(skill_manage(
                action="edit", name="fat", content=_skill_content("fat", 30_000)))
        assert result["success"] is True, result


class TestSmallSkillUnaffected:
    def test_sub_cap_growth_allowed(self):
        _create("small", 5_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="edit", name="small", content=_skill_content("small", 12_000)))
        assert result["success"] is True, result


# ---------------------------------------------------------------------------
# 3-5. Fat skills: reject growth, allow pointers and shrinks
# ---------------------------------------------------------------------------


class TestFatSkillGuarded:
    def test_patch_growth_over_allowance_rejected(self):
        _create("fat", 25_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="patch", name="fat", old_string="# Test Skill",
                new_string="# Test Skill\n" + "y" * 500))
        assert result["success"] is False
        assert "references/" in result["error"]
        assert str(ROUTING_POINTER_ALLOWANCE) in result["error"]

    def test_edit_growth_rejected(self):
        _create("fat", 25_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="edit", name="fat", content=_skill_content("fat", 26_000)))
        assert result["success"] is False
        assert "references/" in result["error"]

    def test_pointer_entry_within_allowance_allowed(self):
        _create("fat", 25_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="patch", name="fat", old_string="# Test Skill",
                new_string="# Test Skill\n- See references/rules.md for the always-on gates."))
        assert result["success"] is True, result

    def test_shrink_via_edit_allowed(self):
        _create("fat", 25_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="edit", name="fat", content=_skill_content("fat", 24_000)))
        assert result["success"] is True, result

    def test_shrink_via_patch_allowed(self):
        _create("fat", 25_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="patch", name="fat", old_string="# Test Skill\n\n" + "x" * 2_000,
                new_string="# Test Skill"))
        assert result["success"] is True, result


# ---------------------------------------------------------------------------
# 6. Crossing the threshold from below
# ---------------------------------------------------------------------------


class TestCrossThresholdGrowth:
    def test_growth_from_sub_cap_past_cap_plus_allowance_rejected(self):
        _create("teen", 19_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="edit", name="teen", content=_skill_content("teen", 25_000)))
        assert result["success"] is False
        assert "references/" in result["error"]

    def test_growth_to_exactly_cap_plus_allowance_allowed(self):
        # allowed_max = max(cap, len(current)) + allowance == 20_400: an inclusive boundary.
        _create("teen", 19_600)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="edit", name="teen", content=_skill_content("teen", 20_400)))
        assert result["success"] is True, result

    def test_one_char_past_the_boundary_rejected(self):
        _create("teen", 19_600)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="edit", name="teen", content=_skill_content("teen", 20_401)))
        assert result["success"] is False


# ---------------------------------------------------------------------------
# 7. write_file paths
# ---------------------------------------------------------------------------


class TestWriteFilePaths:
    def test_references_write_never_affected(self):
        _create("fat", 25_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="write_file", name="fat", file_path="references/deep-dive.md",
                file_content="# Deep dive\n\n" + "x" * 30_000))
        assert result["success"] is True, result

    def test_write_file_direct_to_fat_skill_md_rejected(self):
        _create("fat", 25_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="write_file", name="fat", file_path="SKILL.md",
                file_content=_skill_content("fat", 26_000)))
        assert result["success"] is False
        assert "references/" in result["error"]

    def test_write_file_to_small_skill_md_still_works(self):
        _create("small", 5_000)
        with _routing_cap(20000):
            result = json.loads(skill_manage(
                action="write_file", name="small", file_path="SKILL.md",
                file_content=_skill_content("small", 6_000)))
        assert result["success"] is True, result


# ---------------------------------------------------------------------------
# Batch (operations) path — the per-op handlers run under the same guard
# ---------------------------------------------------------------------------


class TestBatchPath:
    def test_batch_patch_growth_on_fat_skill_rejected(self):
        _create("fat", 25_000)
        operations = [{"name": "fat", "action": "patch",
                       "old_string": "# Test Skill",
                       "new_string": "# Test Skill\n" + "y" * 500}]
        with _routing_cap(20000):
            result = json.loads(skill_manage(action="patch", name=None, operations=operations))
        assert result["success"] is False
        assert "operations[0]" in result["error"]
        assert "references/" in result["error"]

    def test_batch_pointer_entry_on_fat_skill_allowed(self):
        _create("fat", 25_000)
        operations = [{"name": "fat", "action": "patch",
                       "old_string": "# Test Skill",
                       "new_string": "# Test Skill\n- See references/rules.md."}]
        with _routing_cap(20000):
            result = json.loads(skill_manage(action="patch", name=None, operations=operations))
        assert result["success"] is True, result


# ---------------------------------------------------------------------------
# 8. The review prompt is size-aware
# ---------------------------------------------------------------------------


class TestLessonLayerPrompt:
    def test_lesson_layer_block_names_the_routing_threshold(self):
        from agent.background_review import _LESSON_LAYER_BLOCK
        assert "routing threshold" in _LESSON_LAYER_BLOCK
        assert "skills.routing_cap_chars" in _LESSON_LAYER_BLOCK
