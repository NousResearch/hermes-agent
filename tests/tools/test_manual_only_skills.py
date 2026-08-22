"""Manual-invocation gate: the `disable-model-invocation` frontmatter flag.

A skill carrying the flag stays registered and user-invocable as `/name`, but
must not be offered to the model. Three surfaces offer a skill -- the
system-prompt index, the skills_list tool, and the model's skill_view entry
point -- and missing any one leaves a hole, so all three are covered here,
along with the two things easiest to break while tidying the guards up: that
a manual-only skill stays visible to the USER-facing listing, and that
explicit loads still work.
"""

import json
from unittest.mock import patch

import tools.skills_tool as skills_tool_module
from tools.skills_tool import _find_all_skills

FLAG = "disable-model-invocation: true\n"


def _make_skill(skills_dir, name, frontmatter_extra=""):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n"
        f"{frontmatter_extra}---\n\n# {name}\n\nBody.\n"
    )
    return skill_dir


def _names(**kw):
    skills_tool_module._SKILLS_CACHE.clear()
    return {s["name"] for s in _find_all_skills(**kw)}


class TestFindAllSkillsManualOnly:
    def test_hidden_from_model_but_visible_to_user_surfaces(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "auto-skill")
            _make_skill(tmp_path, "manual-skill", frontmatter_extra=FLAG)

            model_facing = _names(hide_manual_only=True)
            assert "manual-skill" not in model_facing
            assert "auto-skill" in model_facing

            # The banner and GET /v1/skills use the default; a manual-only
            # skill must stay visible there — /name is how the user runs it.
            user_facing = _names()
            assert {"auto-skill", "manual-skill"} <= user_facing

            # The config UI sees everything, as before.
            assert "manual-skill" in _names(skip_disabled=True)


class TestSkillViewManualOnly:
    def test_model_entry_point_refuses_flagged_skill(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "manual-skill", frontmatter_extra=FLAG)
            skills_tool_module._SKILLS_CACHE.clear()

            result = json.loads(
                skills_tool_module._skill_view_with_bump({"name": "manual-skill"})
            )
            assert result["success"] is False
            assert "manual-invocation-only" in result["error"]

    def test_unflagged_skill_still_loads(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "auto-skill")
            skills_tool_module._SKILLS_CACHE.clear()

            result = json.loads(
                skills_tool_module._skill_view_with_bump({"name": "auto-skill"})
            )
            assert result["success"] is True

    def test_explicit_load_path_bypasses_the_gate(self, tmp_path):
        """`/name` and --skills reach _load_skill_payload, not the model gate."""
        from agent.skill_commands import _load_skill_payload

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "manual-skill", frontmatter_extra=FLAG)
            skills_tool_module._SKILLS_CACHE.clear()

            assert _load_skill_payload("manual-skill") is not None


class TestSystemPromptIndexManualOnly:
    def test_flagged_skill_is_excluded_from_the_prompt_index(self, tmp_path):
        from agent.prompt_builder import _parse_skill_file

        auto = _make_skill(tmp_path, "auto-skill") / "SKILL.md"
        manual = _make_skill(tmp_path, "manual-skill", frontmatter_extra=FLAG) / "SKILL.md"

        assert _parse_skill_file(auto)[0] is True
        assert _parse_skill_file(manual)[0] is False
