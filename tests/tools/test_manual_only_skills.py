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

    def test_file_path_cannot_smuggle_out_a_flagged_skill(self, tmp_path):
        """references/ must not be a side door: the flag lives in SKILL.md."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "manual-skill", frontmatter_extra=FLAG)
            (skill_dir / "references").mkdir()
            (skill_dir / "references" / "api.md").write_text("Reference body.\n")
            skills_tool_module._SKILLS_CACHE.clear()

            result = json.loads(
                skills_tool_module._skill_view_with_bump(
                    {"name": "manual-skill", "file_path": "references/api.md"}
                )
            )
            assert result["success"] is False
            assert "manual-invocation-only" in result["error"]

    def test_unflagged_skill_supporting_file_still_loads(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "auto-skill")
            (skill_dir / "notes.md").write_text("Notes.\n")
            skills_tool_module._SKILLS_CACHE.clear()

            result = json.loads(
                skills_tool_module._skill_view_with_bump(
                    {"name": "auto-skill", "file_path": "notes.md"}
                )
            )
            assert result["success"] is True

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


class TestSkillViewManualOnlyPayloadShapes:
    """Payload shapes the gate could not identify, and what it did about them.

    The gate identified the skill by walking back up `_source_path`, so a branch
    that set neither that nor `skill_dir` produced a payload it could not recognise
    — and its default was to allow. `skill_dir` is now set wherever a skill's
    material is served, and an unidentifiable payload that still serves something
    is refused rather than waved through.
    """

    def test_binary_file_does_not_smuggle_out_a_flagged_skill(self, tmp_path):
        """Binary content reaches the model through the ordinary file_path path.

        `skill_view`'s `except UnicodeDecodeError` branch cannot fire — the read
        above it passes `errors="replace"`, so decoding never raises — which means
        a PNG is served as replacement characters by the normal route. That route
        is gated, and this pins it: the interesting property is that binary content
        has no path of its own to slip out by.
        """
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "manual-skill", frontmatter_extra=FLAG)
            (skill_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00binary")
            skills_tool_module._SKILLS_CACHE.clear()

            result = json.loads(
                skills_tool_module._skill_view_with_bump(
                    {"name": "manual-skill", "file_path": "logo.png"}
                )
            )
            assert result["success"] is False
            assert "manual-invocation-only" in result["error"]

    def test_a_miss_does_not_disclose_a_flagged_skills_file_listing(self, tmp_path):
        """A wrong file_path answers with `available_files` — the whole tree."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "manual-skill", frontmatter_extra=FLAG)
            (skill_dir / "references").mkdir()
            (skill_dir / "references" / "secret-api.md").write_text("Body.\n")
            skills_tool_module._SKILLS_CACHE.clear()

            result = json.loads(
                skills_tool_module._skill_view_with_bump(
                    {"name": "manual-skill", "file_path": "no-such-file.md"}
                )
            )
            assert result["success"] is False
            assert "manual-invocation-only" in result["error"]
            assert "available_files" not in result
            assert "secret-api" not in json.dumps(result)

    def test_an_unidentifiable_payload_that_serves_content_is_refused(self):
        """Fail closed: no skill_dir, no usable _source_path, but content served."""
        refusal = skills_tool_module._manual_only_refusal(
            "mystery",
            {"success": True, "name": "mystery", "content": "secret", "file": "x.md"},
            "x.md",
        )
        assert refusal is not None
        assert "manual-invocation-only" in json.loads(refusal)["error"]

    def test_an_unidentifiable_payload_with_nothing_to_serve_is_left_alone(self):
        """A bare error must keep skill_view's own message, not become a refusal."""
        assert (
            skills_tool_module._manual_only_refusal(
                "mystery", {"success": False, "error": "Skill 'mystery' is disabled."}
            )
            is None
        )


class TestManualOnlyFlagCoercion:
    """A quoted `"true"` is what a human writes; YAML hands it over as a string."""

    def test_quoted_and_alternate_spellings_are_honoured(self, tmp_path):
        from agent.prompt_builder import _parse_skill_file

        for i, spelling in enumerate(['"true"', "'true'", "yes", "on", "1", "True"]):
            md = _make_skill(
                tmp_path,
                f"manual-{i}",
                frontmatter_extra=f"disable-model-invocation: {spelling}\n",
            ) / "SKILL.md"
            assert _parse_skill_file(md)[0] is False, spelling

    def test_absent_and_false_spellings_stay_auto_invocable(self, tmp_path):
        from agent.prompt_builder import _parse_skill_file

        for i, spelling in enumerate(["false", '"false"', "no", "off", "0"]):
            md = _make_skill(
                tmp_path,
                f"auto-{i}",
                frontmatter_extra=f"disable-model-invocation: {spelling}\n",
            ) / "SKILL.md"
            assert _parse_skill_file(md)[0] is True, spelling

    def test_the_listing_gate_uses_the_same_coercion(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "auto-skill")
            _make_skill(
                tmp_path,
                "quoted-manual",
                frontmatter_extra='disable-model-invocation: "true"\n',
            )
            model_facing = _names(hide_manual_only=True)
            assert "quoted-manual" not in model_facing
            assert "auto-skill" in model_facing
