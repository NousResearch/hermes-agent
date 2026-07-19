"""Focused skill loading and duplicate suppression contracts."""

import json
from unittest.mock import patch

import tools.skills_tool as skills_tool


def _make_sectioned_skill(root):
    skill_dir = root / "provider-ops"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: provider-ops\n"
        "description: Provider operations.\n"
        "---\n\n"
        "# Provider Operations\n\n"
        "Intro.\n\n"
        "## OAuth Setup\n\n"
        "Configure OAuth.\n\n"
        "```bash\n"
        "# This shell comment is not a Markdown heading.\n"
        "hermes auth list\n"
        "```\n\n"
        "### Verification\n\n"
        "Run the provider check.\n\n"
        "## API Keys\n\n"
        "Configure an API key.\n",
        encoding="utf-8",
    )
    return skill_md


def test_skill_view_returns_one_complete_markdown_section(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_sectioned_skill(tmp_path)
        result = json.loads(skills_tool.skill_view("provider-ops", section="OAuth Setup"))

    assert result["success"] is True
    assert result["section"] == "OAuth Setup"
    assert "Configure OAuth" in result["content"]
    assert "### Verification" in result["content"]
    assert "API Keys" not in result["content"]
    assert result["content_hash"]


def test_skill_view_reports_available_sections_for_unknown_section(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_sectioned_skill(tmp_path)
        result = json.loads(skills_tool.skill_view("provider-ops", section="Missing"))

    assert result["success"] is False
    assert "not found" in result["error"].lower()
    assert "OAuth Setup" in result["available_sections"]
    assert "API Keys" in result["available_sections"]
    assert "This shell comment is not a Markdown heading." not in result["available_sections"]


def test_registry_wrapper_suppresses_unchanged_duplicate_within_session(tmp_path):
    skills_tool._clear_skill_view_load_cache()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        skill_md = _make_sectioned_skill(tmp_path)
        first = json.loads(
            skills_tool._skill_view_with_bump({"name": "provider-ops"}, session_id="session-a")
        )
        second = json.loads(
            skills_tool._skill_view_with_bump({"name": "provider-ops"}, session_id="session-a")
        )
        other_session = json.loads(
            skills_tool._skill_view_with_bump({"name": "provider-ops"}, session_id="session-b")
        )

        refs = skill_md.parent / "references"
        refs.mkdir()
        (refs / "new.md").write_text("# New workflow\n\nInstructions.\n", encoding="utf-8")
        manifest_changed = json.loads(
            skills_tool._skill_view_with_bump({"name": "provider-ops"}, session_id="session-a")
        )

        skill_md.write_text(skill_md.read_text(encoding="utf-8") + "\nChanged.\n", encoding="utf-8")
        changed = json.loads(
            skills_tool._skill_view_with_bump({"name": "provider-ops"}, session_id="session-a")
        )

    assert first["success"] is True and "content" in first
    assert second["success"] is True
    assert second["already_loaded"] is True
    assert "content" not in second
    assert second["content_hash"] == first["content_hash"]
    assert "content" in other_session
    assert "content" in manifest_changed
    assert manifest_changed["linked_files"]["references"] == ["references/new.md"]
    assert manifest_changed["content_hash"] != first["content_hash"]
    assert "content" in changed
    assert changed["content_hash"] != first["content_hash"]


def test_force_reload_bypasses_duplicate_suppression(tmp_path):
    skills_tool._clear_skill_view_load_cache()
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_sectioned_skill(tmp_path)
        json.loads(skills_tool._skill_view_with_bump({"name": "provider-ops"}, session_id="s"))
        forced = json.loads(
            skills_tool._skill_view_with_bump(
                {"name": "provider-ops", "force_reload": True}, session_id="s"
            )
        )
        suppressed_after_force = json.loads(
            skills_tool._skill_view_with_bump({"name": "provider-ops"}, session_id="s")
        )

    assert forced["success"] is True
    assert "content" in forced
    assert forced.get("already_loaded") is not True
    assert suppressed_after_force["already_loaded"] is True
