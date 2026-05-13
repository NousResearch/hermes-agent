import importlib
import json
from types import SimpleNamespace


def test_backpack_toolsets_are_resolved_and_callable(monkeypatch, tmp_path):
    skill_root = tmp_path / "skill-backpack-tree"
    skill_dir = skill_root / "modules" / "sample-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: sample-skill\ndescription: Sample skill\n---\n\n# Sample\n",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        json.dumps(
            {
                "modules": {
                    "sample-skill": {
                        "path": "modules/sample-skill/SKILL.md",
                        "status": "enabled",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "skills:\n"
        "  skill_backpack_enabled: true\n"
        f"  skill_backpack_root: {skill_root}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import model_tools
    import toolsets

    importlib.reload(toolsets)
    importlib.reload(model_tools)

    assert toolsets.validate_toolset("tool_backpack") is True
    assert toolsets.validate_toolset("skill_backpack") is True

    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=["tool_backpack", "skill_backpack"],
        quiet_mode=True,
    )
    names = {tool["function"]["name"] for tool in definitions}
    assert names == {"tool_backpack", "skill_backpack"}

    tool_result = json.loads(
        model_tools.handle_function_call(
            "tool_backpack",
            {"request": "select search_files"},
        )
    )
    assert tool_result["status"] == "ok"
    assert tool_result["tool"] == "search_files"

    skill_result = json.loads(
        model_tools.handle_function_call(
            "skill_backpack",
            {"request": "select sample-skill"},
        )
    )
    assert skill_result["status"] == "ok"
    assert skill_result["skill"] == "sample-skill"
    assert "# Sample" in skill_result["content"]


def test_tool_backpack_index_only_lists_available_tools(monkeypatch):
    import tools.tool_backpack as tb

    class FakeRegistry:
        def get_all_tool_names(self):
            return ["available_tool", "unavailable_tool", "tool_backpack"]

        def get_definitions(self, tool_names, quiet=False):
            assert quiet is True
            return [
                {"function": {"name": name}}
                for name in tool_names
                if name == "available_tool"
            ]

    monkeypatch.setattr(tb, "registry", FakeRegistry())

    index = tb.build_tool_prompt_index()

    assert "available_tool" in index
    assert "unavailable_tool" not in index
    assert "tool_backpack -" not in index


def test_tool_backpack_empty_request_does_not_advertise_fallback_paths():
    import tools.tool_backpack as tb

    result = json.loads(tb.tool_backpack({"request": ""}))

    assert result["status"] == "blocked"
    assert "candidate hints" not in result["message"].lower()
    assert "index" not in result["message"].lower()


def test_tool_backpack_index_omits_stateful_agent_loop_tools(monkeypatch):
    import tools.tool_backpack as tb

    names = {
        "memory",
        "session_search",
        "todo",
        "clarify",
        "delegate_task",
        "search_files",
        "read_file",
        "tool_backpack",
    }
    monkeypatch.setattr(tb.registry, "get_all_tool_names", lambda: names)
    monkeypatch.setattr(
        tb.registry,
        "get_definitions",
        lambda requested, quiet=True: [
            {"function": {"name": name}} for name in sorted(requested)
        ],
    )

    indexed_names = set(tb._all_index_tool_names())

    assert indexed_names == {"search_files", "read_file"}


def test_skill_backpack_indexes_only_backpack_tree(monkeypatch, tmp_path):
    skill_root = tmp_path / "skill-backpack-tree"
    backpack_skill = skill_root / "modules" / "backpack-skill"
    backpack_skill.mkdir(parents=True)
    (backpack_skill / "SKILL.md").write_text(
        "---\nname: backpack-skill\ndescription: Backpack skill\n---\n\n# Backpack\n",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        json.dumps(
            {
                "modules": {
                    "backpack-skill": {
                        "path": "modules/backpack-skill/SKILL.md",
                        "status": "enabled",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    home = tmp_path / "home"
    home_skill = home / "skills" / "general" / "home-skill"
    home_skill.mkdir(parents=True)
    (home_skill / "SKILL.md").write_text(
        "---\nname: home-skill\ndescription: Home skill\n---\n\n# Home\n",
        encoding="utf-8",
    )
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "skills:\n"
        "  skill_backpack_enabled: true\n"
        f"  skill_backpack_root: {skill_root}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.skill_backpack as sb

    importlib.reload(sb)
    result = json.loads(sb.skill_backpack({"request": "index"}))
    names = {entry[1] for entry in result["skills"]}

    assert names == {"backpack-skill"}

    selected = json.loads(sb.skill_backpack({"request": "select home-skill"}))
    assert selected["status"] == "error"


def test_skill_backpack_schema_tells_models_to_select_known_skills_directly():
    import tools.skill_backpack as sb

    request_description = sb.SKILL_BACKPACK_SCHEMA["parameters"]["properties"]["request"]["description"]

    assert "select <skill-name>" in request_description
    assert "known" in request_description.lower()
    assert "index only" in request_description.lower()
