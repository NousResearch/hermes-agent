"""Tests for tools/skills_tool_search.py — the model-facing ``skill_search`` tool."""

import json
from unittest.mock import patch

import tools.skills_tool as skills_tool_module
from tools.skills_tool import skill_search, skill_view
from tools.skills_hub_models import SkillMeta


def _make_skill(skills_dir, name, body="Step 1: Do the thing.", category=None):
    skill_dir = skills_dir / category / name if category else skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8")
    return skill_dir


def _tool_names(**kwargs):
    from model_tools import get_tool_definitions
    return {t["function"]["name"] for t in get_tool_definitions(quiet_mode=True, **kwargs)}


class TestSkillSearch:
    def test_installed_search_returns_compact_rows_without_bodies(self, tmp_path):
        _make_skill(tmp_path, "flutter-ui-development", category="software-development",
                    body="FULL BODY SHOULD NOT LEAK INTO SEARCH RESULTS")
        _make_skill(tmp_path, "spotify", category="media")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            raw = skill_search("flutter", source="installed", limit=10)

        result = json.loads(raw)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"] == [{
            "name": "flutter-ui-development", "identifier": "flutter-ui-development",
            "source": "installed", "description": "Description for flutter-ui-development.",
            "installed": True, "category": "software-development", "tags": [],
        }]
        assert "FULL BODY SHOULD NOT LEAK" not in raw

    def test_installed_identifier_loads_with_skill_view(self, tmp_path):
        _make_skill(tmp_path, "deep-skill", category="foundations/runtime", body="Nested skill body.")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            identifier = json.loads(skill_search("deep", source="installed"))["results"][0]["identifier"]
            view = json.loads(skill_view(identifier))

        assert view["success"] is True
        assert "Nested skill body." in view["content"]

    def test_installed_source_never_calls_the_hub(self, tmp_path):
        _make_skill(tmp_path, "alpha-local")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("tools.skills_hub_search.unified_search") as hub:
            result = json.loads(skill_search("alpha", source="installed"))

        hub.assert_not_called()
        assert [r["name"] for r in result["results"]] == ["alpha-local"]

    def test_empty_query_and_unknown_source_are_rejected(self):
        empty = json.loads(skill_search("   "))
        unknown = json.loads(skill_search("x", source="nowhere"))

        assert empty["success"] is False and "query" in empty["error"]
        assert unknown["success"] is False and "nowhere" in unknown["error"]

    def test_limit_is_capped_and_installed_rows_come_first(self, tmp_path):
        _make_skill(tmp_path, "alpha-local")
        remote = SkillMeta(name="alpha-remote", description="Remote alpha skill.", source="hermes-index",
                           identifier="owner/repo/alpha-remote", trust_level="community", tags=["alpha"])

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("tools.skills_hub_search.create_source_router", return_value=[]), \
             patch("tools.skills_hub_search.unified_search", return_value=[remote]) as hub:
            result = json.loads(skill_search("alpha", source="all", limit=99))

        assert result["limit"] == 50
        assert hub.call_args.kwargs["limit"] == 49
        assert [r["source"] for r in result["results"]] == ["installed", "hermes-index"]
        assert result["results"][1]["installed"] is False
        assert result["results"][1]["trust_level"] == "community"


class TestSkillSearchSchemaResolution:
    """Registration alone is not exposure: the default session resolves ``hermes-cli`` from
    ``_HERMES_CORE_TOOLS``, so the schema must survive that path to reach the model."""

    def test_registered_under_skills_toolset(self):
        entry = skills_tool_module.registry.get_entry("skill_search")
        assert entry is not None and entry.toolset == "skills"
        assert entry.schema["parameters"]["required"] == ["query"]

    def test_hermes_cli_tool_definitions_expose_skill_search(self):
        names = _tool_names(enabled_toolsets=["hermes-cli"])
        assert "skills_list" in names  # positive control: the skill tools did resolve here
        assert "skill_search" in names

    def test_default_session_exposes_skill_search_with_skills_list(self):
        names = _tool_names()
        assert "skills_list" in names
        assert "skill_search" in names
