"""skill_view discloses the schemas of a skill's required tools when the tool_search bridge
defers them (port of langchain-ai/deepagents#6552 ``include_tools``)."""

import json

import pytest

from tools.registry import registry
from tools.skills_tool import _skill_view_with_bump, reset_skill_view_dedup

_SKILL = (
    "---\nname: ticket-desk\ndescription: File tickets.\nmetadata:\n  hermes:\n"
    "    requires_tools: [plugin_create_ticket]\n---\n# Ticket desk\n\nCall plugin_create_ticket.\n"
)


@pytest.fixture
def ticket_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skill_dir = home / "skills" / "ticket-desk"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    # A plugin tool (non-core toolset) is deferrable under the default config, and its
    # presence activates the bridge for the session.
    registry.register(
        name="plugin_create_ticket", toolset="ticketing",
        schema={"name": "plugin_create_ticket", "description": "Create a ticket.",
                "parameters": {"type": "object", "properties": {"title": {"type": "string"}},
                               "required": ["title"]}},
        handler=lambda args, **kw: "{}")
    try:
        yield home
    finally:
        registry.deregister("plugin_create_ticket")


def _view(**kw):
    return json.loads(_skill_view_with_bump({"name": "ticket-desk"}, task_id="t-disc", **kw))


def test_skill_view_discloses_required_tool_deferred_behind_the_bridge(ticket_home):
    result = _view()
    assert result["success"] is True
    assert set(result["deferred_tools"]) == {"plugin_create_ticket"}
    assert result["deferred_tools"]["plugin_create_ticket"]["parameters"]["required"] == ["title"]
    assert "tool_call" in result["deferred_tools_note"]


def test_disclosure_stays_inside_the_session_scope(ticket_home):
    # A session whose grant excludes the plugin's toolset must not be shown a schema its
    # own tool_call would refuse; a skill-only grant also has no bridge to speak of.
    result = _view(enabled_toolsets=["skills"])
    assert result["success"] is True
    assert "deferred_tools" not in result
