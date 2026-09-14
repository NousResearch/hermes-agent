"""The merged snapshot rebuild keeps classic Files and canonical Bot Chat gates separate."""

from types import SimpleNamespace

from hermes_constants import get_hermes_home
from tools.bot_mode_dm import message_agent_tool_schema
from tools.mcp_tool_agent import refresh_agent_mcp_tools
from tui_gateway import classic_exports
from tests.tui_gateway.test_classic_export_tool_refresh import build, _assert_share


def managed_session(agent, title):
    home = get_hermes_home()
    profile = home / "profiles" / "writer"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "profile.yaml").write_text("ui_meta:\n  hermes-bots:\n    shape: cloud\n", encoding="utf-8")
    agent._session_db = SimpleNamespace(db_path=str(home / "state.db"), get_session_title=lambda _sid: title)
    agent.session_id = "merge-test-session"
    agent._session_title_hint = None
    agent._bot_mode_protocol = True


def test_classic_refresh_retains_files_without_canonical_dm_authority(build):
    agent, session = build(room_plumbing=True)
    managed_session(agent, "Group Chat: merge")
    classic_exports.install_schema(session)
    agent.tools.append(message_agent_tool_schema())
    agent.valid_tool_names.add("message_agent")
    for preserve_prefix in (False, True):
        refresh_agent_mcp_tools(agent, content_aware=True, preserve_prefix=preserve_prefix)
        _assert_share(agent, 1)
        assert "message_agent" not in agent.valid_tool_names
        assert {tool["function"]["name"] for tool in agent.tools} == agent.valid_tool_names


def test_canonical_dm_refresh_does_not_resurrect_a_classic_export_grant(build):
    agent, session = build()
    managed_session(agent, "Bot Chat")
    classic_exports.install_schema(session)
    agent._classic_export_enabled = False
    schema = classic_exports.tool_schema(agent)
    assert schema is not None
    agent.tools.append({"type": "function", "function": schema})
    agent.valid_tool_names.add("share_group_file")
    for preserve_prefix in (False, True):
        refresh_agent_mcp_tools(agent, content_aware=True, preserve_prefix=preserve_prefix)
        _assert_share(agent, 0)
        assert "message_agent" in agent.valid_tool_names
        assert sum(tool["function"]["name"] == "message_agent" for tool in agent.tools) == 1
        assert agent._classic_export_enabled is False
