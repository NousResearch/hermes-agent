"""A session resumed with fewer toolsets than it was pinned with.

``restore_agent_tool_prefix`` keeps a pinned tool's bytes so tools[] (the head of every request)
stays cached. Once the restore drops a pinned tool that prefix has moved anyway, so a survivor whose
description is derived from the other tool names (``model_tools._DYNAMIC_SCHEMA_REWRITERS``) must
not keep naming the dropped ones. A restore that drops nothing stays byte-identical to the pin.
"""

import copy
import json
import types

import model_tools
from tools import mcp_tool_agent

PINNED = ["terminal", "file", "code_execution"]


def _agent(toolsets):
    agent = types.SimpleNamespace(enabled_toolsets=list(toolsets), disabled_toolsets=None, session_id=None)
    agent.tools = model_tools.get_tool_definitions(enabled_toolsets=list(toolsets), quiet_mode=True)
    agent.valid_tool_names = {t["function"]["name"] for t in agent.tools}
    return agent


def _pin(toolsets):
    return {"version": mcp_tool_agent.tool_pin_version(), "tools": copy.deepcopy(_agent(toolsets).tools)}


def _by_name(tools):
    return {t["function"]["name"]: t for t in tools}


def test_a_narrowed_resume_stops_naming_the_tools_it_dropped():
    pin = _pin(PINNED)
    resumed = _agent(["terminal", "code_execution"])  # the file toolset is gone

    mcp_tool_agent.restore_agent_tool_prefix(resumed, pin)

    names = {t["function"]["name"] for t in resumed.tools}
    dropped = {t["function"]["name"] for t in pin["tools"]} - names
    assert {"read_file", "write_file", "patch", "search_files"} <= dropped
    described = _by_name(resumed.tools)["execute_code"]["function"]["description"]
    assert not [name for name in dropped if name in described]
    assert _by_name(resumed.tools)["execute_code"] == _by_name(_agent(["terminal", "code_execution"]).tools)["execute_code"]


def test_a_carried_survivor_is_rewritten_against_the_final_tool_names():
    # execute_code's check_fn fails in the resuming process (a sandbox went away), so it is carried from the pin
    # rather than built here; the file toolset is gone too. Its pinned helper list must not survive either.
    pin = _pin(PINNED)
    resumed = _agent(["terminal", "code_execution"])
    resumed.tools = [t for t in resumed.tools if t["function"]["name"] != "execute_code"]

    mcp_tool_agent.restore_agent_tool_prefix(resumed, pin)

    restored = _by_name(resumed.tools)
    assert "execute_code" in restored
    named = [n for n in ("read_file", "write_file", "patch", "search_files")
             if n in restored["execute_code"]["function"]["description"]]
    assert not named


def test_a_resume_with_the_same_toolsets_sends_the_pinned_bytes():
    pin = _pin(PINNED)
    for tool in pin["tools"]:  # bytes this process would not derive: the pin must still win
        tool["function"]["description"] += " (as first sent)"
    resumed = _agent(PINNED)

    mcp_tool_agent.restore_agent_tool_prefix(resumed, pin)

    assert json.dumps(resumed.tools) == json.dumps(pin["tools"])
