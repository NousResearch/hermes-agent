"""Regression for #137639: a ``tool_call`` naming a directly-listed tool is dispatched as
the local call instead of refused.

The bridge exists to reach tools a session does NOT list directly; a name it *does* list
carries unambiguous intent, so the dispatcher runs it rather than spending an iteration on
a refusal the model then has to correct by hand. These tests pin both halves: a direct name
routes through to the real tool, and a name outside the session's grant keeps its refusal —
the bridge must not widen into a registry-wide call door.
"""

from __future__ import annotations

import json

from model_tools import _dispatch_bridge_tool, handle_function_call


def _bridge_call(name: str, arguments: dict) -> dict:
    return {"calls": [{"name": name, "arguments": arguments}]}


def test_bridge_unwraps_a_directly_listed_tool_for_local_dispatch():
    result, underlying = _dispatch_bridge_tool(
        "tool_call", _bridge_call("read_file", {"path": "/tmp/direct-listing-probe.txt"}),
        ["file"], None,
    )
    assert result is None
    assert underlying == ("read_file", {"path": "/tmp/direct-listing-probe.txt"})


def test_handle_function_call_runs_the_directly_listed_tool(tmp_path):
    target = tmp_path / "direct.txt"
    target.write_text("BRIDGE-DIRECT-DISPATCH", encoding="utf-8")
    out = json.loads(handle_function_call(
        "tool_call", _bridge_call("read_file", {"path": str(target)}),
        enabled_toolsets=["file"],
    ))
    assert "BRIDGE-DIRECT-DISPATCH" in json.dumps(out)
    assert "directly-listed" not in json.dumps(out)


def test_direct_dispatch_does_not_widen_the_session_grant():
    # ``terminal`` is a real registered tool, but this session grants only ``file``: the
    # bridge keeps refusing instead of reaching outside the session's tool defs.
    result, underlying = _dispatch_bridge_tool(
        "tool_call", _bridge_call("terminal", {"command": "echo nope"}), ["file"], None,
    )
    assert underlying is None
    assert "terminal" in json.loads(result)["error"]
