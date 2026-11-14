"""approvals.tools: an ask grant covers only the exact invocation shown, and an unusable policy blocks.

Both tests drive the real ``model_tools.handle_function_call`` with a real config.yaml in the
sandboxed HERMES_HOME and a real registered handler; only the human's answer is a callback.
"""

import json
import os
from pathlib import Path

import pytest

import model_tools
from tools import approval
from tools.registry import registry

_TOOL = "zz_policy_probe"


@pytest.fixture
def probe():
    calls: list[dict] = []
    registry.register(_TOOL, "zz_policy_probe", {"name": _TOOL, "description": "probe", "parameters": {
        "type": "object", "properties": {"to": {"type": "string"}}}},
        handler=lambda args, **_: calls.append(args) or json.dumps({"ok": True}))
    approval.clear_session("default")
    yield calls
    registry.deregister(_TOOL)
    approval.clear_session("default")


def _write_policy(tools_block: str) -> None:
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(f"approvals:\n  tools:\n{tools_block}", encoding="utf-8")


def test_session_grant_is_bound_to_the_shown_arguments(probe, monkeypatch):
    _write_policy(f"    {_TOOL}: ask\n")
    prompts: list[str] = []
    answers = iter(["session", "deny"])

    def human(command, description, **_):
        prompts.append(command)
        return next(answers)

    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(approval, "_yolo_active", lambda: True)  # an explicit ask outranks yolo
    monkeypatch.setattr(approval, "prompt_dangerous_approval", human)

    first = json.loads(model_tools.handle_function_call(_TOOL, {"to": "alice"}))
    again = json.loads(model_tools.handle_function_call(_TOOL, {"to": "alice"}))
    other = json.loads(model_tools.handle_function_call(_TOOL, {"to": "bob"}))

    assert len(prompts) == 2 and "alice" in prompts[0] and "bob" in prompts[1]
    assert first == again == {"ok": True}
    assert "BLOCKED" in other["error"]
    assert [c["to"] for c in probe] == ["alice", "alice"]


@pytest.mark.parametrize("tools_block", [
    "    {}\n",                      # valid empty policy: compatibility control
    f"    {_TOOL}: maybe\n",         # unknown verb
    "    - not-a-mapping\n",         # malformed map
    "reader-fault",                  # the config reader itself raises
])
def test_unusable_policy_blocks_instead_of_allowing(probe, monkeypatch, tools_block):
    if tools_block == "reader-fault":
        _write_policy(f"    {_TOOL}: deny\n")
        def _boom():
            raise OSError("EIO")
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
    else:
        _write_policy(tools_block)
    payload = json.loads(model_tools.handle_function_call(_TOOL, {"to": "alice"}))
    if tools_block == "    {}\n":
        assert payload == {"ok": True} and len(probe) == 1
    else:
        assert "could not be evaluated" in payload["error"] and probe == []
