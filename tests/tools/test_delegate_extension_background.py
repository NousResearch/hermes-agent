"""Extensions must receive the host's model-delegation delivery decision.

The documented model surface always backgrounds top-level delegations and joins
subagent delegations. Native discovery, both model producers, and real async
completion delivery expose a raw-request forwarding regression that synchronous
fallback fixtures conceal. Provider execution is replaced by a pure result.
"""

import json
from pathlib import Path
from queue import Empty

import pytest


PLUGIN = '''
import json
from agent.subagent_extension import SubagentExtension

class Extension(SubagentExtension):
    def handle_action(self, action, request):
        if action != "resume":
            return None
        result = {"results": [{"status": "completed", "session_id": request["session_id"],
                              "summary": request["consumer_argument"]}]}
        if not request.get("background", False):
            return json.dumps(result)
        from tools.async_delegation import dispatch_async_delegation_batch
        return json.dumps(dispatch_async_delegation_batch(
            goals=[request["message"]], context=None, toolsets=None, role="leaf", model=None,
            session_key="review-test", parent_session_id="owner", runner=lambda: result))

def register(ctx):
    ctx.register_subagent_extension(Extension())
'''


@pytest.fixture
def extension(tmp_path, monkeypatch):
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from tools import async_delegation
    from tools.process_registry import process_registry

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    folder = tmp_path / "plugins" / "delivery-consumer"
    folder.mkdir(parents=True)
    (folder / "plugin.yaml").write_text("name: delivery-consumer\nversion: 0.1.0\n")
    (folder / "__init__.py").write_text(PLUGIN)
    (tmp_path / "config.yaml").write_text(json.dumps({"plugins": {
        "enabled": ["delivery-consumer"],
        "entries": {"delivery-consumer": {"allow_tool_override": True}},
    }}))
    async_delegation._reset_for_tests()
    discover_plugins(force=True)
    assert get_plugin_manager()._subagent_extensions
    yield process_registry.completion_queue
    # Join actual workers before resetting/draining their completion destination.
    if async_delegation._executor is not None:
        async_delegation._executor.shutdown(wait=True)
    async_delegation._reset_for_tests()
    get_plugin_manager().unload("delivery-consumer")
    while True:
        try:
            process_registry.completion_queue.get_nowait()
        except Empty:
            break


@pytest.mark.parametrize("surface", ["inline", "registry"])
@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("hidden_background", [None, False, True])
def test_model_delivery_decision_reaches_extension(extension, surface, depth, hidden_background):
    from run_agent import AIAgent
    from tools.registry import registry

    parent = object.__new__(AIAgent)
    parent._delegate_depth = depth
    args = {"action": "resume", "session_id": "retained-child", "message": "Follow up",
            "consumer_argument": "Keep this plugin-owned argument"}
    if hidden_background is not None:
        args["background"] = hidden_background
    original = dict(args)
    raw = (parent._dispatch_delegate_task(args) if surface == "inline" else
           registry.get_entry("delegate_task").handler(args, parent_agent=parent))
    result = json.loads(raw)
    expected = {"status": "completed", "session_id": "retained-child",
                "summary": "Keep this plugin-owned argument"}
    if depth == 0:
        assert result["status"] == "dispatched", result
        event = extension.get(timeout=10)
        assert event["delegation_id"] == result["delegation_id"]
        assert event["parent_session_id"] == "owner"
        assert event["results"] == [expected]
    else:
        assert result["results"] == [expected]
        assert extension.empty(), "joined child work must not reenter the parent as async completion"
    assert args == original, "host delivery metadata must not mutate model-supplied arguments"
