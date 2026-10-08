"""Named delegation routes select isolated per-dispatch worker runtimes."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.registry import registry


def _route_config():
    return {
        "model": "global-local-model",
        "provider": "custom:global-local",
        "base_url": "http://127.0.0.1:9292/v1",
        "api_key": "global-local-key",
        "api_mode": "chat_completions",
        "reasoning_effort": "low",
        "fallback_providers": [{"provider": "global-fallback"}],
        "command": "global-command",
        "args": ["--global"],
        "max_iterations": 77,
        "routes": {
            "local": {
                "description": "Fast local worker",
                "model": "local-model",
                "provider": "custom:local",
                "base_url": "http://127.0.0.1:9292/v1",
                "api_key": "local-route-key",
                "api_mode": "chat_completions",
                "reasoning_effort": "low",
                "fallback_providers": [],
            },
            "cloud": {
                "description": "Deep worker api_key=sk-abcdefghijklmnopqrstuvwxyz012345",
                "model": "cloud-model",
                "provider": "cloud-provider",
                "reasoning_effort": "high",
                "fallback_providers": [{"provider": "cloud-fallback", "model": "fallback-model"}],
            },
        },
    }


def test_named_route_snapshots_are_atomic_repeatable_and_thread_isolated():
    from tools.delegate_tool_config import _resolve_named_delegation_route

    cfg = _route_config()
    local_a = _resolve_named_delegation_route("local", cfg)
    cloud = _resolve_named_delegation_route("cloud", cfg)
    local_b = _resolve_named_delegation_route("local", cfg)

    assert local_a == local_b and local_a is not local_b
    assert local_a["base_url"] == "http://127.0.0.1:9292/v1"
    assert local_a["api_key"] == "local-route-key"
    assert cloud["provider"] == "cloud-provider"
    assert cloud["model"] == "cloud-model"
    assert cloud["reasoning_effort"] == "high"
    assert cloud["fallback_providers"] == [{"provider": "cloud-fallback", "model": "fallback-model"}]
    assert not ({"base_url", "api_key", "api_mode"} & cloud.keys())
    assert not ({"command", "args", "acp_command", "acp_args"} & cloud.keys())
    assert cloud["max_iterations"] == 77

    cloud["fallback_providers"][0]["model"] = "mutated"
    assert cfg["routes"]["cloud"]["fallback_providers"][0]["model"] == "fallback-model"

    names = ["local", "cloud"] * 16
    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(lambda name: _resolve_named_delegation_route(name, cfg), names))
    assert all(snapshot["model"] == ("local-model" if name == "local" else "cloud-model")
               for name, snapshot in zip(names, snapshots))
    assert len({id(snapshot) for snapshot in snapshots}) == len(snapshots)

    cfg["routes"]["model-less"] = {"provider": "cloud-provider"}
    with pytest.raises(ValueError, match="must configure an exact model"):
        _resolve_named_delegation_route("model-less", cfg)


def test_named_route_reasoning_reaches_child_without_changing_internal_route_policy(monkeypatch):
    import tools.delegate_tool as delegate_tool

    global_cfg = {"reasoning_effort": "low"}
    named_cfg = {"model": "cloud-model", "provider": "cloud-provider", "reasoning_effort": "high"}
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: global_cfg)

    parent = MagicMock()
    parent.base_url = "https://parent.example/v1"
    parent.api_key = "parent-key"
    parent.provider = "parent-provider"
    parent.api_mode = "chat_completions"
    parent.model = "parent-model"
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = []
    parent.disabled_toolsets = []

    with patch("run_agent.AIAgent") as mock_agent:
        mock_agent.return_value = MagicMock()
        delegate_tool._build_child_agent(
            task_index=0, goal="named", context=None, toolsets=None, model="cloud-model",
            max_iterations=10, task_count=1, parent_agent=parent,
            routing_cfg=named_cfg, runtime_cfg=named_cfg,
        )
        named_kwargs = mock_agent.call_args.kwargs

        delegate_tool._build_child_agent(
            task_index=0, goal="internal", context=None, toolsets=None, model="cloud-model",
            max_iterations=10, task_count=1, parent_agent=parent,
            routing_cfg=named_cfg,
        )
        internal_kwargs = mock_agent.call_args.kwargs

    assert named_kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert internal_kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_named_route_schema_errors_and_both_dispatch_paths_are_safe(monkeypatch):
    import tools.delegate_tool as delegate_tool
    from run_agent import AIAgent

    cfg = _route_config()
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: cfg)

    definition = registry.get_definitions({"delegate_task"})[0]
    route_schema = definition["function"]["parameters"]["properties"]["route"]
    serialized = json.dumps(route_schema)
    assert route_schema["enum"] == ["cloud", "local"]
    assert "Fast local worker" in route_schema["description"]
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in serialized
    assert "global-local-key" not in serialized
    assert "local-route-key" not in serialized

    parent = MagicMock(spec=AIAgent, _delegate_depth=0)
    unknown_result = registry.dispatch(
        "delegate_task", {"tasks": [{"goal": "work"}], "route": "missing"}, parent_agent=parent,
    )
    assert isinstance(unknown_result, str)
    unknown = json.loads(unknown_result)
    assert "Unknown delegation route 'missing'" in unknown["error"]
    assert "local" in unknown["error"] and "cloud" in unknown["error"]

    calls = []
    monkeypatch.setattr(delegate_tool, "delegate_task", lambda **kwargs: calls.append(kwargs) or "{}")
    registry.dispatch(
        "delegate_task", {"tasks": [{"goal": "registry"}], "route": "cloud"}, parent_agent=parent,
    )
    AIAgent._dispatch_delegate_task(parent, {"tasks": [{"goal": "agent"}], "route": "local"})
    assert [call["route"] for call in calls] == ["cloud", "local"]
    assert all(call["parent_agent"] is parent for call in calls)


def test_named_routes_do_not_advertise_or_inherit_direct_acp_commands():
    from tools.delegate_tool_config import _configured_delegation_routes, _resolve_named_delegation_route
    cfg = _route_config()
    cfg["routes"]["cloud"].update(command="unresolved-command", args=["--direct"])
    route = _configured_delegation_routes(cfg)["cloud"]
    assert "command" not in route and "args" not in route
    selected = _resolve_named_delegation_route("cloud", cfg)
    assert not ({"command", "args", "acp_command", "acp_args"} & selected.keys())
