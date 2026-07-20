import json
import sys
import threading
from types import SimpleNamespace
from types import ModuleType

import pytest

from tools import delegate_tool


def _parent(**overrides):
    base = {
        "base_url": "https://parent.invalid/v1",
        "api_key": "parent-key",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "model": "parent-model",
        "enabled_toolsets": ["terminal", "file", "web", "read_only_file", "mcp-notes"],
        "valid_tool_names": ["read_file", "write_file", "patch", "search_files", "web_search"],
        "_delegate_depth": 0,
        "_active_children": [],
        "_active_children_lock": threading.Lock(),
        "_fallback_chain": [{"provider": "fallback", "model": "fallback-model"}],
        "providers_allowed": None,
        "providers_ignored": None,
        "providers_order": None,
        "provider_sort": None,
        "provider_require_parameters": False,
        "provider_data_collection": "",
        "openrouter_min_coding_score": None,
        "max_tokens": None,
        "request_overrides": {},
        "prefill_messages": None,
        "_session_db": None,
        "session_id": "parent-session",
        "_current_turn_id": "turn-1",
        "tool_progress_callback": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeAgent:
    captures = []

    def __init__(self, **kwargs):
        FakeAgent.captures.append(kwargs)
        self.__dict__.update(kwargs)
        self.session_id = "child-session"
        self._session_init_model_config = {}
        self.valid_tool_names = []


@pytest.fixture(autouse=True)
def _clean_fake_agent():
    FakeAgent.captures.clear()
    yield
    FakeAgent.captures.clear()


def test_read_only_file_toolset_contains_only_read_tools():
    read_only = delegate_tool.TOOLSETS["read_only_file"]

    assert sorted(read_only["tools"]) == ["read_file", "search_files"]
    assert "write_file" not in read_only["tools"]
    assert "patch" not in read_only["tools"]


def test_delegate_schema_exposes_lane_but_no_raw_model_override_fields():
    props = delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    task_props = props["tasks"]["items"]["properties"]

    assert "lane" in props
    assert "lane" in task_props
    for forbidden in ("model", "provider", "base_url", "api_key"):
        assert forbidden not in props
        assert forbidden not in task_props


def test_unknown_lane_fails_with_metadata_safe_error(monkeypatch):
    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {
            "lanes": {
                "source": {
                    "provider": "secret-provider",
                    "model": "secret-model",
                    "base_url": "https://secret.invalid/v1",
                }
            }
        },
    )

    with pytest.raises(ValueError) as exc:
        delegate_tool._resolve_delegation_lane("missing")

    msg = str(exc)
    assert "Unknown delegation lane 'missing'" in msg
    assert "source" in msg
    assert "secret-provider" not in msg
    assert "secret-model" not in msg
    assert "secret.invalid" not in msg


def test_lane_resolution_rejects_credential_fields(monkeypatch):
    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {"lanes": {"fusion": {"model": "aegis-model", "api_key": "nope"}}},
    )

    with pytest.raises(ValueError) as exc:
        delegate_tool._resolve_delegation_lane("fusion")

    assert "credential-shaped fields" in str(exc.value)
    assert "nope" not in str(exc.value)


def test_lane_resolution_normalizes_allowed_fields_and_defaults(monkeypatch):
    raw = {
        "provider": "custom:aegis",
        "model": "aegis-model",
        "toolsets": ["read_only_file"],
        "max_iterations": 12,
        "reasoning_effort": "high",
    }
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"lanes": {"fusion": raw}})

    first = delegate_tool._resolve_delegation_lane("fusion")
    second = delegate_tool._resolve_delegation_lane("fusion")

    assert first == {
        "name": "fusion",
        "provider": "custom:aegis",
        "model": "aegis-model",
        "toolsets": ["read_only_file"],
        "max_iterations": 12,
        "reasoning_effort": "high",
        "inherit_parent_mcp": False,
        "inherit_fallback": False,
    }
    assert first is not second


def test_lane_resolution_rejects_endpoint_fields(monkeypatch):
    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {
            "lanes": {
                "fusion": {
                    "provider": "custom:aegis",
                    "base_url": "https://secret-endpoint.invalid/v1",
                }
            }
        },
    )

    with pytest.raises(ValueError) as exc:
        delegate_tool._resolve_delegation_lane("fusion")

    assert "unsupported field(s): base_url" in str(exc.value)
    assert "secret-endpoint" not in str(exc.value)


def test_lane_named_custom_provider_uses_runtime_credential_resolution(monkeypatch):
    from hermes_cli import runtime_provider

    calls = []

    def fake_resolve_runtime_provider(*, requested, target_model):
        calls.append((requested, target_model))
        return {
            "provider": "custom",
            "model": target_model,
            "base_url": "https://aegis.invalid/v1",
            "api_key": "resolved-key",
            "api_mode": "chat_completions",
            "request_overrides": {},
            "max_output_tokens": None,
        }

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)
    lane = {
        "name": "fusion",
        "provider": "custom:aegis",
        "model": "aegis-model",
    }

    resolved = delegate_tool._resolve_delegation_credentials({}, _parent(), lane)

    assert calls == [("custom:aegis", "aegis-model")]
    assert resolved["provider"] == "custom:aegis"
    assert resolved["model"] == "aegis-model"
    assert resolved["base_url"] == "https://aegis.invalid/v1"
    assert resolved["api_key"] == "resolved-key"
    assert resolved["api_mode"] == "chat_completions"


def test_lane_child_cannot_widen_toolsets_or_inherit_mcp_or_fallback(monkeypatch):
    fake_run_agent = ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    lane = {
        "name": "fusion",
        "model": "lane-model",
        "provider": None,
        "base_url": None,
        "api_mode": None,
        "toolsets": ["read_only_file"],
        "inherit_parent_mcp": False,
        "inherit_fallback": False,
    }

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="inspect only",
        context=None,
        toolsets=["read_only_file", "terminal", "mcp-notes"],
        model="lane-model",
        max_iterations=9,
        task_count=1,
        parent_agent=_parent(),
        lane=lane,
    )

    captured = FakeAgent.captures[-1]
    assert captured["enabled_toolsets"] == ["read_only_file"]
    assert captured["fallback_model"] is None
    assert child._delegate_lane == "fusion"
    assert child._delegate_requested_model == "lane-model"
    assert child._delegate_fallback_used is False


def test_lane_result_metadata_reports_post_run_fallback_model():
    child = SimpleNamespace(
        _delegate_lane="fusion",
        _delegate_requested_provider="custom:aegis",
        _delegate_requested_model="requested-model",
        _delegate_actual_provider="custom:aegis",
        _delegate_actual_model="requested-model",
        _delegate_fallback_used=False,
        _fallback_index=1,
        _fallback_activated=True,
        provider="openrouter",
        model="fallback-model",
    )

    metadata = delegate_tool._lane_result_metadata(child)

    assert metadata == {
        "lane": "fusion",
        "requested_provider": "custom:aegis",
        "requested_model": "requested-model",
        "actual_provider": "openrouter",
        "actual_model": "fallback-model",
        "fallback_used": True,
    }


def test_lane_toolset_ceiling_downgrades_orchestrator_without_delegation(monkeypatch):
    fake_run_agent = ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 2)
    lane = {
        "name": "fusion",
        "model": "lane-model",
        "provider": None,
        "base_url": None,
        "api_mode": None,
        "toolsets": ["read_only_file"],
        "inherit_parent_mcp": False,
        "inherit_fallback": False,
    }

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="inspect only",
        context=None,
        toolsets=None,
        model="lane-model",
        max_iterations=9,
        task_count=1,
        parent_agent=_parent(),
        lane=lane,
        role="orchestrator",
    )

    captured = FakeAgent.captures[-1]
    assert captured["enabled_toolsets"] == ["read_only_file"]
    assert child._delegate_role == "leaf"
    assert "Subagent Spawning" not in captured["ephemeral_system_prompt"]


def test_no_lane_keeps_parent_mcp_and_fallback_inheritance(monkeypatch):
    fake_run_agent = ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"inherit_mcp_toolsets": True})

    delegate_tool._build_child_agent(
        task_index=0,
        goal="legacy",
        context=None,
        toolsets=["read_only_file"],
        model=None,
        max_iterations=9,
        task_count=1,
        parent_agent=_parent(),
    )

    captured = FakeAgent.captures[-1]
    assert captured["enabled_toolsets"] == ["read_only_file", "mcp-notes"]
    assert captured["fallback_model"] == [{"provider": "fallback", "model": "fallback-model"}]


def test_delegate_task_resolves_top_level_and_per_task_lanes(monkeypatch):
    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {
            "max_concurrent_children": 3,
            "lanes": {
                "architect": {"model": "architect-model", "toolsets": ["read_only_file"]},
                "builder": {"model": "builder-model", "toolsets": []},
            },
        },
    )
    captured = []

    def fake_build(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(_delegate_role="leaf", _delegate_lane=kwargs["lane"]["name"])

    monkeypatch.setattr(delegate_tool, "_build_child_agent", fake_build)
    monkeypatch.setattr(
        delegate_tool,
        "_run_single_child",
        lambda task_index, goal, child, parent_agent: {
            "task_index": task_index,
            "status": "completed",
            "summary": goal,
            "api_calls": 0,
            "duration_seconds": 0,
            "lane": child._delegate_lane,
        },
    )

    result = json.loads(
        delegate_tool.delegate_task(
            tasks=[{"goal": "a"}, {"goal": "b", "lane": "builder"}],
            lane="architect",
            background=False,
            parent_agent=_parent(),
        )
    )

    assert [call["lane"]["name"] for call in captured] == ["architect", "builder"]
    assert [call["model"] for call in captured] == ["architect-model", "builder-model"]
    assert [r["lane"] for r in result["results"]] == ["architect", "builder"]
