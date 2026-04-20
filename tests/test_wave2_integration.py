import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.intent_preclassifier import preclassify_intent
from run_agent import AIAgent
from tools.delegate_tool import delegate_task


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


class _FakeClient(MagicMock):
    pass


class _FakeChild(SimpleNamespace):
    pass


DEFAULT_CFG = {
    "provider": "",
    "model": "",
    "base_url": "",
    "api_key": "",
    "runtime_mode": "default",
    "route_category": "unspecified_low",
    "default_route_category": "unspecified_low",
    "default_delegation_profile": "general",
    "default_category": "general",
    "default_skills": [],
    "task_contract": None,
    "permission_preset": "inherit",
    "fallback_policy": "legacy_default_mapping",
    "max_iterations": 50,
    "max_concurrent_children": 3,
}


def _sample_task_contract(required_tools=None) -> dict:
    return {
        "task": "Implement the delegated change",
        "expected_outcome": "A passing implementation with verification evidence",
        "required_skills": ["python", "testing"],
        "required_tools": required_tools or ["terminal", "read_file", "patch"],
        "must_do": ["inspect repo patterns before editing"],
        "must_not_do": {"forbidden_files": ["run_agent.py"]},
        "context": {"repo": "/root/.hermes/hermes-agent"},
    }


def _make_agent(*tool_names: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://example.test/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = _FakeClient()
    return agent


def _make_mock_parent(runtime_activation_state: dict | None = None, depth: int = 0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = ["terminal", "file", "web"]
    parent.valid_tool_names = {"terminal", "read_file", "search_files", "patch", "web_search"}
    parent.runtime_activation_state = runtime_activation_state or {
        "specialist": "builder",
        "archetype": "implementer",
        "route_category": "deep",
        "delegation_profile": "implementation",
        "runtime_mode": "ultrawork",
        "task_contract": None,
        "activation_applied": True,
        "activation_note": "Wave 2 Runtime Activation",
        "wave1_overlay_prompt": "# Wave 1 Prompt Overlays\n\n## Archetype\nname: implementer",
    }
    parent.get_runtime_activation_state.return_value = dict(parent.runtime_activation_state)
    return parent


def _fake_build_child_agent(**kwargs):
    return _FakeChild(
        _delegate_resolution=dict(kwargs.get("delegate_resolution") or {}),
        enabled_toolsets=list(kwargs.get("toolsets") or []),
        enabled_tools=list(kwargs.get("enabled_tools") or []),
        ephemeral_system_prompt=kwargs.get("wave1_overlay_prompt") or "",
        session_id=f"child-{kwargs['task_index']}",
        tool_progress_callback=None,
    )


def _fake_run_single_child(task_index, goal, child=None, parent_agent=None, **_kwargs):
    return {
        "task_index": task_index,
        "status": "completed",
        "summary": goal,
        "api_calls": 1,
        "duration_seconds": 0.01,
        "resolved_inputs": child._delegate_resolution,
        "enabled_toolsets": list(getattr(child, "enabled_toolsets", [])),
        "enabled_tools": list(getattr(child, "enabled_tools", [])),
        "system_prompt": getattr(child, "ephemeral_system_prompt", ""),
    }


def test_top_level_runtime_activation_smoke():
    agent = _make_agent("delegate_task")
    seen_state = {}

    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="Delegating now.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            type="function",
                            function=SimpleNamespace(name="delegate_task", arguments='{"goal": "Implement subtask"}'),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="Delegation finished.", tool_calls=[]),
            )
        ],
        usage=None,
    )

    def fake_delegate_task(**kwargs):
        parent_agent = kwargs["parent_agent"]
        seen_state.update(parent_agent.get_runtime_activation_state())
        return json.dumps({"status": "ok"})

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=[first_response, second_response]),
        patch("tools.delegate_tool.delegate_task", side_effect=fake_delegate_task),
    ):
        result = agent.run_conversation("Use ultrawork mode to implement the patch and run tests.")

    assert result["final_response"] == "Delegation finished."
    assert seen_state["specialist"] == "builder"
    assert seen_state["archetype"] == "implementer"
    assert seen_state["route_category"] == "deep"
    assert seen_state["delegation_profile"] == "implementation"
    assert seen_state["runtime_mode"] == "ultrawork"
    assert seen_state["activation_applied"] is True
    assert "Wave 2 Runtime Activation" in seen_state["activation_note"]


def test_deterministic_specialist_mapping_smoke():
    message = "Review this patch, verify the risky changes, and call out regressions."

    first = preclassify_intent(message)
    second = preclassify_intent(message)

    assert first == second
    assert first.inferred_specialist == "code_reviewer"
    assert first.inferred_archetype == "verifier"
    assert first.inferred_route_category == "quick"
    assert first.inferred_delegation_profile == "verification"
    assert first.inferred_runtime_mode == "default"
    assert first.inferred_specialist != first.inferred_archetype
    assert first.inferred_specialist != first.inferred_route_category
    assert first.inferred_specialist != first.inferred_runtime_mode
    assert first.inferred_delegation_profile != first.inferred_route_category


@patch("tools.delegate_tool._load_config", return_value=DEFAULT_CFG)
@patch("tools.delegate_tool._run_single_child", side_effect=_fake_run_single_child)
@patch("tools.delegate_tool._build_child_agent", side_effect=_fake_build_child_agent)
def test_specialist_changes_child_construction_smoke(_mock_build, _mock_run, _mock_load):
    activation = preclassify_intent("Review this patch, verify the risky changes, and call out regressions.")
    parent = _make_mock_parent(
        {
            "specialist": activation.inferred_specialist,
            "archetype": activation.inferred_archetype,
            "route_category": activation.inferred_route_category,
            "delegation_profile": activation.inferred_delegation_profile,
            "runtime_mode": activation.inferred_runtime_mode,
            "task_contract": None,
            "activation_applied": True,
            "activation_note": activation.activation_reason,
            "wave1_overlay_prompt": "# Wave 1 Prompt Overlays\n\n## Archetype\nname: verifier",
        }
    )

    result = json.loads(delegate_task(goal="Review the implementation", parent_agent=parent))
    payload = result["results"][0]
    resolved = payload["resolved_inputs"]

    assert parent.runtime_activation_state["specialist"] == "code_reviewer"
    assert resolved["archetype"] == "verifier"
    assert resolved["route_category"] == "quick"
    assert resolved["delegation_profile"] == "verification"
    assert resolved["runtime_mode"] == "default"
    assert resolved["route_category"] != resolved["delegation_profile"]
    assert "## Archetype" in payload["system_prompt"]


@patch("tools.delegate_tool._load_config", return_value=DEFAULT_CFG)
@patch("tools.delegate_tool._run_single_child", side_effect=_fake_run_single_child)
@patch("tools.delegate_tool._build_child_agent", side_effect=_fake_build_child_agent)
def test_contract_aware_delegation_smoke(_mock_build, _mock_run, _mock_load):
    parent = _make_mock_parent()
    parent.runtime_activation_state["task_contract"] = _sample_task_contract(["terminal", "read_file", "patch"])
    parent.get_runtime_activation_state.return_value = dict(parent.runtime_activation_state)

    result = json.loads(delegate_task(goal="Implement the fix", parent_agent=parent))
    payload = result["results"][0]

    assert payload["resolved_inputs"]["task_contract"] == parent.runtime_activation_state["task_contract"]
    assert "terminal" in payload["enabled_toolsets"]
    assert "read_file" in payload["enabled_tools"]
    assert "patch" in payload["enabled_tools"]


def test_start_work_explicit_contract_suppresses_conflicting_named_workflow_metadata():
    contract = _sample_task_contract()
    contract["context"]["command_runtime"] = {
        "command_name": "handoff",
        "runtime_mode": "ralph",
    }

    from hermes_cli.command_templates import build_command_invocation

    invocation = build_command_invocation("start-work", raw_args=json.dumps(contract), session_id="sess-wave2", cwd="/tmp")

    assert invocation.task_contract == contract
    assert invocation.named_workflow is None
    assert "NAMED_WORKFLOW_JSON:" not in invocation.prompt_text
    inferred = preclassify_intent({"message": contract["task"], "task_contract": invocation.task_contract})
    assert inferred.inferred_runtime_mode == "ralph"


@patch("tools.delegate_tool._load_config", return_value=DEFAULT_CFG)
@patch("tools.delegate_tool._run_single_child", side_effect=_fake_run_single_child)
@patch("tools.delegate_tool._build_child_agent", side_effect=_fake_build_child_agent)
def test_mixed_batch_smoke(_mock_build, _mock_run, _mock_load):
    parent = _make_mock_parent()

    result = json.loads(
        delegate_task(
            tasks=[
                {
                    "goal": "Research the regression",
                    "archetype": "researcher",
                    "route_category": "quick",
                    "delegation_profile": "research",
                    "runtime_mode": "default",
                },
                {"goal": "Verify the implementation"},
            ],
            parent_agent=parent,
        )
    )

    first = result["results"][0]["resolved_inputs"]
    second = result["results"][1]["resolved_inputs"]

    assert first["archetype"] == "researcher"
    assert first["route_category"] == "quick"
    assert first["delegation_profile"] == "research"
    assert first["runtime_mode"] == "default"

    assert second["archetype"] == "implementer"
    assert second["route_category"] == "deep"
    assert second["delegation_profile"] == "implementation"
    assert second["runtime_mode"] == "ultrawork"
