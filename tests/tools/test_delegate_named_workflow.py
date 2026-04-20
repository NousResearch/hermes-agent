from tools.delegate_tool import _build_child_agent, _resolve_wave1_task_inputs


DEFAULT_CFG = {
    "route_category": "unspecified_low",
    "default_route_category": "unspecified_low",
    "default_delegation_profile": "general",
    "default_skills": [],
    "task_contract": None,
    "permission_preset": "inherit",
    "fallback_policy": "legacy_default_mapping",
}

SAMPLE_CONTRACT = {
    "task": "Execute the delegated implementation",
    "expected_outcome": "The delegated child receives the structured contract unchanged.",
    "required_skills": ["python"],
    "required_tools": ["read_file", "patch"],
    "must_do": ["inspect existing patterns first"],
    "must_not_do": ["do not discard the handoff contract"],
    "context": {"ticket": "swarm-d-delegate"},
}

SAMPLE_NAMED_WORKFLOW = {
    "schema": "hermes/named-workflow",
    "schema_version": "1.0",
    "workflow_name": "planner",
    "mode": "plan",
    "objective": "Plan the implementation before execution.",
    "plan": ["inspect", "plan", "handoff"],
    "acceptance": ["machine-readable artifact present"],
    "taxonomy": {
        "named_workflow": "planner",
        "workflow": "planner",
        "specialist": "planner",
        "archetype": "generalist",
        "route_category": "deep",
        "runtime_mode": "execution_supervisor",
        "delegation_profile": "implementation",
    },
    "execution_task_contract": SAMPLE_CONTRACT,
    "consumption": {"downstream_role": "deep_worker", "consumes": "execution_task_contract"},
}


def test_resolve_wave1_task_inputs_preserves_inherited_named_workflow():
    resolved = _resolve_wave1_task_inputs(
        {"goal": "Execute the delegated implementation"},
        cfg=DEFAULT_CFG,
        top_level_archetype="generalist",
        top_level_route_category="deep",
        top_level_delegation_profile="implementation",
        top_level_runtime_mode="execution_supervisor",
        top_level_task_contract=SAMPLE_CONTRACT,
        top_level_named_workflow=SAMPLE_NAMED_WORKFLOW,
    )

    assert resolved["task_contract"] == SAMPLE_CONTRACT
    assert resolved["named_workflow"] == SAMPLE_NAMED_WORKFLOW


def test_resolve_wave1_task_inputs_builds_deep_worker_named_workflow_when_missing():
    resolved = _resolve_wave1_task_inputs(
        {"goal": "Execute the delegated implementation", "task_contract": SAMPLE_CONTRACT},
        cfg=DEFAULT_CFG,
        top_level_archetype="generalist",
        top_level_route_category="deep",
        top_level_delegation_profile="implementation",
        top_level_runtime_mode="execution_supervisor",
    )

    assert resolved["named_workflow"]["workflow_name"] == "deep_worker"
    assert resolved["named_workflow"]["execution_task_contract"] == SAMPLE_CONTRACT
    assert resolved["named_workflow"]["taxonomy"]["route_category"] == "deep"


def test_resolve_wave1_task_inputs_rebuilds_when_inherited_named_workflow_is_invalid():
    invalid_named_workflow = {**SAMPLE_NAMED_WORKFLOW, "workflow_name": "unknown-workflow"}
    resolved = _resolve_wave1_task_inputs(
        {"goal": "Execute the delegated implementation", "task_contract": SAMPLE_CONTRACT},
        cfg=DEFAULT_CFG,
        top_level_archetype="generalist",
        top_level_route_category="deep",
        top_level_delegation_profile="implementation",
        top_level_runtime_mode="execution_supervisor",
        top_level_named_workflow=invalid_named_workflow,
    )

    assert resolved["named_workflow"]["workflow_name"] == "deep_worker"
    assert resolved["named_workflow"]["execution_task_contract"] == SAMPLE_CONTRACT


def test_build_child_agent_includes_named_workflow_artifact_in_prompt(monkeypatch):
    import run_agent

    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
    monkeypatch.setattr("tools.delegate_tool._resolve_workspace_hint", lambda _parent: "/tmp")
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {})

    parent_agent = type(
        "ParentAgent",
        (),
        {
            "enabled_toolsets": ["file"],
            "valid_tool_names": {"read_file", "search_files", "patch", "terminal"},
            "api_key": None,
            "_client_kwargs": {},
            "model": "test-model",
            "provider": None,
            "base_url": "http://example.test",
            "api_mode": None,
            "acp_command": None,
            "acp_args": [],
            "max_tokens": None,
            "reasoning_config": None,
            "prefill_messages": None,
            "platform": "cli",
            "providers_allowed": None,
            "providers_ignored": None,
            "providers_order": None,
            "provider_sort": None,
            "_delegate_depth": 0,
            "session_id": "sess-parent",
            "_session_db": None,
        },
    )()

    _build_child_agent(
        task_index=0,
        goal="Execute the delegated implementation",
        context="Carry out the plan",
        toolsets=["file"],
        max_iterations=5,
        task_count=1,
        parent_agent=parent_agent,
        wave1_overlay_prompt="## Archetype\nname: generalist",
        delegate_resolution={"named_workflow": SAMPLE_NAMED_WORKFLOW},
    )

    prompt = captured["ephemeral_system_prompt"]
    assert "NAMED WORKFLOW ARTIFACT:" in prompt
    assert '"workflow_name": "planner"' in prompt
    assert '"schema": "hermes/named-workflow"' in prompt
