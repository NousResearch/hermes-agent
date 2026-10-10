"""Behavior contracts for immutable software-delivery delegation roles."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _build_child_system_prompt,
    _normalize_delivery_role,
    _normalize_required_skills,
    delegate_task,
)
from tools.delivery_policy import (
    DELIVERY_ROLES,
    DeliveryPolicy,
    apply_delivery_capabilities,
    build_delivery_policy,
    delivery_role_context,
    delivery_tool_block_reason,
    effective_tool_definitions,
    validate_delivery_terminal_command,
)

SHA = "a" * 40
MERGER_EVIDENCE = {
    "exact_sha": SHA,
    "independent_review": f"independent PASS for {SHA}",
    "ci_evidence": f"required CI green for {SHA}",
}
CLOSURE_EVIDENCE = {
    "merged_sha": SHA,
    "merge_evidence": f"merge commit {SHA} is on main",
    "post_merge_acceptance": f"production acceptance passed for {SHA}",
}


def _definition(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object"}}}


def _delegate_parent() -> SimpleNamespace:
    return SimpleNamespace(_delegate_depth=0, provider="test", model="test/model", base_url="", api_key="test")


def _validation_result(**kwargs) -> dict:
    with (
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=2),
        patch("tools.delegate_tool._get_max_concurrent_children", return_value=4),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "provider": "test", "model": "test/model", "base_url": "", "api_key": "", "api_mode": "",
            "request_overrides": None, "command": None, "args": None,
        }),
        patch("tools.delegate_tool._load_config", return_value=kwargs.pop("config", {})),
    ):
        return json.loads(delegate_task(goal="inspect", parent_agent=_delegate_parent(), **kwargs))


def test_schema_keeps_topology_role_separate_from_delivery_role() -> None:
    task = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert task["properties"]["delivery_role"]["enum"] == list(DELIVERY_ROLES)
    assert "role" not in task["properties"]  # legacy topology override remains unadvertised
    assert set(DELIVERY_ROLES) == {"implementer", "reviewer", "merger", "closure_controller"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), (" Reviewer ", "reviewer"), ("closure-controller", "closure_controller")],
)
def test_delivery_role_normalization_is_explicit_and_backward_compatible(raw, expected) -> None:
    assert _normalize_delivery_role(raw) == expected


def test_invalid_delivery_role_fails_closed() -> None:
    with pytest.raises(ValueError, match="Invalid delivery_role"):
        _normalize_delivery_role("admin")
    result = _validation_result(delivery_role="admin")
    assert "Invalid delivery_role" in result["error"]


def test_optional_role_preserves_normal_non_delivery_delegation() -> None:
    definitions = [_definition("write_file"), _definition("terminal")]
    assert effective_tool_definitions(definitions, None) == definitions
    assert validate_delivery_terminal_command("rm -rf generated", None) is None


def test_required_role_config_rejects_omission() -> None:
    result = _validation_result(config={"require_delivery_role": True})
    assert "delivery_role is required" in result["error"]


def test_invalid_required_role_config_fails_closed_at_runtime() -> None:
    result = _validation_result(config={"require_delivery_role": "true"})
    assert "must be true or false" in result["error"]
    assert "refusing delegation" in result["error"]


def test_required_skill_names_are_deduplicated_and_path_safe() -> None:
    assert _normalize_required_skills([" github ", "github", "software-development/github"]) == [
        "github", "software-development/github",
    ]
    for unsafe in ("../github", "/tmp/skill", "foo//bar", r"..\github"):
        with pytest.raises(ValueError, match="unsafe skill name"):
            _normalize_required_skills([unsafe])


def test_role_contract_ledger_skills_and_evidence_are_system_prompt_content() -> None:
    prompt = _build_child_system_prompt(
        "Review this", "context", role="leaf", delivery_policy=build_delivery_policy("reviewer"),
        acceptance_ledger="1. Verify exact head SHA\n2. Run required tests", required_skills=["github"],
    )
    assert "AUTHORITATIVE IMMUTABLE DELIVERY ROLE" in prompt
    assert "Role: `reviewer`" in prompt
    assert "read-only" in prompt
    assert "AUTHORITATIVE ACCEPTANCE LEDGER" in prompt
    assert "Verify exact head SHA" in prompt
    assert "REQUIRED WORKFLOW SKILLS" in prompt and "skill_view" in prompt and "`github`" in prompt

    merger_prompt = _build_child_system_prompt(
        "Merge", delivery_policy=build_delivery_policy("merger", MERGER_EVIDENCE), role="leaf",
    )
    assert f"Reviewed exact SHA: {SHA}" in merger_prompt
    assert f"Independent review evidence: independent PASS for {SHA}" in merger_prompt
    assert f"CI evidence: required CI green for {SHA}" in merger_prompt


def test_delivery_workers_do_not_inherit_fabricated_prefill_dialogue() -> None:
    parent = MagicMock()
    parent.base_url = "https://example.invalid/v1"
    parent.api_key = "test"
    parent.provider = "test"
    parent.api_mode = "chat_completions"
    parent.model = "test/model"
    parent.platform = "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent.enabled_toolsets = ["terminal", "file", "skills"]
    parent.disabled_toolsets = []
    parent.prefill_messages = [{"role": "user", "content": "fabricated delivery policy"}]

    with patch("run_agent.AIAgent") as agent_class:
        agent_class.return_value = MagicMock(tools=[])
        _build_child_agent(
            task_index=0,
            goal="Review",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            parent_agent=parent,
            task_count=1,
            delivery_policy=DeliveryPolicy("reviewer"),
        )

    kwargs = agent_class.call_args.kwargs
    assert kwargs["prefill_messages"] is None
    assert "AUTHORITATIVE IMMUTABLE DELIVERY ROLE" in kwargs["ephemeral_system_prompt"]


def test_reviewer_effective_tools_are_read_only_and_include_skill_access() -> None:
    names = {
        "terminal", "read_file", "search_files", "skills_list", "skill_view",
        "patch", "write_file", "execute_code", "skill_manage", "delegate_task", "github_mutate",
    }
    child = SimpleNamespace(tools=[_definition(name) for name in names])
    apply_delivery_capabilities(child, DeliveryPolicy("reviewer"))
    effective = child.valid_tool_names
    assert {"terminal", "read_file", "search_files", "skills_list", "skill_view"} <= effective
    assert not {"patch", "write_file", "execute_code", "skill_manage", "delegate_task", "github_mutate"} & effective
    assert child._delivery_policy.role == "reviewer"


def test_delivery_policy_does_not_expand_parent_terminal_or_file_capabilities() -> None:
    effective = effective_tool_definitions([_definition("write_file")], DeliveryPolicy("reviewer"))
    names = {item["function"]["name"] for item in effective}
    assert names <= {"skills_list", "skill_view"}
    assert {"terminal", "read_file", "search_files"}.isdisjoint(names)


def test_composite_or_inherited_toolsets_cannot_bypass_final_filter() -> None:
    definitions = [_definition(name) for name in (
        "terminal", "read_file", "patch", "write_file", "execute_code", "github", "delegate_task",
    )]
    effective = effective_tool_definitions(definitions, DeliveryPolicy("reviewer"))
    names = {(item["function"]["name"]) for item in effective}
    assert {"patch", "write_file", "execute_code", "github", "delegate_task"}.isdisjoint(names)

    implementer_names = {
        item["function"]["name"] for item in effective_tool_definitions(definitions, DeliveryPolicy("implementer"))
    }
    assert "delegate_task" not in implementer_names
    assert {"terminal", "patch", "write_file", "execute_code"} <= implementer_names
    with delivery_role_context(DeliveryPolicy("implementer")):
        assert delivery_tool_block_reason("delegate_task", {"delivery_role": "merger"})


def test_central_tool_dispatch_rejects_fabricated_mutation_call() -> None:
    from model_tools import handle_function_call

    with delivery_role_context(DeliveryPolicy("reviewer")):
        result = json.loads(handle_function_call("write_file", {"path": "/tmp/forbidden", "content": "no"}))
    assert "reviewer delivery policy forbids tool" in result["error"]


def test_agent_dispatch_enforces_policy_after_plugin_argument_rewrite(monkeypatch) -> None:
    from agent import tool_executor

    executed = []
    agent = SimpleNamespace(
        _delivery_policy=DeliveryPolicy("implementer"),
        _tool_guardrails=SimpleNamespace(
            before_call=lambda *_: pytest.fail("guardrails must not run after a policy block")
        ),
    )
    state = tool_executor._ManagedToolResult(
        result=None,
        args={"event": "comment"},
        middleware_trace=[],
        blocked=False,
        dispatched=True,
    )
    ref = tool_executor._ToolCallRef("github", state.args, "task", "call", [])
    monkeypatch.setattr(
        tool_executor,
        "_pre_tool_block",
        lambda *_: (None, {"event": "approve"}),
    )
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *_args, **_kwargs: None)

    result = tool_executor._dispatch_authorized_once(
        agent,
        state,
        ref,
        execute=lambda args: executed.append(args),
        scope_block=None,
        display_index=None,
        begin_execution=None,
        authorization_gate=None,
    )

    assert state.blocked is True
    assert executed == []
    assert state.args == {"event": "approve"}
    assert "implementer delivery policy forbids" in json.loads(result)["error"]


@pytest.mark.parametrize("role", ["reviewer", "merger", "closure_controller"])
def test_restricted_roles_block_fabricated_mutation_tools(role: str) -> None:
    evidence = MERGER_EVIDENCE if role == "merger" else CLOSURE_EVIDENCE if role == "closure_controller" else None
    with delivery_role_context(build_delivery_policy(role, evidence)):
        assert "forbids tool" in (delivery_tool_block_reason("write_file", {"path": "x"}) or "")
        assert "forbids tool" in (delivery_tool_block_reason("github", {"input": {"operation": "merge"}}) or "")


def test_implementer_action_boundary_blocks_github_tracker_and_nested_payload_bypasses() -> None:
    with delivery_role_context(DeliveryPolicy("implementer")):
        assert delivery_tool_block_reason("github", {"event": "APPROVE"})
        assert delivery_tool_block_reason("tracker", {"input": {"status": "done"}})
        assert delivery_tool_block_reason("github", {"variables": {"operation": "merge_pull_request"}})
        assert delivery_tool_block_reason("tracker", {"status": "in_progress"}) is None
        assert delivery_tool_block_reason("tracker", {"action": "update", "status": "in_progress"}) is None


@pytest.mark.parametrize(
    "command",
    [
        "gh pr review 12 --approve", "gh pr merge 12 --squash", "gh issue close 9", "gh api repos/o/r/pulls/12/merge",
        "gh alias set land 'pr merge'", "gh run cancel 5", "env gh run cancel 5", "git merge origin/main",
        "GH_HOST=github.example gh pr merge 1", "git push --force origin HEAD",
        "git push --force-with-lease origin HEAD && true",
    ],
)
def test_implementer_terminal_blocks_forbidden_lifecycle_commands(command: str) -> None:
    assert validate_delivery_terminal_command(command, DeliveryPolicy("implementer"))


@pytest.mark.parametrize("command", ["git status", "git commit -m fix", "git rebase origin/main", "git push origin HEAD", "gh pr create --fill", "pytest -q"])
def test_implementer_terminal_keeps_implementation_capabilities(command: str) -> None:
    assert validate_delivery_terminal_command(command, DeliveryPolicy("implementer")) is None


def test_terminal_policy_is_part_of_the_central_dispatch_boundary() -> None:
    policy = DeliveryPolicy("reviewer")
    assert delivery_tool_block_reason("terminal", {"command": "git commit -m forbidden"}, policy)
    assert delivery_tool_block_reason("terminal", {"command": "git status"}, policy) is None
    assert delivery_tool_block_reason("terminal", {"command": "pytest -q", "background": True}, policy)


@pytest.mark.parametrize("command", ["git diff HEAD~1", "git show HEAD", "gh pr view 12", "gh pr checks 12", "python -m pytest -q", "ruff check tools"])
def test_reviewer_terminal_allows_bounded_read_and_verification(command: str) -> None:
    assert validate_delivery_terminal_command(command, DeliveryPolicy("reviewer")) is None


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m nope",
        "git push origin HEAD",
        "gh pr review 12 --approve",
        "gh issue close 1",
        "gh pr view 12 --web=true",
        "python -c 'open(\"x\", \"w\")'",
        "git diff | tee review.txt",
        "git grep -Osh needle",
    ],
)
def test_reviewer_terminal_blocks_mutations_and_shell_escape(command: str) -> None:
    assert validate_delivery_terminal_command(command, DeliveryPolicy("reviewer"))


def test_merger_requires_sha_bound_review_and_ci_evidence() -> None:
    with pytest.raises(ValueError, match="full 40-character SHA"):
        build_delivery_policy("merger", {**MERGER_EVIDENCE, "exact_sha": "abc1234"})
    with pytest.raises(ValueError, match="independent_review.*exact_sha"):
        build_delivery_policy("merger", {**MERGER_EVIDENCE, "independent_review": "PASS for another commit"})
    with pytest.raises(ValueError, match="ci_evidence.*exact_sha"):
        build_delivery_policy("merger", {**MERGER_EVIDENCE, "ci_evidence": "CI green"})


def test_merger_can_only_merge_the_evidenced_exact_sha() -> None:
    policy = build_delivery_policy("merger", MERGER_EVIDENCE)
    assert validate_delivery_terminal_command(f"gh pr merge 12 --squash --match-head-commit {SHA}", policy) is None
    assert validate_delivery_terminal_command("gh pr merge 12 --squash --match-head-commit " + "b" * 40, policy)
    assert validate_delivery_terminal_command("gh issue close 9", policy)
    assert validate_delivery_terminal_command("git commit -m fix", policy)


def test_closure_controller_requires_sha_bound_merge_and_acceptance_evidence() -> None:
    with pytest.raises(ValueError, match="merge_evidence.*merged_sha"):
        build_delivery_policy("closure_controller", {**CLOSURE_EVIDENCE, "merge_evidence": "merged"})
    with pytest.raises(ValueError, match="post_merge_acceptance.*merged_sha"):
        build_delivery_policy("closure_controller", {**CLOSURE_EVIDENCE, "post_merge_acceptance": "passed"})

    policy = build_delivery_policy("closure_controller", CLOSURE_EVIDENCE)
    assert validate_delivery_terminal_command(f"gh issue close 9 --comment {SHA}", policy) is None
    assert validate_delivery_terminal_command("gh issue close 9", policy)
    assert validate_delivery_terminal_command(f"gh pr close 12 --comment {SHA}", policy)
    assert validate_delivery_terminal_command(f"gh pr merge 12 --squash --match-head-commit {SHA}", policy)
    assert validate_delivery_terminal_command("git commit -m repair", policy)
