from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.task_card import CompiledIntent, ExecutionPlan, TaskCard
from tools.delegate_tool import ExternalCliChild, delegate_task
from tools import delegate_tool


def _write_managed_agents_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "managed_agents"
    path.mkdir(parents=True, exist_ok=True)
    yaml_path = path / "agents.yaml"
    yaml_path.write_text(
        """
version: "2026-05-21"
agents:
  - agent_id: claude
    name: Claude 主程执行官
    role: lead_implementer
    aliases: [主程, claude]
    role_summary: 复杂代码修改、命令执行、git 操作。
    model_ref: claude_opus
    runtime: claude_code_cli
    skills: [github-pr-workflow, codebase-inspection]
    tools: [file, terminal, git]
    permission: ask
    can_delegate: false
    capabilities: [code_edit, test_run, refactor]
    risk_allowed: [R0, R1, R2, R3]
  - agent_id: deepseek-tui
    name: DeepSeek 低成本快工
    role: fast_worker
    aliases: [低成本快工, deepseek]
    role_summary: 小改、小测试、低风险机械执行。外部搜索遵循 DeepSeek Worker Search Policy：本地优先，必要时用 anysearch-lite。
    model_ref: deepseek_pro
    skills: [spike, github-issues, anysearch-lite]
    tools: [file, terminal]
    permission: ask
    can_delegate: false
    capabilities: [test_generation, small_fix]
    risk_allowed: [R0, R1, R2]
  - agent_id: codex
    name: Codex 代码审查官
    role: principal_engineer
    aliases: [代码审查官, codex]
    role_summary: 外部 Codex CLI 审查器。
    model_ref: codex_cli
    runtime: codex_cli
    skills: [codex-superpowers, github-code-review]
    tools: [file, mcp-codegraph]
    permission: read_only
    can_delegate: false
    capabilities: [code_review, architecture_review]
    risk_allowed: [R0, R1, R2, R3, R4]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def _write_models_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "config"
    path.mkdir(parents=True, exist_ok=True)
    yaml_path = path / "models.yaml"
    yaml_path.write_text(
        """
models:
  claude_opus:
    provider: anthropic
    base_url: https://anthropic.test
    api_key_env: ANTHROPIC_API_KEY
    model: claude-opus-4-7
  deepseek_pro:
    provider: deepseek
    base_url: https://deepseek.test
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-v4-pro
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def _write_managed_policy_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "managed_agents"
    path.mkdir(parents=True, exist_ok=True)
    yaml_path = path / "policy.yaml"
    yaml_path.write_text(
        """
version: "2026-05-21"
priority_order:
  - safety
  - user_explicit_instruction
  - soul_global_policy
  - managed_agents_policy
  - router_policy
  - skill_policy
  - agent_preference
rules:
  - id: destructive_actions_require_safety_block
    priority: safety
    when:
      action_type: delete_file
    decision: deny
    reason: destructive_actions_require_human_approval
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def _make_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = MagicMock()
    parent._print_fn = None
    parent.session_id = "sess-1"
    parent._current_task_id = "task-1"
    parent._current_task_card = None
    return parent


def test_managed_gateway_preflight_runs_before_legacy_child_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    _write_managed_agents_yaml(tmp_path)
    _write_models_yaml(tmp_path)
    parent = _make_parent()

    with patch("tools.delegate_tool._run_single_child") as mock_run:
        mock_run.return_value = {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }
        result = json.loads(delegate_task(goal="implement feature", agent_id="claude", parent_agent=parent))

    assert result["results"][0]["status"] == "completed"
    assert mock_run.called
    child = mock_run.call_args.args[2]
    assert isinstance(child, ExternalCliChild)
    assert child.command == ["claude"]


def test_managed_gateway_accepts_agent_alias_before_legacy_child_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    _write_managed_agents_yaml(tmp_path)
    _write_models_yaml(tmp_path)
    parent = _make_parent()

    with patch("tools.delegate_tool._run_single_child") as mock_run:
        mock_run.return_value = {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }
        result = json.loads(delegate_task(goal="write tests", agent_id="低成本快工", parent_agent=parent))

    assert result["results"][0]["status"] == "completed"
    child = mock_run.call_args.args[2]
    assert child._subagent_agent_id == "deepseek-tui"
    assert child._skill_scope_agent_id == "deepseek-tui"
    assert "DeepSeek Worker Search Policy" in child.ephemeral_system_prompt
    assert "anysearch-lite" in child.ephemeral_system_prompt


def test_managed_agent_model_ref_overrides_provider_model_for_internal_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    _write_managed_agents_yaml(tmp_path)
    _write_models_yaml(tmp_path)
    parent = _make_parent()

    with (
        patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_runtime,
        patch("tools.delegate_tool._build_child_agent") as mock_build,
        patch("tools.delegate_tool._run_single_child") as mock_run,
    ):
        mock_runtime.side_effect = lambda requested, explicit_api_key=None, explicit_base_url=None, target_model=None: {
            "provider": requested,
            "base_url": explicit_base_url,
            "api_key": explicit_api_key,
            "api_mode": "anthropic_messages" if requested == "anthropic" else "chat_completions",
        }
        mock_build.return_value = MagicMock()
        mock_run.return_value = {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }

        delegate_task(
            tasks=[{"goal": "test", "agent_id": "低成本快工"}],
            parent_agent=parent,
        )

    call = mock_build.call_args.kwargs
    assert call["model"] == "deepseek-v4-pro"
    assert call["override_provider"] == "deepseek"
    assert call["override_base_url"] == "https://deepseek.test"
    assert call["override_api_key"] == "sk-deepseek"
    assert call["override_api_mode"] == "chat_completions"


def test_claude_agent_uses_external_claude_code_cli_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_managed_agents_yaml(tmp_path)
    parent = _make_parent()

    with (
        patch("tools.delegate_tool._build_child_agent") as mock_build,
        patch("tools.delegate_tool._run_single_child") as mock_run,
    ):
        mock_run.return_value = {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }

        delegate_task(goal="complex refactor", agent_id="claude", parent_agent=parent)

    assert mock_build.call_count == 0
    child = mock_run.call_args.args[2]
    assert isinstance(child, ExternalCliChild)
    assert child._subagent_agent_id == "claude"
    assert child.command == ["claude"]
    assert "complex refactor" == child._goal


def test_claude_code_cli_runtime_preserves_context_in_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_managed_agents_yaml(tmp_path)
    parent = _make_parent()

    with patch("tools.delegate_tool._run_single_child") as mock_run:
        mock_run.return_value = {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }

        delegate_task(
            goal="complex refactor",
            context="preserve this context",
            agent_id="claude",
            parent_agent=parent,
        )

    child = mock_run.call_args.args[2]
    assert isinstance(child, ExternalCliChild)
    assert child._goal == "complex refactor\n\nContext:\npreserve this context"


def test_codex_agent_uses_external_codex_cli_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_managed_agents_yaml(tmp_path)
    parent = _make_parent()

    with (
        patch("tools.delegate_tool._build_child_agent") as mock_build,
        patch("tools.delegate_tool._run_single_child") as mock_run,
    ):
        mock_run.return_value = {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }

        delegate_task(goal="review this design", agent_id="codex", parent_agent=parent)

    assert mock_build.call_count == 0
    child = mock_run.call_args.args[2]
    assert isinstance(child, ExternalCliChild)
    assert child._subagent_agent_id == "codex"
    assert child.command == ["codex", "exec", "--sandbox", "read-only"]
    assert child.output_last_message is True


def test_managed_gateway_rejection_short_circuits_legacy_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_managed_agents_yaml(tmp_path)
    parent = _make_parent()
    task_card = TaskCard(
        task_id="task-1",
        session_id="sess-1",
        raw_user_request="implement feature",
        compiled_intent=CompiledIntent(real_task="implement feature", task_category="feature"),
        execution_plan=ExecutionPlan(mode="single_agent", agents=["claude"], delegation_reason="route"),
    )
    parent._current_task_card = task_card
    task_card.risk_level = "R4"

    with patch("tools.delegate_tool._run_single_child") as mock_run:
        result = delegate_task(goal="implement feature", agent_id="claude", parent_agent=parent)

    assert "Managed agent preflight rejected delegation" in result
    assert mock_run.call_count == 0


def test_managed_preflight_validates_requested_agent_not_parent_task_route(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_managed_agents_yaml(tmp_path)
    parent = _make_parent()
    task_card = TaskCard(
        task_id="task-1",
        session_id="sess-1",
        raw_user_request="implement feature",
        compiled_intent=CompiledIntent(real_task="implement feature", task_category="feature"),
        execution_plan=ExecutionPlan(mode="single_agent", agents=["claude"], delegation_reason="route"),
    )
    task_card.risk_level = "R3"
    parent._current_task_card = task_card

    with patch("tools.delegate_tool._run_single_child") as mock_run:
        result = delegate_task(goal="write tests", agent_id="deepseek-tui", parent_agent=parent)

    assert "Managed agent preflight rejected delegation" in result
    assert "deepseek-tui" in result
    assert mock_run.call_count == 0


def test_managed_preflight_event_log_failure_blocks_legacy_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_managed_agents_yaml(tmp_path)
    parent = _make_parent()

    with (
        patch("agent.session_event_log.EventLog") as mock_event_log,
        patch("tools.delegate_tool._run_single_child") as mock_run,
    ):
        mock_event_log.return_value.log_dispatch_decision.side_effect = RuntimeError("event log unavailable")
        result = delegate_task(goal="implement feature", agent_id="claude", parent_agent=parent)

    assert "Managed agent preflight failed" in result
    assert "event log unavailable" in result
    assert mock_run.call_count == 0


def test_managed_read_only_profile_blocks_write_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_managed_agents_yaml(tmp_path)

    _, profile = delegate_tool._load_managed_subagent_profile("codex")

    assert profile["permission_mode"] == "read_only"
    assert profile["isolation"] == "readonly"
    assert {"write_file", "patch"} <= set(profile["blocked_tools"])
    assert profile["skills"] == ["codex-superpowers", "github-code-review"]


def test_managed_preflight_loads_policy_yaml_and_blocks_delete_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_managed_agents_yaml(tmp_path)
    _write_managed_policy_yaml(tmp_path)
    parent = _make_parent()
    task_card = TaskCard(
        task_id="task-1",
        session_id="sess-1",
        raw_user_request="delete generated file",
        compiled_intent=CompiledIntent(real_task="delete generated file", task_category="feature"),
        execution_plan=ExecutionPlan(mode="single_agent", agents=["claude"], delegation_reason="route"),
    )
    task_card.risk_level = "R2"
    task_card.action_type = "delete_file"
    parent._current_task_card = task_card

    with patch("tools.delegate_tool._run_single_child") as mock_run:
        result = delegate_task(goal="delete generated file", agent_id="claude", parent_agent=parent)

    assert "Managed agent preflight rejected delegation" in result
    assert "Policy denied" in result
    assert mock_run.call_count == 0
