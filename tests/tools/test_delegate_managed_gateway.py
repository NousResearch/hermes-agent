from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.task_card import CompiledIntent, ExecutionPlan, TaskCard
from tools.delegate_tool import delegate_task
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
    name: Claude Code
    role: lead_implementer
    tools: [file, terminal, git]
    permission: ask
    can_delegate: false
    capabilities: [code_edit, test_run, refactor]
    risk_allowed: [R0, R1, R2, R3]
  - agent_id: deepseek-tui
    name: DeepSeek TUI
    role: fast_worker
    tools: [file, terminal]
    permission: ask
    can_delegate: false
    capabilities: [test_generation, small_fix]
    risk_allowed: [R0, R1, R2]
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
        result = json.loads(delegate_task(goal="implement feature", agent_id="claude", parent_agent=parent))

    assert result["results"][0]["status"] == "completed"
    assert mock_run.called


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
    config_path = tmp_path / "configs" / "managed_agents" / "agents.yaml"
    raw = config_path.read_text(encoding="utf-8")
    raw += """
  - agent_id: codex
    name: Codex
    role: principal_engineer
    tools: [file]
    permission: read_only
    can_delegate: false
    capabilities: [code_review]
    risk_allowed: [R0, R1, R2]
"""
    config_path.write_text(raw, encoding="utf-8")

    _, profile = delegate_tool._load_managed_subagent_profile("codex")

    assert profile["permission_mode"] == "read_only"
    assert profile["isolation"] == "readonly"
    assert {"write_file", "patch"} <= set(profile["blocked_tools"])
