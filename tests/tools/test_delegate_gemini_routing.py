from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.antigravity_delegate import AntigravityDelegateChild
from agent.antigravity_worker import AntigravityResult
from agent.gemini_route_receipts import GeminiReceiptStore
from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


def parent() -> MagicMock:
    value = MagicMock()
    value.base_url = "https://example.invalid/v1"
    value.api_key = "test"
    value.provider = "openai-codex"
    value.api_mode = "chat_completions"
    value.model = "gpt-5.6-sol"
    value.platform = "cli"
    value.providers_allowed = None
    value.providers_ignored = None
    value.providers_order = None
    value.provider_sort = None
    value._session_db = None
    value._delegate_depth = 0
    value._active_children = []
    value._active_children_lock = threading.Lock()
    value._print_fn = None
    value.tool_progress_callback = None
    value.thinking_callback = None
    value._current_turn_id = "turn-1"
    value.session_id = "parent-1"
    value._memory_manager = None
    value.session_estimated_cost_usd = 0.0
    return value


def routing_config(**overrides):
    gemini = {
        "enabled": True,
        "profiles": ["default"],
        "default_route": "gemini",
        "default_data_classification": "standard",
        "command": "agy",
        "model": "gemini-3.8-flash-low",
        "effort": "low",
        "timeout_seconds": 120,
        "max_input_bytes": 262144,
        "max_output_bytes": 131072,
        "fallback_to_delegation_model": True,
        "receipt_db": "routing/gemini-routing.sqlite3",
    }
    gemini.update(overrides)
    return {
        "max_iterations": 2,
        "max_concurrent_children": 3,
        "max_spawn_depth": 1,
        "gemini_routing": gemini,
    }


def fake_child(summary: str = "sol answer") -> MagicMock:
    child = MagicMock()
    child.session_id = "sol-child"
    child.model = "gpt-5.6-sol"
    child._delegate_role = "leaf"
    child._delegate_saved_tool_names = []
    child.tool_progress_callback = None
    child._credential_pool = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.session_estimated_cost_usd = 0.0
    child.run_conversation.return_value = {
        "final_response": summary,
        "completed": True,
        "api_calls": 1,
        "messages": [],
    }
    child.get_activity_summary.return_value = {
        "api_call_count": 0,
        "max_iterations": 2,
        "current_tool": None,
    }
    return child


def test_schema_exposes_only_bounded_routing_metadata():
    properties = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    for name in ("route", "data_classification", "output_contract", "output_schema"):
        assert name in properties
        assert name in properties["tasks"]["items"]["properties"]
    assert properties["route"]["enum"] == ["auto", "gemini", "sol"]
    assert properties["data_classification"]["enum"] == ["standard", "restricted"]
    assert properties["output_contract"]["enum"] == ["text", "json"]


def test_disabled_config_preserves_existing_sol_child_path():
    sol = fake_child()
    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config(enabled=False)),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol) as build_sol,
        patch("tools.delegate_tool._build_antigravity_delegate_child") as build_gemini,
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    assert result["results"][0]["summary"] == "sol answer"
    build_sol.assert_called_once()
    build_gemini.assert_not_called()


def test_eligible_leaf_builds_gemini_adapter_with_sol_fallback():
    routed = fake_child("gemini answer")
    sol = fake_child()
    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol) as build_sol,
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed) as build_gemini,
    ):
        result = json.loads(
            delegate_task(
                goal="summarize",
                route="auto",
                data_classification="standard",
                output_contract="text",
                parent_agent=parent(),
            )
        )

    assert result["results"][0]["summary"] == "gemini answer"
    build_sol.assert_called_once()
    build_gemini.assert_called_once()
    kwargs = build_gemini.call_args.kwargs
    assert kwargs["fallback_child"] is sol
    assert kwargs["route_reason"] == "eligible output-only leaf delegation"


@pytest.mark.parametrize(
    ("task", "role", "reason"),
    [
        ({"goal": "private", "route": "gemini", "data_classification": "restricted"}, "leaf", "restricted"),
        ({"goal": "coordinate", "route": "gemini"}, "orchestrator", "orchestrator"),
        ({"goal": "stay sol", "route": "sol"}, "leaf", "Sol"),
    ],
)
def test_hard_exclusions_use_sol_and_surface_route_reason(task, role, reason):
    sol = fake_child()
    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch("tools.delegate_tool._build_antigravity_delegate_child") as build_gemini,
    ):
        result = json.loads(delegate_task(tasks=[{**task, "role": role}], parent_agent=parent()))

    build_gemini.assert_not_called()
    assert result["results"][0]["route"] == "sol"
    assert reason.lower() in result["results"][0]["route_reason"].lower()


def test_invalid_routing_metadata_fails_before_child_construction():
    with patch("tools.delegate_tool._build_child_preserving_parent_tools") as builder:
        result = json.loads(
            delegate_task(
                tasks=[{"goal": "x", "data_classification": "secret"}],
                parent_agent=parent(),
            )
        )
    assert "data_classification" in result["error"]
    builder.assert_not_called()


class FakeWorker:
    def __init__(self, result: AntigravityResult, *, invoke_started: bool = True):
        self.result = result
        self.invoke_started = invoke_started
        self.closed = False

    def run(self, *, goal, context, output_schema, on_process_started=None):
        if self.invoke_started and on_process_started:
            on_process_started()
        return self.result

    def cancel(self):
        pass

    def close(self):
        self.closed = True


def worker_result(*, ok: bool) -> AntigravityResult:
    return AntigravityResult(
        status="success" if ok else "failed",
        response="gemini answer" if ok else None,
        conversation_id="conversation" if ok else None,
        usage={"input_tokens": 1},
        raw_envelope={"status": "SUCCESS" if ok else "ERROR"},
        exit_code=0 if ok else 7,
        duration_ms=12,
        error_code=None if ok else "nonzero_exit",
        error_message=None if ok else "Antigravity exited unsuccessfully",
    )


def make_adapter(tmp_path: Path, worker: FakeWorker, *, fallback=True):
    return AntigravityDelegateChild(
        worker=worker,
        fallback_child=fake_child("fallback answer") if fallback else None,
        store=GeminiReceiptStore(tmp_path / "routing.sqlite3"),
        task_index=0,
        goal="summarize",
        context="bounded",
        output_schema=None,
        output_contract="text",
        route_requested="auto",
        route_reason="eligible output-only leaf delegation",
        data_classification="standard",
        requested_provider="antigravity-subscription",
        requested_model="gemini-3.8-flash-low",
        requested_effort="low",
        parent_session_id="parent",
        parent_turn_id="turn",
    )


def test_adapter_returns_gemini_output_and_records_two_phase_receipt(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=True)))

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["final_response"] == "gemini answer"
    assert result["completed"] is True
    rows = adapter.store.list_started_attempts_for_day(
        adapter.store.get_attempt(adapter.receipt_id)["routing_day"]
    )
    assert len(rows) == 1
    assert rows[0]["worker_status"] == "completed"
    assert rows[0]["process_started_at_utc"] is not None
    assert rows[0]["fallback_used"] == 0


def test_adapter_records_failure_then_runs_prebuilt_sol_fallback(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["final_response"] == "fallback answer"
    assert result["route"] == "gemini_then_sol"
    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["worker_status"] == "failed"
    assert row["fallback_used"] == 1
    assert row["error_code"] == "nonzero_exit"


def test_adapter_without_fallback_returns_a_structured_failure(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)), fallback=False)

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["completed"] is False
    assert result["final_response"] == ""
    assert result["error"] == "Antigravity exited unsuccessfully"
    assert result["route"] == "gemini"


def test_adapter_does_not_mark_process_started_when_spawn_never_happens(tmp_path: Path):
    adapter = make_adapter(
        tmp_path,
        FakeWorker(worker_result(ok=False), invoke_started=False),
        fallback=False,
    )

    adapter.run_conversation("summarize", task_id="child-task")

    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["process_started_at_utc"] is None
