from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.antigravity_delegate import AntigravityDelegateChild
from agent.antigravity_worker import AntigravityResult
from agent.gemini_route_receipts import GeminiReceiptStore
from agent.route_receipts import RouteReceiptChild, RouteReceiptStore
from run_agent import AIAgent
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_antigravity_delegate_child,
    _merge_child_route_metadata,
    _run_single_child,
    delegate_task,
)


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


class RemovalTrackingList(list):
    def __init__(self, *items):
        super().__init__(items)
        self.removed = threading.Event()

    def remove(self, item):
        super().remove(item)
        self.removed.set()


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
        "retention": {"raw_days": 30, "aggregate_days": 180},
        "review": {
            "enabled": False,
            "timezone": "America/Los_Angeles",
            "sample_size": 5,
            "not_before_local": "00:15",
            "review_provider": "openai-codex",
            "review_model": "gpt-5.6-sol",
            "alert_target": "slack:C0AEMP1AG0H",
            "alert_workspace_id": "",
        },
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
    child.provider = "openai-codex"
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
    assert "enum" not in properties["data_classification"]
    assert "any other value" in properties["data_classification"]["description"]
    assert "enum" not in properties["tasks"]["items"]["properties"]["data_classification"]
    assert properties["output_contract"]["enum"] == ["text", "json"]


def test_disabled_config_preserves_sol_execution_and_writes_frontier_receipt():
    sol = fake_child()
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config(enabled=False)),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol) as build_sol,
        patch("tools.delegate_tool._build_antigravity_delegate_child") as build_gemini,
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    public = result["results"][0]
    assert public["summary"] == "sol answer"
    assert public["route"] == "sol"
    assert public["worker_route"] == "sol"
    assert public["route_receipt_id"]
    assert public["fallback_used"] is False
    assert stops[0]["worker_route"] == "sol"
    assert stops[0]["route_receipt_id"] == public["route_receipt_id"]
    build_sol.assert_called_once()
    build_gemini.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"command": ""},
        {"model": False},
        {"effort": None},
        {"timeout_seconds": 0},
        {"max_input_bytes": 0},
        {"max_output_bytes": 0},
        {"fallback_to_delegation_model": 0},
        {"receipt_db": ""},
        {"extra_args": None},
        {"extra_args": ["--add-dir=/tmp"]},
        {"extra_args": ["--model=other"]},
        {"extra_args": ["--effort=high"]},
        {"extra_args": ["--mode=agent"]},
        {"extra_args": ["--sandbox=false"]},
        {"extra_args": ["--no-sandbox"]},
        {"extra_args": ["--print"]},
        {"extra_args": ["--output-format=text"]},
        {"extra_args": ["--print-timeout=999s"]},
        {"extra_args": ["--json-schema=other.json"]},
        {"extra_args": ["--disable-slash-commands=false"]},
        {"retention": None},
        {"review": None},
    ],
)
def test_malformed_enabled_routing_config_fails_closed_to_sol(overrides):
    sol = fake_child()
    gemini = fake_child("gemini answer")

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config(**overrides)),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=gemini) as build_gemini,
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    assert result["results"][0]["summary"] == "sol answer"
    assert result["results"][0]["worker_route"] == "sol"
    assert result["results"][0]["route_receipt_id"]
    build_gemini.assert_not_called()


def test_absent_gemini_routing_config_still_writes_frontier_receipt():
    sol = fake_child()
    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch("tools.delegate_tool._build_antigravity_delegate_child") as build_gemini,
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    public = result["results"][0]
    assert public["summary"] == "sol answer"
    assert public["route"] == "sol"
    assert public["worker_route"] == "sol"
    assert public["route_receipt_id"]
    assert public["fallback_used"] is False
    build_gemini.assert_not_called()


@pytest.mark.parametrize(
    "alert_target",
    ["email:operator@example.com", "slack:", "slack:C:extra", "slack: C0A12345678"],
)
def test_enabled_review_with_invalid_slack_alert_target_fails_closed_to_sol(alert_target):
    config = routing_config()
    review = config["gemini_routing"]["review"]
    review["enabled"] = True
    review["alert_target"] = alert_target
    review["alert_workspace_id"] = "T0A12345678"
    sol = fake_child()
    gemini = fake_child("gemini answer")

    with (
        patch("tools.delegate_tool._load_config", return_value=config),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=gemini) as build_gemini,
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    assert result["results"][0]["summary"] == "sol answer"
    assert result["results"][0]["worker_route"] == "sol"
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
    assert isinstance(kwargs["fallback_child"], RouteReceiptChild)
    assert kwargs["fallback_child"].child is sol
    assert kwargs["route_reason"] == "eligible output-only leaf delegation"


def test_receipt_initialization_failure_runs_prebuilt_sol_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sol = fake_child("fallback answer")
    parent_agent = parent()
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch(
            "agent.gemini_route_receipts.GeminiReceiptStore",
            side_effect=RuntimeError("receipt unavailable"),
        ),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(
            delegate_task(goal="summarize", parent_agent=parent_agent)
        )

    public = result["results"][0]
    assert public["summary"] == "fallback answer"
    assert public["route"] == "sol_after_receipt_error"
    assert public["worker_route"] == "sol"
    assert public["fallback_used"] is True
    assert public["gemini_error_code"] == "receipt_initialization_failed"
    assert public["route_receipt_id"]
    assert stops[0]["worker_route"] == "sol"
    assert stops[0]["fallback_used"] is True
    assert stops[0]["gemini_error_code"] == "receipt_initialization_failed"
    assert stops[0]["route_receipt_id"] == public["route_receipt_id"]
    assert parent_agent._active_children == []
    row = RouteReceiptStore(
        tmp_path / "routing" / "gemini-routing.sqlite3"
    ).get_attempt(public["route_receipt_id"])
    assert row["route_decision"] == "frontier"
    assert row["status"] == "completed"


def test_receipt_initialization_failure_without_fallback_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch(
            "tools.delegate_tool._load_config",
            return_value=routing_config(fallback_to_delegation_model=False),
        ),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch(
            "agent.gemini_route_receipts.GeminiReceiptStore",
            side_effect=RuntimeError("receipt unavailable"),
        ),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    public = result["results"][0]
    assert public["status"] == "failed"
    assert public["error"] == "Gemini receipt initialization failed"
    assert public["gemini_error_code"] == "receipt_initialization_failed"
    assert public["fallback_used"] is False
    assert "route_receipt_id" in public
    assert public["route_receipt_id"] is None
    assert "route_receipt_id" in stops[0]
    assert stops[0]["route_receipt_id"] is None


@pytest.mark.parametrize(
    "receipt_error",
    [
        RuntimeError("receipt unavailable"),
        sqlite3.OperationalError("database unavailable"),
        OSError("filesystem unavailable"),
    ],
)
def test_ordinary_frontier_receipt_initialization_failure_is_structured_and_does_not_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_error: Exception,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sol = fake_child("must not execute")
    parent_agent = parent()

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch(
            "agent.route_receipts.RouteReceiptStore",
            side_effect=receipt_error,
        ),
    ):
        result = json.loads(
            delegate_task(goal="summarize", parent_agent=parent_agent)
        )

    public = result["results"][0]
    assert public["status"] == "failed"
    assert public["error"] == "Delegation receipt initialization failed"
    assert public["worker_route"] == "sol"
    assert public["route_receipt_id"] is None
    sol.run_conversation.assert_not_called()
    sol.close.assert_called_once()
    assert parent_agent._active_children == []


def test_omitted_single_task_metadata_uses_standard_profile_defaults():
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
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed) as build_gemini,
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    assert result["results"][0]["summary"] == "gemini answer"
    build_gemini.assert_called_once()


def test_omitted_classification_uses_restricted_profile_default_and_blocks_explicit_gemini():
    sol = fake_child()
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch(
            "tools.delegate_tool._load_config",
            return_value=routing_config(default_data_classification="restricted"),
        ),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=sol),
        patch("tools.delegate_tool._build_antigravity_delegate_child") as build_gemini,
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(
            delegate_task(goal="summarize", route="gemini", parent_agent=parent())
        )

    assert result["results"][0]["worker_route"] == "sol"
    assert "route_receipt_id" in result["results"][0]
    assert isinstance(result["results"][0]["route_receipt_id"], str)
    assert result["results"][0]["route_receipt_id"]
    assert stops[0]["route_receipt_id"] == result["results"][0]["route_receipt_id"]
    assert "restricted" in result["results"][0]["route_reason"]
    build_gemini.assert_not_called()


@pytest.mark.parametrize(
    "classification",
    ["restricted", "sensitive", "local-only", "secret", "ambiguous", "unknown-value"],
)
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_nonstandard_classification_fails_closed_to_sol(
    classification: str, batch: bool
):
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
        if batch:
            response = delegate_task(
                tasks=[{
                    "goal": "summarize",
                    "route": "gemini",
                    "data_classification": classification,
                }],
                parent_agent=parent(),
            )
        else:
            response = delegate_task(
                goal="summarize",
                route="gemini",
                data_classification=classification,
                parent_agent=parent(),
            )
        result = json.loads(response)

    assert result["results"][0]["summary"] == "sol answer"
    assert result["results"][0]["worker_route"] == "sol"
    assert "restricted" in result["results"][0]["route_reason"]
    build_gemini.assert_not_called()


def test_terminal_null_route_metadata_clears_stale_pre_execution_value():
    entry = {"route_receipt_id": "grt_stale"}

    _merge_child_route_metadata(entry, None, {"route_receipt_id": None})

    assert "route_receipt_id" in entry
    assert entry["route_receipt_id"] is None


def test_public_results_and_lifecycle_hooks_carry_exact_route_metadata():
    parent_agent = parent()
    parent_agent._subagent_id = "parent-sa"
    routed = fake_child("gemini answer")
    routed.requested_provider = "google-antigravity"
    routed.requested_model = "gemini-3.8-flash-low"
    routed._route_metadata = {
        "route": "gemini",
        "route_reason": "eligible output-only leaf delegation",
        "worker_route": "gemini",
        "worker_provider": "google-antigravity",
        "worker_model_requested": "gemini-3.8-flash-low",
        "route_receipt_id": "grt_123",
        "fallback_used": False,
    }
    routed.run_conversation.return_value = {
        "final_response": "gemini answer",
        "completed": True,
        "api_calls": 1,
        "messages": [],
        "route": "gemini",
        "route_reason": "eligible output-only leaf delegation",
        "worker_route": "gemini",
        "worker_provider": "google-antigravity",
        "worker_model_requested": "gemini-3.8-flash-low",
        "route_receipt_id": "grt_123",
        "fallback_used": False,
    }
    starts: list[dict] = []
    stops: list[dict] = []

    def start_hook(event, **kwargs):
        if event == "subagent_start":
            starts.append(kwargs)

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=fake_child()),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=start_hook),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(
            delegate_task(
                goal="summarize",
                route="gemini",
                data_classification="standard",
                parent_agent=parent_agent,
            )
        )

    public = result["results"][0]
    expected = {
        "worker_route": "gemini",
        "worker_provider": "google-antigravity",
        "worker_model_requested": "gemini-3.8-flash-low",
        "route_receipt_id": "grt_123",
        "fallback_used": False,
    }
    assert {key: public[key] for key in expected} == expected
    assert {key: starts[0][key] for key in expected} == expected
    assert starts[0]["parent_subagent_id"] == "parent-sa"
    assert {key: stops[0][key] for key in expected} == expected


def test_terminal_result_overrides_start_metadata_after_gemini_fallback():
    routed = fake_child("fallback answer")
    routed._route_metadata = {
        "route": "gemini",
        "route_reason": "eligible output-only leaf delegation",
        "worker_route": "gemini",
        "worker_provider": "google-antigravity",
        "worker_model_requested": "gemini-3.8-flash-low",
        "route_receipt_id": "grt_fallback",
        "fallback_used": False,
    }
    routed.run_conversation.return_value = {
        "final_response": "fallback answer",
        "completed": True,
        "api_calls": 1,
        "messages": [],
        "route": "gemini_then_sol",
        "route_reason": "eligible output-only leaf delegation",
        "worker_route": "sol",
        "worker_provider": "openai-codex",
        "worker_model_requested": "gpt-5.6-sol",
        "route_receipt_id": "grt_fallback",
        "fallback_used": True,
        "gemini_error_code": "worker_timeout",
    }
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=fake_child()),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    expected = {
        "route": "gemini_then_sol",
        "worker_route": "sol",
        "worker_provider": "openai-codex",
        "worker_model_requested": "gpt-5.6-sol",
        "route_receipt_id": "grt_fallback",
        "fallback_used": True,
        "gemini_error_code": "worker_timeout",
    }
    public = result["results"][0]
    assert {key: public[key] for key in expected} == expected
    stop_expected = {key: value for key, value in expected.items() if key != "route"}
    assert {key: stops[0][key] for key in stop_expected} == stop_expected


def test_outer_exception_keeps_current_gemini_fallback_metadata():
    routed = fake_child()
    routed._route_metadata = {
        "route": "gemini_then_sol",
        "route_reason": "eligible output-only leaf delegation",
        "worker_route": "sol",
        "worker_provider": "openai-codex",
        "worker_model_requested": "gpt-5.6-sol",
        "route_receipt_id": "grt_fallback_outer",
        "fallback_used": True,
        "gemini_error_code": "worker_timeout",
    }
    routed.run_conversation.side_effect = RuntimeError("private fallback details")
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=fake_child()),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    public = result["results"][0]
    assert public["status"] == "error"
    assert public["worker_route"] == "sol"
    assert public["worker_provider"] == "openai-codex"
    assert public["worker_model_requested"] == "gpt-5.6-sol"
    assert public["route_receipt_id"] == "grt_fallback_outer"
    assert public["fallback_used"] is True
    assert public["gemini_error_code"] == "worker_timeout"
    assert stops[0]["worker_route"] == "sol"
    assert stops[0]["fallback_used"] is True
    assert stops[0]["gemini_error_code"] == "worker_timeout"


def test_malformed_terminal_result_keeps_current_gemini_fallback_metadata():
    routed = fake_child()
    routed._route_metadata = {
        "route": "gemini_then_sol",
        "route_reason": "eligible output-only leaf delegation",
        "worker_route": "sol",
        "worker_provider": "openai-codex",
        "worker_model_requested": "gpt-5.6-sol",
        "route_receipt_id": "grt_malformed",
        "fallback_used": True,
        "gemini_error_code": "invalid_response",
    }
    routed.run_conversation.return_value = None
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=fake_child()),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    public = result["results"][0]
    assert public["status"] == "error"
    assert public["worker_route"] == "sol"
    assert public["route_receipt_id"] == "grt_malformed"
    assert public["fallback_used"] is True
    assert public["gemini_error_code"] == "invalid_response"
    assert stops[0]["worker_route"] == "sol"
    assert stops[0]["fallback_used"] is True


@pytest.mark.parametrize("fabricated_status", ["error", "interrupted"])
def test_batch_fabricated_exit_keeps_current_gemini_fallback_metadata(
    fabricated_status: str,
):
    routed_children = [fake_child(), fake_child()]
    for index, child in enumerate(routed_children):
        child._route_metadata = {
            "route": "gemini_then_sol",
            "route_reason": "eligible output-only leaf delegation",
            "worker_route": "sol",
            "worker_provider": "openai-codex",
            "worker_model_requested": "gpt-5.6-sol",
            "route_receipt_id": f"grt_batch_{index}",
            "fallback_used": True,
            "gemini_error_code": "worker_exception",
        }
    parent_agent = parent()
    parent_agent._interrupt_requested = fabricated_status == "interrupted"
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    def fabricated_run(*_args, **_kwargs):
        if fabricated_status == "error":
            raise RuntimeError("fabricated future failure")
        time.sleep(0.1)
        return {"status": "completed", "summary": "too late", "task_index": 0}

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch(
            "tools.delegate_tool._build_child_preserving_parent_tools",
            side_effect=[fake_child(), fake_child()],
        ),
        patch(
            "tools.delegate_tool._build_antigravity_delegate_child",
            side_effect=routed_children,
        ),
        patch("tools.delegate_tool._run_single_child", side_effect=fabricated_run),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(
            delegate_task(
                tasks=[
                    {"goal": "first routed task"},
                    {"goal": "second routed task"},
                ],
                parent_agent=parent_agent,
            )
        )

    assert len(result["results"]) == 2
    for index, public in enumerate(result["results"]):
        assert public["status"] == fabricated_status
        assert public["worker_route"] == "sol"
        assert public["route_receipt_id"] == f"grt_batch_{index}"
        assert public["fallback_used"] is True
        assert public["gemini_error_code"] == "worker_exception"
    assert len(stops) == 2
    assert all(stop["worker_route"] == "sol" for stop in stops)
    assert all(stop["fallback_used"] is True for stop in stops)


def test_real_adapter_timeout_keeps_current_gemini_fallback_metadata(tmp_path: Path):
    routed = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    assert routed.fallback_child is not None

    def slow_fallback(**_kwargs):
        time.sleep(1.0)
        return {"final_response": "too late", "completed": True}

    routed.fallback_child.run_conversation.side_effect = slow_fallback
    stops: list[dict] = []

    def stop_hook(event, **kwargs):
        if event == "subagent_stop":
            stops.append(kwargs)

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._get_child_timeout", return_value=0.5),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=fake_child()),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed),
        patch("hermes_cli.plugins.invoke_hook", side_effect=stop_hook),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    public = result["results"][0]
    assert public["status"] == "timeout"
    assert public["worker_route"] == "sol"
    assert public["worker_provider"] == "openai-codex"
    assert public["worker_model_requested"] == "gpt-5.6-sol"
    assert public["route_receipt_id"] == routed.receipt_id
    assert public["fallback_used"] is True
    assert public["gemini_error_code"] == "nonzero_exit"
    assert stops[0]["worker_route"] == "sol"
    assert stops[0]["fallback_used"] is True
    assert stops[0]["gemini_error_code"] == "nonzero_exit"


def test_parent_timeout_cancels_blocked_gemini_worker_without_late_receipt_mutation(
    tmp_path: Path,
):
    worker = BlockingWorker()
    routed = make_adapter(tmp_path, worker)
    assert routed.fallback_child is not None

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._get_child_timeout", return_value=0.1),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=fake_child()),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    try:
        assert result["results"][0]["status"] == "timeout"
        assert worker.cancelled.wait(timeout=1)
        assert worker.finished.wait(timeout=1)
        assert routed._execution_done.is_set()
        assert routed.fallback_child.run_conversation.call_count == 0
        terminal = routed.store.get_attempt(routed.receipt_id)
        assert terminal["worker_status"] == "cancelled"
        assert routed.store.get_attempt(routed.receipt_id) == terminal
    finally:
        worker.release.set()


def test_cancel_admitted_during_receipt_completion_wins_terminal_publication(
    tmp_path: Path,
):
    adapter = make_adapter(
        tmp_path,
        FakeWorker(worker_result(ok=True)),
        fallback=False,
    )
    completion_entered = threading.Event()
    completion_release = threading.Event()
    original_complete_attempt = adapter.store.complete_attempt

    def blocked_complete_attempt(receipt_id, **kwargs):
        if kwargs.get("worker_status") == "completed":
            completion_entered.set()
            assert completion_release.wait(timeout=3)
        return original_complete_attempt(receipt_id, **kwargs)

    adapter.store.complete_attempt = blocked_complete_attempt
    result_holder: dict[str, dict] = {}
    run_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", adapter.run_conversation("prompt")
        )
    )
    run_thread.start()
    assert completion_entered.wait(timeout=1)

    cancel_thread = threading.Thread(target=adapter.cancel)
    cancel_thread.start()
    assert adapter._cancel_event.wait(timeout=1)
    completion_release.set()
    run_thread.join(timeout=3)
    cancel_thread.join(timeout=3)

    assert not run_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert result_holder["result"]["gemini_error_code"] == "cancelled"
    receipt = adapter.store.get_attempt(adapter.receipt_id)
    assert receipt["worker_status"] == "cancelled"
    assert receipt["response_text"] is None


def test_cancel_admitted_during_fallback_receipt_wins_result_publication(
    tmp_path: Path,
):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    fallback_completion_entered = threading.Event()
    fallback_completion_release = threading.Event()
    original_record_fallback = adapter.store.record_fallback_outcome

    def blocked_record_fallback(receipt_id, **kwargs):
        fallback_completion_entered.set()
        assert fallback_completion_release.wait(timeout=3)
        return original_record_fallback(receipt_id, **kwargs)

    adapter.store.record_fallback_outcome = blocked_record_fallback
    result_holder: dict[str, dict] = {}
    run_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", adapter.run_conversation("prompt")
        )
    )
    run_thread.start()
    assert fallback_completion_entered.wait(timeout=1)

    cancel_thread = threading.Thread(target=adapter.cancel)
    cancel_thread.start()
    assert adapter._cancel_event.wait(timeout=1)
    fallback_completion_release.set()
    run_thread.join(timeout=3)
    cancel_thread.join(timeout=3)

    assert not run_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert result_holder["result"]["gemini_error_code"] == "cancelled"
    receipt = adapter.store.get_attempt(adapter.receipt_id)
    assert receipt["terminal_worker_route"] is None
    assert receipt["terminal_worker_status"] is None


@pytest.mark.parametrize("close_fails", [False, True])
def test_parent_timeout_retains_child_until_unconfirmed_cancellation_finishes(
    close_fails: bool,
):
    child = fake_child()
    run_started = threading.Event()
    run_release = threading.Event()
    close_attempted = threading.Event()
    cleanup_done = threading.Event()

    def blocked_run(**_kwargs):
        run_started.set()
        run_release.wait(timeout=5)
        return {"final_response": "", "completed": False, "messages": []}

    child.run_conversation.side_effect = blocked_run
    def failed_hard_interrupt(_message=None):
        raise RuntimeError("private /tmp/delegate.sock?token=secret")

    child.hard_interrupt = failed_hard_interrupt
    def close_child():
        close_attempted.set()
        if close_fails:
            raise RuntimeError("private final close failure")
        cleanup_done.set()

    child.close.side_effect = close_child
    parent_agent = parent()
    parent_agent._active_children = RemovalTrackingList(child)

    try:
        with patch("tools.delegate_tool._get_child_timeout", return_value=0.1):
            result = _run_single_child(
                task_index=0,
                goal="blocked child",
                child=child,
                parent_agent=parent_agent,
            )

        assert run_started.is_set()
        assert result["status"] == "error"
        assert result["exit_reason"] == "cancellation_unconfirmed"
        assert result["error"] == "delegation cancellation could not be confirmed"
        assert "private" not in str(result)
        assert child in parent_agent._active_children
        assert not cleanup_done.is_set()

        run_release.set()
        assert close_attempted.wait(timeout=1)
        if close_fails:
            assert child in parent_agent._active_children
            assert not cleanup_done.is_set()
        else:
            assert cleanup_done.is_set()
            assert parent_agent._active_children.removed.wait(timeout=1)
            assert child not in parent_agent._active_children
    finally:
        run_release.set()


def test_parent_timeout_does_not_treat_interrupt_request_as_child_quiescence():
    child = fake_child()
    run_started = threading.Event()
    run_release = threading.Event()
    interrupt_requested = threading.Event()
    close_attempted = threading.Event()

    def blocked_run(**_kwargs):
        run_started.set()
        run_release.wait(timeout=5)
        return {"final_response": "", "completed": False, "messages": []}

    def request_interrupt(_message=None):
        interrupt_requested.set()
        return True

    def close_child():
        close_attempted.set()

    child.run_conversation.side_effect = blocked_run
    child.hard_interrupt = request_interrupt
    child.close.side_effect = close_child
    parent_agent = parent()
    parent_agent._active_children = RemovalTrackingList(child)

    try:
        with patch("tools.delegate_tool._get_child_timeout", return_value=0.1):
            result = _run_single_child(
                task_index=0,
                goal="blocked after interrupt request",
                child=child,
                parent_agent=parent_agent,
            )

        assert run_started.is_set()
        assert interrupt_requested.is_set()
        assert result["status"] == "error"
        assert result["exit_reason"] == "cancellation_unconfirmed"
        assert result["error"] == "delegation cancellation could not be confirmed"
        assert child in parent_agent._active_children
        assert not close_attempted.is_set()

        run_release.set()
        assert close_attempted.wait(timeout=1)
        assert parent_agent._active_children.removed.wait(timeout=1)
        assert child not in parent_agent._active_children
    finally:
        run_release.set()


def test_parent_timeout_unregisters_relay_after_deferred_future_settles():
    child = fake_child()
    run_started = threading.Event()
    run_release = threading.Event()
    relay_unregistered = threading.Event()
    runtime = MagicMock()
    runtime.unregister_subagent.side_effect = lambda _payload: relay_unregistered.set()

    def blocked_run(**_kwargs):
        run_started.set()
        run_release.wait(timeout=5)
        return {"final_response": "", "completed": False, "messages": []}

    child.run_conversation.side_effect = blocked_run
    child.hard_interrupt.return_value = True
    parent_agent = parent()
    parent_agent._active_children.append(child)

    try:
        with (
            patch("tools.delegate_tool._get_child_timeout", return_value=0.1),
            patch("agent.relay_runtime.get_runtime", return_value=runtime),
            patch("agent.relay_runtime.current_profile_key", return_value="default"),
            patch(
                "agent.relay_runtime.SESSION_COORDINATOR.has_active_turn",
                return_value=False,
            ),
        ):
            result = _run_single_child(
                task_index=0,
                goal="deferred relay cleanup",
                child=child,
                parent_agent=parent_agent,
            )

            assert run_started.is_set()
            assert result["exit_reason"] == "cancellation_unconfirmed"
            assert child in parent_agent._active_children
            runtime.unregister_subagent.assert_not_called()

            run_release.set()
            assert relay_unregistered.wait(timeout=1)

        assert child not in parent_agent._active_children
        runtime.unregister_subagent.assert_called_once()
    finally:
        run_release.set()


def test_deferred_child_cleanup_is_serialized_against_parent_release_clients():
    child = fake_child()
    run_started = threading.Event()
    run_release = threading.Event()
    close_entered = threading.Event()
    allow_close = threading.Event()
    concurrent_close = threading.Event()
    parent_retry_entered = threading.Event()
    close_count = 0
    close_count_lock = threading.Lock()

    def blocked_run(**_kwargs):
        run_started.set()
        run_release.wait(timeout=5)
        return {"final_response": "", "completed": False, "messages": []}

    def blocked_close():
        nonlocal close_count
        with close_count_lock:
            close_count += 1
            if close_count > 1:
                concurrent_close.set()
        close_entered.set()
        allow_close.wait(timeout=5)

    child.run_conversation.side_effect = blocked_run
    child.hard_interrupt.return_value = True
    child.close.side_effect = blocked_close
    parent_agent = parent()
    parent_agent._active_children.append(child)
    parent_agent._codex_session_lock = None
    parent_agent.client = None
    # Current main factors child teardown into this method; bind the real
    # implementation on the MagicMock parent used by this focused test.
    parent_agent._close_active_children = AIAgent._close_active_children.__get__(
        parent_agent, AIAgent
    )
    release_thread = None

    try:
        with patch("tools.delegate_tool._get_child_timeout", return_value=0.1):
            result = _run_single_child(
                task_index=0,
                goal="serialize deferred cleanup",
                child=child,
                parent_agent=parent_agent,
            )

        assert run_started.is_set()
        assert result["exit_reason"] == "cancellation_unconfirmed"
        assert child in parent_agent._active_children
        retry_cleanup = getattr(child, "_delegate_release_ownership")

        def observed_parent_retry():
            parent_retry_entered.set()
            return retry_cleanup()

        child._delegate_release_ownership = observed_parent_retry
        run_release.set()
        assert close_entered.wait(timeout=1)

        release_thread = threading.Thread(
            target=AIAgent.release_clients,
            args=(parent_agent,),
        )
        release_thread.start()
        assert parent_retry_entered.wait(timeout=1)
        assert not concurrent_close.wait(timeout=0.5)
    finally:
        run_release.set()
        allow_close.set()
        if release_thread is not None:
            release_thread.join(timeout=2)

    assert release_thread is not None
    assert not release_thread.is_alive()
    assert close_count == 1
    assert child not in parent_agent._active_children


@pytest.mark.parametrize("dispatch_accepted", [True, False])
def test_background_cleanup_failure_retains_parent_retry_ownership(dispatch_accepted: bool):
    child = fake_child()
    parent_agent = parent()
    parent_agent._active_children.append(child)
    close_attempts = 0
    close_succeeds_on = 3 if dispatch_accepted else 2
    captured_runner: dict[str, Any] = {}

    def flaky_close():
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts < close_succeeds_on:
            raise RuntimeError("cleanup unavailable")

    def dispatch_background(**kwargs):
        if dispatch_accepted:
            captured_runner["runner"] = kwargs["runner"]
            return {"status": "dispatched", "delegation_id": "deleg-test"}
        return {"status": "rejected", "error": "capacity reached"}

    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "request_overrides": {},
        "max_output_tokens": None,
        "command": None,
        "args": [],
    }
    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config(enabled=False)),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value=credentials),
        patch(
            "tools.delegate_tool._build_child_preserving_parent_tools",
            return_value=child,
        ),
        patch("gateway.session_context.async_delivery_supported", return_value=True),
        patch(
            "tools.async_delegation.dispatch_async_delegation_batch",
            side_effect=dispatch_background,
        ),
    ):
        child.close.side_effect = flaky_close
        delegate_task(goal="retain cleanup ownership", background=True, parent_agent=parent_agent)
        if dispatch_accepted:
            captured_runner["runner"]()

    tracked = next(
        item
        for item in parent_agent._active_children
        if isinstance(item, RouteReceiptChild) and item.child is child
    )
    release_ownership = getattr(tracked, "_delegate_release_ownership")
    released = False
    for _ in range(close_succeeds_on):
        released = release_ownership()
        if released:
            break
    assert released is True
    assert tracked not in parent_agent._active_children
    assert close_attempts == close_succeeds_on


@pytest.mark.parametrize("dispatch_accepted", [True, False])
def test_background_timeout_defers_cleanup_until_child_execution_quiesces(
    dispatch_accepted: bool,
):
    child = fake_child()
    parent_agent = parent()
    parent_agent._active_children.append(child)
    run_started = threading.Event()
    release_run = threading.Event()
    close_called = threading.Event()
    premature_close = threading.Event()
    captured_runner: dict[str, Any] = {}

    def blocked_run(**_kwargs):
        run_started.set()
        release_run.wait(timeout=5)
        return {"final_response": "done", "completed": True, "messages": []}

    def observed_close():
        if not release_run.is_set():
            premature_close.set()
        close_called.set()

    def dispatch_background(**kwargs):
        if dispatch_accepted:
            captured_runner["runner"] = kwargs["runner"]
            return {"status": "dispatched", "delegation_id": "deleg-test"}
        return {"status": "rejected", "error": "capacity reached"}

    child.run_conversation.side_effect = blocked_run
    child.hard_interrupt.return_value = True
    child.close.side_effect = observed_close
    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "request_overrides": {},
        "max_output_tokens": None,
        "command": None,
        "args": [],
    }

    try:
        with (
            patch(
                "tools.delegate_tool._load_config",
                return_value=routing_config(enabled=False),
            ),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value=credentials,
            ),
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                return_value=child,
            ),
            patch("tools.delegate_tool._get_child_timeout", return_value=0.05),
            patch("gateway.session_context.async_delivery_supported", return_value=True),
            patch(
                "tools.async_delegation.dispatch_async_delegation_batch",
                side_effect=dispatch_background,
            ),
        ):
            delegate_task(
                goal="defer cleanup until quiescent",
                background=True,
                parent_agent=parent_agent,
            )
            if dispatch_accepted:
                captured_runner["runner"]()

        assert run_started.is_set()
        assert not premature_close.is_set()
        tracked = next(
            item
            for item in parent_agent._active_children
            if isinstance(item, RouteReceiptChild) and item.child is child
        )
    finally:
        release_run.set()

    assert close_called.wait(timeout=2)
    deadline = time.monotonic() + 2
    while tracked in parent_agent._active_children and time.monotonic() < deadline:
        time.sleep(0.01)
    assert tracked not in parent_agent._active_children


def test_child_ownership_is_retained_until_credential_release_succeeds():
    child = fake_child()
    pool = MagicMock()
    pool.acquire_lease.return_value = "lease-1"
    pool.current.return_value = None
    pool.release_lease.side_effect = [RuntimeError("release unavailable"), None]
    child._credential_pool = pool
    parent_agent = parent()
    parent_agent._active_children.append(child)

    result = _run_single_child(
        task_index=0,
        goal="retain complete cleanup graph",
        child=child,
        parent_agent=parent_agent,
    )

    assert result["status"] == "completed"
    assert child in parent_agent._active_children
    retry_cleanup = getattr(child, "_delegate_release_ownership", None)
    assert callable(retry_cleanup)

    assert retry_cleanup() is True
    assert child not in parent_agent._active_children
    assert pool.release_lease.call_count == 2


def test_child_ownership_is_retained_until_relay_unregistration_succeeds():
    child = fake_child()
    pool = MagicMock()
    pool.acquire_lease.return_value = "lease-1"
    pool.current.return_value = None
    child._credential_pool = pool
    parent_agent = parent()
    parent_agent._active_children.append(child)
    runtime = MagicMock()
    runtime.unregister_subagent.side_effect = [RuntimeError("relay unavailable"), None]

    with (
        patch("agent.relay_runtime.get_runtime", return_value=runtime),
        patch("agent.relay_runtime.current_profile_key", return_value="default"),
        patch(
            "agent.relay_runtime.SESSION_COORDINATOR.has_active_turn",
            return_value=False,
        ),
    ):
        result = _run_single_child(
            task_index=0,
            goal="retain relay cleanup ownership",
            child=child,
            parent_agent=parent_agent,
        )

        assert result["status"] == "completed"
        assert child in parent_agent._active_children
        retry_cleanup = getattr(child, "_delegate_release_ownership", None)
        assert callable(retry_cleanup)

        assert retry_cleanup() is True

    assert child not in parent_agent._active_children
    assert runtime.unregister_subagent.call_count == 2
    pool.release_lease.assert_called_once_with("lease-1")


def test_parent_timeout_interrupts_active_sol_fallback_without_late_receipt_mutation(
    tmp_path: Path,
):
    fallback = BlockingFallback()
    routed = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    routed.fallback_child = fallback

    with (
        patch("tools.delegate_tool._load_config", return_value=routing_config()),
        patch("tools.delegate_tool._active_profile_name", return_value="default"),
        patch("tools.delegate_tool._get_child_timeout", return_value=0.1),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
            "model": None, "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "request_overrides": {}, "max_output_tokens": None,
            "command": None, "args": [],
        }),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=fake_child()),
        patch("tools.delegate_tool._build_antigravity_delegate_child", return_value=routed),
    ):
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    try:
        assert result["results"][0]["status"] == "timeout"
        assert fallback.interrupted_after_start is True
        assert fallback.interrupted.wait(timeout=1)
        assert fallback.finished.wait(timeout=1)
        assert routed._execution_done.is_set()
        terminal = routed.store.get_attempt(routed.receipt_id)
        assert terminal["terminal_worker_status"] is None
        assert routed.store.get_attempt(routed.receipt_id) == terminal
    finally:
        fallback.release.set()


@pytest.mark.parametrize("method_name", ["interrupt", "hard_interrupt"])
def test_adapter_exposes_parent_interrupt_abi(method_name: str, tmp_path: Path):
    worker = FakeWorker(worker_result(ok=False))
    fallback = BlockingFallback()
    adapter = make_adapter(tmp_path, worker)
    adapter.fallback_child = fallback

    getattr(adapter, method_name)("parent stopped")

    assert worker.cancel_calls == 1
    assert fallback.interrupted.is_set()


def test_cancel_failure_is_surfaced_without_publishing_cancelled_receipt(
    tmp_path: Path,
):
    class FailingCloseWorker(FakeWorker):
        def close(self):
            raise RuntimeError("private worker shutdown detail")

    adapter = make_adapter(tmp_path, FailingCloseWorker(worker_result(ok=True)))
    adapter.prepare_receipt()
    before = adapter.store.get_attempt(adapter.receipt_id)

    with pytest.raises(RuntimeError, match="cancellation could not be confirmed"):
        adapter.cancel()

    assert adapter.store.get_attempt(adapter.receipt_id) == before


def test_cancel_receipt_failure_is_fixed_and_metadata_only(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=True)), fallback=False)
    adapter.prepare_receipt()
    adapter.store.complete_attempt = MagicMock(
        side_effect=RuntimeError("private /tmp/receipt.sqlite?token=secret")
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.close()

    assert str(exc_info.value) == "delegation cancellation could not be confirmed"
    assert "private" not in str(exc_info.value)
    assert adapter._closed is False


def test_cancel_from_execution_thread_does_not_wait_for_itself(tmp_path: Path):
    class CancellingWorker(FakeWorker):
        adapter: Any = None
        cancel_error = None

        def run(self, **_kwargs):
            try:
                self.adapter.cancel()
            except Exception as exc:
                self.cancel_error = exc
            return worker_result(ok=True)

    worker = CancellingWorker(worker_result(ok=True))
    adapter = make_adapter(tmp_path, worker, fallback=False)
    worker.adapter = adapter
    adapter._execution_done.wait = MagicMock(
        side_effect=AssertionError("execution thread attempted to wait for itself")
    )

    result = adapter.run_conversation("cancel from callback")

    assert worker.cancel_error is None
    adapter._execution_done.wait.assert_not_called()
    assert result["gemini_error_code"] == "cancelled"


def test_failed_fallback_interrupt_uses_close_before_confirming_cancel(
    tmp_path: Path,
):
    class InterruptFailureFallback:
        def __init__(self):
            self.close_calls = 0

        def hard_interrupt(self, _message=None):
            raise RuntimeError("private interrupt detail")

        def close(self):
            self.close_calls += 1

    fallback = InterruptFailureFallback()
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    adapter.fallback_child = fallback
    adapter.prepare_receipt()

    adapter.cancel()

    assert fallback.close_calls == 1
    assert adapter.store.get_attempt(adapter.receipt_id)["worker_status"] == "cancelled"


def test_close_interrupts_active_fallback_before_returning(tmp_path: Path):
    fallback = BlockingFallback()
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    adapter.fallback_child = fallback
    result_holder: dict[str, dict] = {}
    run_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", adapter.run_conversation("prompt")
        )
    )
    run_thread.start()
    assert fallback.started.wait(timeout=1)
    terminal_before_close = adapter.store.get_attempt(adapter.receipt_id)

    adapter.close()
    run_thread.join(timeout=3)

    assert fallback.interrupted_after_start is True
    assert fallback.finished.is_set()
    assert fallback.close_calls == 1
    assert not run_thread.is_alive()
    assert result_holder["result"]["gemini_error_code"] == "cancelled"
    assert adapter.store.get_attempt(adapter.receipt_id) == terminal_before_close


def test_close_can_retry_after_unconfirmed_cancellation(tmp_path: Path):
    class RetryableCloseWorker(FakeWorker):
        def __init__(self):
            super().__init__(worker_result(ok=True))
            self.close_attempts = 0

        def close(self):
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("first close failed")
            super().close()

    worker = RetryableCloseWorker()
    adapter = make_adapter(tmp_path, worker)

    with pytest.raises(RuntimeError, match="cancellation could not be confirmed"):
        adapter.close()
    adapter.close()

    assert worker.close_attempts == 2
    assert adapter._closed is True


def test_cancel_waits_for_active_fallback_to_finish(tmp_path: Path):
    class QuiescenceFallback(BlockingFallback):
        def __init__(self):
            super().__init__()
            self.cleanup_release = threading.Event()

        def run_conversation(self, **_kwargs):
            self.started.set()
            self.release.wait(timeout=5)
            self.cleanup_release.wait(timeout=5)
            self.finished.set()
            return {"final_response": "", "completed": False, "messages": []}

    fallback = QuiescenceFallback()
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    adapter.fallback_child = fallback
    execution_wait_entered = threading.Event()
    original_execution_wait = adapter._execution_done.wait

    def observed_execution_wait(timeout=None):
        execution_wait_entered.set()
        return original_execution_wait(timeout)

    adapter._execution_done.wait = observed_execution_wait
    run_thread = threading.Thread(target=lambda: adapter.run_conversation("prompt"))
    run_thread.start()
    assert fallback.started.wait(timeout=1)
    cancel_returned = threading.Event()
    cancel_thread = threading.Thread(
        target=lambda: (adapter.cancel(), cancel_returned.set())
    )
    cancel_thread.start()
    assert fallback.interrupted.wait(timeout=1)

    assert execution_wait_entered.wait(timeout=1)
    assert not cancel_returned.is_set()
    fallback.cleanup_release.set()
    assert cancel_returned.wait(timeout=1)
    cancel_thread.join(timeout=1)
    run_thread.join(timeout=1)
    assert fallback.finished.is_set()


def test_adapter_rejects_overlapping_run_before_replacing_execution_ownership(
    tmp_path: Path,
):
    class SingleRunWorker(FakeWorker):
        def __init__(self):
            super().__init__(worker_result(ok=True))
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def run(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                self.release.wait(timeout=5)
                return worker_result(ok=True)
            return AntigravityResult(
                status="failed",
                response=None,
                conversation_id=None,
                usage={},
                raw_envelope=None,
                exit_code=None,
                duration_ms=0,
                error_code="already_running",
                error_message="worker already has an active run",
            )

    worker = SingleRunWorker()
    adapter = make_adapter(tmp_path, worker)
    first_result: dict[str, dict] = {}
    first_thread = threading.Thread(
        target=lambda: first_result.setdefault(
            "result", adapter.run_conversation("first")
        )
    )
    first_thread.start()
    assert worker.started.wait(timeout=1)

    try:
        second = adapter.run_conversation("second")

        assert second["completed"] is False
        assert second["gemini_error_code"] == "already_running"
        assert worker.calls == 1
        assert not adapter._execution_done.is_set()
    finally:
        worker.release.set()
        first_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert first_result["result"]["completed"] is True
    assert adapter._execution_done.is_set()


def test_cancel_between_receipt_prep_and_worker_admission_prevents_worker_run(
    tmp_path: Path,
):
    worker = FakeWorker(worker_result(ok=True))
    adapter = make_adapter(tmp_path, worker)
    prepared = threading.Event()
    release_prepare = threading.Event()
    original_prepare = adapter.prepare_receipt
    result_holder: dict[str, dict] = {}

    def blocked_prepare():
        receipt_id = original_prepare()
        prepared.set()
        release_prepare.wait(timeout=3)
        return receipt_id

    adapter.prepare_receipt = blocked_prepare
    run_thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", adapter.run_conversation("prompt"))
    )
    run_thread.start()
    assert prepared.wait(timeout=1)
    cancel_thread = threading.Thread(target=adapter.hard_interrupt)
    cancel_thread.start()
    assert adapter._cancel_event.wait(timeout=1)
    release_prepare.set()
    run_thread.join(timeout=3)
    cancel_thread.join(timeout=3)

    assert not run_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert worker.run_calls == 0
    assert result_holder["result"]["gemini_error_code"] == "cancelled"


def test_cancel_during_fallback_metadata_prevents_fallback_admission(tmp_path: Path):
    fallback = AdmissionBlockedFallback()
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    adapter.fallback_child = fallback
    result_holder: dict[str, dict] = {}
    run_thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", adapter.run_conversation("prompt"))
    )
    run_thread.start()
    assert fallback.metadata_started.wait(timeout=1)

    adapter.interrupt("parent stopped")
    fallback.release_metadata.set()
    run_thread.join(timeout=3)

    assert not run_thread.is_alive()
    assert fallback.run_calls == 0
    assert result_holder["result"]["gemini_error_code"] == "cancelled"


def test_cancel_after_fallback_admission_prevents_codex_session_start(tmp_path: Path):
    fallback = PostAdmissionCodexFallback()
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    adapter.fallback_child = fallback
    result_holder: dict[str, dict] = {}
    run_thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", adapter.run_conversation("prompt"))
    )
    run_thread.start()
    assert fallback.admission_entered.wait(timeout=2)

    adapter.hard_interrupt("parent timed out")
    assert fallback._interrupt_requested is True
    fallback.admission_release.set()
    run_thread.join(timeout=3)

    try:
        assert not run_thread.is_alive()
        assert getattr(fallback, "_codex_session", None) is None
        assert result_holder["result"]["gemini_error_code"] == "cancelled"
    finally:
        fallback.admission_release.set()
        adapter.close()


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
                tasks=[{"goal": "x", "output_contract": "xml"}],
                parent_agent=parent(),
            )
        )
    assert "output_contract" in result["error"]
    builder.assert_not_called()


class FakeWorker:
    def __init__(self, result: AntigravityResult, *, invoke_started: bool = True):
        self.result = result
        self.invoke_started = invoke_started
        self.closed = False
        self.cancel_calls = 0
        self.run_calls = 0

    def run(self, *, goal, context, output_schema, on_process_started=None):
        self.run_calls += 1
        if self.invoke_started and on_process_started:
            on_process_started()
        return self.result

    def cancel(self):
        self.cancel_calls += 1

    def close(self):
        self.closed = True
        self.cancel()


class PostAdmissionCodexFallback(AIAgent):
    def __init__(self):
        super().__init__(
            api_key="stub",
            base_url="https://stub.invalid",
            provider="openai",
            api_mode="codex_app_server",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        self.admission_entered = threading.Event()
        self.admission_release = threading.Event()

    def __getattribute__(self, name):
        if name == "run_conversation":
            entered = object.__getattribute__(self, "admission_entered")
            release = object.__getattribute__(self, "admission_release")
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("fallback admission was not released")
        return super().__getattribute__(name)


class BlockingWorker(FakeWorker):
    def __init__(self):
        super().__init__(worker_result(ok=False))
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event()

    def run(self, *, goal, context, output_schema, on_process_started=None):
        if on_process_started:
            on_process_started()
        self.release.wait(timeout=5)
        self.finished.set()
        return AntigravityResult(
            status="failed",
            response=None,
            conversation_id=None,
            usage={},
            raw_envelope=None,
            exit_code=None,
            duration_ms=1,
            error_code="cancelled" if self.cancelled.is_set() else "worker_exception",
            error_message="cancelled" if self.cancelled.is_set() else "released",
        )

    def cancel(self):
        self.cancelled.set()
        self.release.set()


class BlockingFallback:
    def __init__(self):
        self.session_id = "sol-child"
        self.provider = "openai-codex"
        self.model = "gpt-5.6-sol"
        self._delegate_role = "leaf"
        self._delegate_saved_tool_names = []
        self.tool_progress_callback = None
        self._credential_pool = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.started = threading.Event()
        self.interrupted = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event()
        self.interrupted_after_start = False
        self.close_calls = 0

    def run_conversation(self, **_kwargs):
        self.started.set()
        self.release.wait(timeout=5)
        self.finished.set()
        return {"final_response": "", "completed": False, "messages": []}

    def hard_interrupt(self, _message=None):
        self.interrupted_after_start = self.started.is_set()
        self.interrupted.set()
        self.release.set()

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 1, "current_tool": None}

    def close(self):
        self.close_calls += 1
        self.release.set()


class AdmissionBlockedFallback:
    def __init__(self):
        self.session_id = "sol-child"
        self.model = "gpt-5.6-sol"
        self._delegate_role = "leaf"
        self._delegate_saved_tool_names = []
        self.tool_progress_callback = None
        self._credential_pool = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.metadata_started = threading.Event()
        self.release_metadata = threading.Event()
        self.interrupted = threading.Event()
        self.run_calls = 0

    @property
    def provider(self):
        self.metadata_started.set()
        self.release_metadata.wait(timeout=3)
        return "openai-codex"

    def run_conversation(self, **_kwargs):
        self.run_calls += 1
        return {"final_response": "late", "completed": True, "messages": []}

    def hard_interrupt(self, _message=None):
        self.interrupted.set()

    def close(self):
        self.release_metadata.set()


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


def test_builder_prepares_receipt_before_lifecycle_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    worker = FakeWorker(worker_result(ok=True))

    with patch("agent.antigravity_worker.AntigravityWorker", return_value=worker):
        adapter = _build_antigravity_delegate_child(
            task_index=0,
            task={"goal": "summarize", "data_classification": "standard"},
            fallback_child=fake_child("fallback answer"),
            routing_cfg=routing_config()["gemini_routing"],
            route_reason="eligible output-only leaf delegation",
            parent_agent=parent(),
        )

    assert adapter.receipt_id.startswith("grt_")
    assert adapter._route_metadata["route_receipt_id"] == adapter.receipt_id
    assert adapter.store.get_attempt(adapter.receipt_id)["process_started_at_utc"] is None


def test_adapter_returns_gemini_output_and_records_two_phase_receipt(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=True)))

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["final_response"] == "gemini answer"
    assert result["completed"] is True
    assert result["worker_route"] == "gemini"
    assert result["worker_provider"] == "antigravity-subscription"
    assert result["worker_model_requested"] == "gemini-3.8-flash-low"
    assert result["route_receipt_id"] == adapter.receipt_id
    assert result["fallback_used"] is False
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
    assert result["worker_route"] == "sol"
    assert result["worker_provider"] == "openai-codex"
    assert result["worker_model_requested"] == "gpt-5.6-sol"
    assert result["route_receipt_id"] == adapter.receipt_id
    assert result["fallback_used"] is True
    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["worker_status"] == "failed"
    assert row["fallback_used"] == 1
    assert row["error_code"] == "nonzero_exit"
    assert row["terminal_worker_route"] == "sol"
    assert row["terminal_provider"] == "openai-codex"
    assert row["terminal_model"] == "gpt-5.6-sol"
    assert row["terminal_worker_status"] == "completed"
    assert row["terminal_response_text"] is None
    assert row["terminal_response_sha256"] == hashlib.sha256(
        b"fallback answer"
    ).hexdigest()
    assert row["terminal_response_bytes"] == len(b"fallback answer")
    assert row["terminal_error_code"] is None


def test_adapter_bounds_persisted_fallback_response_but_hashes_complete_value(tmp_path: Path):
    complete_response = "🔥" * 50_000
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    assert adapter.fallback_child is not None
    adapter.fallback_child.run_conversation.return_value["final_response"] = complete_response

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["final_response"] == complete_response
    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["terminal_response_text"] is None
    assert row["terminal_response_sha256"] == hashlib.sha256(
        complete_response.encode("utf-8")
    ).hexdigest()
    assert row["terminal_response_bytes"] == len(complete_response.encode("utf-8"))


def test_adapter_persists_complete_oversized_gemini_output_digest(tmp_path: Path):
    complete_output = "🔥" * 50_000
    encoded = complete_output.encode("utf-8")
    oversized = AntigravityResult(
        status="failed",
        response=None,
        conversation_id=None,
        usage={},
        raw_envelope=None,
        exit_code=0,
        duration_ms=1,
        error_code="output_too_large",
        error_message="Antigravity output exceeds byte limit",
        output_excerpt=encoded[:32_768].decode("utf-8", errors="ignore"),
        output_sha256=hashlib.sha256(encoded).hexdigest(),
        output_bytes=len(encoded),
    )
    adapter = make_adapter(tmp_path, FakeWorker(oversized))

    adapter.run_conversation("summarize", task_id="child-task")

    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["response_text"] is None
    assert row["response_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert row["response_bytes"] == len(encoded)


def test_sol_fallback_reports_and_records_post_run_runtime_identity(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    fallback_child = adapter.fallback_child
    assert fallback_child is not None

    def activate_internal_fallback(**_kwargs):
        fallback_child.provider = "anthropic"
        fallback_child.model = "claude-sonnet-4-6"
        return {
            "final_response": "fallback answer",
            "completed": True,
            "api_calls": 2,
            "messages": [],
        }

    fallback_child.run_conversation.side_effect = activate_internal_fallback

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["worker_provider"] == "anthropic"
    assert result["worker_model_requested"] == "claude-sonnet-4-6"
    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["terminal_worker_route"] == "sol"
    assert row["terminal_provider"] == "anthropic"
    assert row["terminal_model"] == "claude-sonnet-4-6"
    assert row["terminal_worker_status"] == "completed"


def test_adapter_returns_structured_metadata_when_sol_fallback_raises(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    assert adapter.fallback_child is not None
    adapter.fallback_child.run_conversation.side_effect = RuntimeError(
        "private fallback details"
    )

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["completed"] is False
    assert result["error"] == "Sol fallback raised RuntimeError"
    assert "private fallback details" not in result["error"]
    assert result["route"] == "gemini_then_sol"
    assert result["worker_route"] == "sol"
    assert result["worker_provider"] == "openai-codex"
    assert result["worker_model_requested"] == "gpt-5.6-sol"
    assert result["route_receipt_id"] == adapter.receipt_id
    assert result["fallback_used"] is True
    assert result["gemini_error_code"] == "nonzero_exit"
    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["terminal_worker_status"] == "failed"
    assert row["terminal_response_text"] is None
    assert row["terminal_response_sha256"] is None
    assert row["terminal_error_code"] == "sol_fallback_failed"


def test_legacy_completion_failure_without_fallback_preserves_gemini_result(
    tmp_path: Path,
):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=True)), fallback=False)
    adapter.store.complete_attempt = MagicMock(side_effect=sqlite3.OperationalError("locked"))

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["completed"] is True
    assert result["route"] == "gemini"
    assert result["worker_route"] == "gemini"
    assert result["worker_provider"] == "antigravity-subscription"
    assert result["worker_model_requested"] == "gemini-3.8-flash-low"
    assert result["fallback_used"] is False
    assert adapter.route_store.get_attempt(adapter.receipt_id)["status"] == "completed"


def test_legacy_completion_failure_with_unused_fallback_preserves_gemini_result(
    tmp_path: Path,
):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=True)))
    adapter.store.complete_attempt = MagicMock(side_effect=sqlite3.OperationalError("locked"))

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["completed"] is True
    assert result["route"] == "gemini"
    assert result["worker_route"] == "gemini"
    assert result["worker_provider"] == "antigravity-subscription"
    assert result["worker_model_requested"] == "gemini-3.8-flash-low"
    assert result["fallback_used"] is False
    assert adapter.route_store.get_attempt(adapter.receipt_id)["status"] == "completed"


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
