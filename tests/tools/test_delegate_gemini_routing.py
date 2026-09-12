from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.antigravity_delegate import AntigravityDelegateChild
from agent.antigravity_worker import AntigravityResult
from agent.gemini_route_receipts import GeminiReceiptStore
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_antigravity_delegate_child,
    _merge_child_route_metadata,
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


def test_disabled_config_preserves_existing_sol_child_path():
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

    assert result["results"][0]["summary"] == "sol answer"
    assert not {
        "route",
        "route_reason",
        "worker_route",
        "worker_provider",
        "worker_model_requested",
        "route_receipt_id",
        "fallback_used",
    } & result["results"][0].keys()
    assert not {
        "worker_route",
        "worker_provider",
        "worker_model_requested",
        "route_receipt_id",
        "fallback_used",
    } & stops[0].keys()
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
    assert kwargs["fallback_child"] is sol
    assert kwargs["route_reason"] == "eligible output-only leaf delegation"


def test_receipt_initialization_failure_runs_prebuilt_sol_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sol = fake_child("fallback answer")
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
        result = json.loads(delegate_task(goal="summarize", parent_agent=parent()))

    public = result["results"][0]
    assert public["summary"] == "fallback answer"
    assert public["route"] == "sol_after_receipt_error"
    assert public["worker_route"] == "sol"
    assert public["fallback_used"] is True
    assert public["gemini_error_code"] == "receipt_initialization_failed"
    assert "route_receipt_id" in public
    assert public["route_receipt_id"] is None
    assert stops[0]["worker_route"] == "sol"
    assert stops[0]["fallback_used"] is True
    assert stops[0]["gemini_error_code"] == "receipt_initialization_failed"
    assert "route_receipt_id" in stops[0]
    assert stops[0]["route_receipt_id"] is None


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
    assert result["results"][0]["route_receipt_id"] is None
    assert "route_receipt_id" in stops[0]
    assert stops[0]["route_receipt_id"] is None
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
                tasks=[{"goal": "first"}, {"goal": "second"}],
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
    assert row["terminal_response_text"] == "fallback answer"
    assert row["terminal_response_sha256"] == hashlib.sha256(
        b"fallback answer"
    ).hexdigest()
    assert row["terminal_error_code"] is None


def test_adapter_bounds_persisted_fallback_response_but_hashes_complete_value(tmp_path: Path):
    complete_response = "🔥" * 50_000
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=False)))
    assert adapter.fallback_child is not None
    adapter.fallback_child.run_conversation.return_value["final_response"] = complete_response

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["final_response"] == complete_response
    row = adapter.store.get_attempt(adapter.receipt_id)
    assert row["terminal_response_text"] != complete_response
    assert len(row["terminal_response_text"].encode("utf-8")) <= 32_768
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
    assert row["response_text"] == oversized.output_excerpt
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


def test_receipt_completion_failure_without_fallback_preserves_gemini_worker_truth(
    tmp_path: Path,
):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=True)), fallback=False)
    adapter.store.complete_attempt = MagicMock(side_effect=sqlite3.OperationalError("locked"))

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["completed"] is False
    assert result["route"] == "sol_after_receipt_error"
    assert result["worker_route"] == "gemini"
    assert result["worker_provider"] == "antigravity-subscription"
    assert result["worker_model_requested"] == "gemini-3.8-flash-low"
    assert result["fallback_used"] is False


def test_receipt_completion_failure_with_fallback_reports_sol_worker(tmp_path: Path):
    adapter = make_adapter(tmp_path, FakeWorker(worker_result(ok=True)))
    adapter.store.complete_attempt = MagicMock(side_effect=sqlite3.OperationalError("locked"))

    result = adapter.run_conversation("summarize", task_id="child-task")

    assert result["route"] == "sol_after_receipt_error"
    assert result["worker_route"] == "sol"
    assert result["worker_provider"] == "openai-codex"
    assert result["worker_model_requested"] == "gpt-5.6-sol"
    assert result["fallback_used"] is True


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
