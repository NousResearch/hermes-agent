#!/usr/bin/env python3
"""Per-task reasoning_effort / service_tier on delegate_task.

A task's thinking level and speed tier apply to that child only, are validated
before any child is built, and a tier the child's route can't serve fails the
spawn instead of silently running at another tier.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


CODEX_CREDS = {
    "provider": "openai-codex",
    "model": "gpt-6-astra",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_key": "codex-key",
    "api_mode": "codex_responses",
    "request_overrides": {},
}


def _make_mock_parent(request_overrides=None):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.request_overrides = request_overrides or {}
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _resolve(cfg, _parent_agent):
    if str((cfg or {}).get("provider") or "").strip() == "openai-codex":
        return dict(CODEX_CREDS, model=(cfg or {}).get("model") or CODEX_CREDS["model"])
    return {"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None,
            "request_overrides": None}


def _spawn(tasks, *, cfg=None, parent=None):
    parent = parent or _make_mock_parent()
    child = MagicMock()
    with (
        patch("tools.delegate_tool._load_config", return_value=cfg or {}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_resolve),
        patch("run_agent.AIAgent", return_value=child) as agent,
        patch("tools.delegate_tool._run_batch", return_value='{"status":"built"}'),
    ):
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))
    return result, [call.kwargs for call in agent.call_args_list]


def test_reviewer_pin_sets_route_thinking_and_fast_tier():
    """'Astra codex medium fast': one child on that route; its sibling keeps the config/parent defaults."""
    result, (reviewer, sibling) = _spawn(
        [
            {"goal": "Review the PR diff for regressions", "provider": "openai-codex", "model": "gpt-6-astra",
             "reasoning_effort": "medium", "service_tier": "fast"},
            {"goal": "Summarize the PR discussion thread"},
        ],
        cfg={"reasoning_effort": "low"},
    )

    assert result == {"status": "built"}
    assert (reviewer["provider"], reviewer["model"]) == ("openai-codex", "gpt-6-astra")
    assert reviewer["reasoning_config"] == {"enabled": True, "effort": "medium"}
    assert reviewer["request_overrides"]["service_tier"] == "priority"
    assert reviewer["service_tier"] == "priority"
    assert sibling["reasoning_config"] == {"enabled": True, "effort": "low"}
    assert "service_tier" not in sibling["request_overrides"]
    assert "service_tier" not in sibling


def test_reasoning_none_disables_thinking_for_that_child():
    _, (child,) = _spawn([{"goal": "Quick lookup task here", "reasoning_effort": "none"}])
    assert child["reasoning_config"] == {"enabled": False}


def test_normal_tier_drops_inherited_fast_overrides():
    parent = _make_mock_parent(request_overrides={"service_tier": "priority", "extra_body": {"keep": 1}})
    _, (child,) = _spawn([{"goal": "Cheap background task", "service_tier": "normal"}], parent=parent)
    assert child["request_overrides"] == {"extra_body": {"keep": 1}}
    assert parent.request_overrides["service_tier"] == "priority"  # parent untouched


@pytest.mark.parametrize("field,value", [
    ("reasoning_effort", "hgih"),
    ("service_tier", "turbo"),
    ("service_tier", "auto"),  # bounded windows are per user turn, meaningless for a child
])
def test_unknown_words_fail_before_any_child_is_built(field, value):
    result, built = _spawn([
        {"goal": "First task in the batch", field: value},
        {"goal": "Second task in the batch"},
    ])
    assert "error" in result and value in result["error"]
    assert built == []


def test_tier_the_route_cannot_serve_fails_the_spawn():
    """A local route has no fast tier: refuse rather than silently run at the normal tier."""
    parent = _make_mock_parent()
    parent.provider, parent.model, parent.base_url = "custom", "qwen3-local", "http://127.0.0.1:8080/v1"
    result, built = _spawn([{"goal": "Review this change set", "service_tier": "fast"}], parent=parent)
    assert "not available" in result["error"]
    assert built == []


def test_schema_advertises_task_tuning_fields():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert {"none", "medium", "xhigh"} <= set(props["reasoning_effort"]["enum"])
    assert props["service_tier"]["enum"] == ["normal", "fast", "ultrafast"]
