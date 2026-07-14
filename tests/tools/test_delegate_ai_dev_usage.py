from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _classify_ai_dev_usage_route,
    delegate_task,
)


def make_parent() -> MagicMock:
    parent = MagicMock()
    parent.base_url = "https://chatgpt.com/backend-api/codex"
    parent.api_key = "oauth-token-placeholder"
    parent.provider = "openai-codex"
    parent.api_mode = "codex_responses"
    parent.model = "gpt-5.6-sol"
    parent.platform = "cli"
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


def test_openai_codex_delegate_is_subscription_delegate_route() -> None:
    route = _classify_ai_dev_usage_route(
        {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "oauth-token-placeholder",
        },
        make_parent(),
    )

    assert route == ("openai-codex", "gpt-5.6-sol", "delegate")


def test_delegate_schema_exposes_explicit_variable_cost_override() -> None:
    prop = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["allow_variable_cost"]
    assert prop["type"] == "boolean"


@patch("tools.delegate_tool._run_single_child")
@patch("tools.delegate_tool._record_ai_dev_usage", return_value=42)
@patch("tools.delegate_tool._resolve_delegation_credentials")
def test_api_key_delegate_fails_closed_before_child_spawn(
    resolve_credentials: MagicMock,
    record_usage: MagicMock,
    run_child: MagicMock,
) -> None:
    resolve_credentials.return_value = {
        "provider": "openai-api",
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-not-a-real-key",
        "api_mode": "chat_completions",
    }

    result = json.loads(
        delegate_task(goal="Review current diff", parent_agent=make_parent())
    )

    assert "error" in result
    assert "variable-cost" in result["error"].lower()
    record_usage.assert_called_once()
    run_child.assert_not_called()
