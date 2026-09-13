#!/usr/bin/env python3
"""Per-task provider/model pinning on delegate_task (#107937).

A mixed batch must construct child 0 on the pinned route and child 1 on the
inherited batch/parent route. Blank pins fail-open; unknown providers fail-closed.
"""

import json
import threading
from unittest.mock import MagicMock, patch

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


PARENT_CREDS = {
    "provider": "nous",
    "model": "muse-glimmer",
    "base_url": "http://parent-gpu/v1",
    "api_key": "parent-key",
    "api_mode": "chat_completions",
}

PIN_CREDS = {
    "provider": "lmstudio-x121",
    "model": "ling-3.0-tiny",
    "base_url": "http://lmstudio-x121/v1",
    "api_key": "lm-key",
    "api_mode": "chat_completions",
}

UNRESOLVABLE_PIN_ERROR = "Unknown provider 'not-a-real-provider'"


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
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
    return parent


def _fake_resolve(cfg, parent_agent):
    provider = str((cfg or {}).get("provider") or "").strip()
    if provider == "not-a-real-provider":
        raise ValueError(UNRESOLVABLE_PIN_ERROR)
    if provider == "lmstudio-x121":
        return dict(PIN_CREDS)
    return dict(PARENT_CREDS)


def _ok_child():
    child = MagicMock()
    child.run_conversation.return_value = {
        "final_response": "ok",
        "completed": True,
        "api_calls": 1,
    }
    return child


def test_mixed_batch_pins_only_task_zero():
    """Detection: task 0 pin must not reuse batch creds; task 1 inherits."""
    parent = _make_mock_parent()
    resolve_calls = []

    def tracking_resolve(cfg, parent_agent):
        resolve_calls.append(dict(cfg) if isinstance(cfg, dict) else cfg)
        return _fake_resolve(cfg, parent_agent)

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=tracking_resolve),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        MockAgent.return_value = _ok_child()
        delegate_task(
            tasks=[
                {
                    "goal": "Research topic A with enough length",
                    "provider": "lmstudio-x121",
                    "model": "ling-3.0-tiny",
                },
                {"goal": "Research topic B with enough length"},
            ],
            parent_agent=parent,
        )

    assert MockAgent.call_count == 2
    child0 = MockAgent.call_args_list[0].kwargs
    child1 = MockAgent.call_args_list[1].kwargs
    assert child0["provider"] == "lmstudio-x121"
    assert child0["model"] == "ling-3.0-tiny"
    assert child0["base_url"] == "http://lmstudio-x121/v1"
    assert child1["provider"] == "nous"
    assert child1["model"] == "muse-glimmer"
    assert child1["base_url"] == "http://parent-gpu/v1"
    assert any(
        str((cfg or {}).get("provider") or "").strip() == "lmstudio-x121"
        for cfg in resolve_calls
    )


def test_blank_pin_inherits_batch_creds():
    """Fail-open: empty / whitespace-only pins skip the per-task overlay."""
    parent = _make_mock_parent()
    resolve_calls = []

    def tracking_resolve(cfg, parent_agent):
        resolve_calls.append(dict(cfg) if isinstance(cfg, dict) else cfg)
        return _fake_resolve(cfg, parent_agent)

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=tracking_resolve),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        MockAgent.return_value = _ok_child()
        delegate_task(
            tasks=[
                {
                    "goal": "Research topic A with enough length",
                    "provider": "",
                    "model": "  ",
                },
            ],
            parent_agent=parent,
        )

    assert MockAgent.call_count == 1
    kwargs = MockAgent.call_args.kwargs
    assert kwargs["provider"] == "nous"
    assert kwargs["model"] == "muse-glimmer"
    assert kwargs["base_url"] == "http://parent-gpu/v1"
    assert len(resolve_calls) == 1


def test_unresolvable_pin_fails_closed():
    """A bad per-task provider must tool_error; no silent inherit."""
    parent = _make_mock_parent()

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_fake_resolve),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        MockAgent.return_value = _ok_child()
        raw = delegate_task(
            tasks=[
                {
                    "goal": "Research topic A with enough length",
                    "provider": "not-a-real-provider",
                },
            ],
            parent_agent=parent,
        )

    payload = json.loads(raw)
    assert "error" in payload
    assert UNRESOLVABLE_PIN_ERROR in payload["error"]
    MockAgent.assert_not_called()


def test_schema_advertises_provider_and_model():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "provider" in props
    assert "model" in props
    assert props["provider"]["type"] == "string"
    assert props["model"]["type"] == "string"
