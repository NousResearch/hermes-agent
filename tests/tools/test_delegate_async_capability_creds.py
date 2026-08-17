"""Async batch dispatch must not crash when ALL tasks use named capabilities.

Regression: fix2 correctly skipped default-chain credential resolution when
every task carries a named capability (creds = None), but the background
batch dispatch still read creds["model"] for completion-block metadata ->
TypeError: 'NoneType' object is not subscriptable. Symptom seen 2026-08-17:
every capability delegation from an async-capable session (WhatsApp chat)
failed at dispatch; sync-path children were unaffected.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task


def _parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.model = "claude-fable-5"
    parent.provider = "anthropic"
    parent.api_key = "sk-parent"
    parent.base_url = "https://api.anthropic.com"
    parent.api_mode = "anthropic_messages"
    parent.enabled_toolsets = ["terminal"]
    parent.platform = "whatsapp"
    parent._interrupt_requested = False
    parent._delegate_spinner = None
    return parent


def test_async_batch_with_capability_only_tasks_does_not_crash():
    parent = _parent()
    resolved = {
        "provider": "openai-codex",
        "model": "openai-codex/gpt-5.6-sol",
        "context": "You are the debugger.",
        "fallback_models": [],
        "toolsets": ["terminal"],
        "workload": "coding",
        "trusted_toolset_elevation": False,
    }
    creds = {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "api_key": "sk-codex-test",
        "base_url": "https://codex.example",
        "api_mode": "openai_chat",
    }
    child = MagicMock()
    child._delegate_role = "leaf"
    child._delegate_saved_tool_names = []
    child.tool_progress_callback = None
    child.model = "gpt-5.6-sol"

    captured = {}

    def _dispatch(**kwargs):
        captured.update(kwargs)
        return {
            "status": "dispatched",
            "delegation_id": "test-id",
            "message": "dispatched",
        }

    def _creds(cfg, _parent_agent):
        if cfg.get("provider") == "openai-codex" and "capability" not in cfg:
            raise AssertionError("default-chain creds must not be resolved")
        return creds

    with patch(
        "tools.delegate_tool._load_config",
        return_value={
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "max_iterations": 5,
        },
    ), patch(
        "tools.delegate_tool._resolve_plugin_capability", return_value=resolved
    ), patch(
        "tools.delegate_tool._resolve_delegation_credentials",
        side_effect=lambda cfg, p: creds,
    ), patch(
        "tools.delegate_tool._build_child_agent", return_value=child
    ), patch(
        "tools.delegate_tool._get_max_concurrent_children", return_value=3
    ), patch(
        "tools.delegate_tool._get_max_spawn_depth", return_value=2
    ), patch(
        "tools.delegate_tool.is_spawn_paused", return_value=False
    ), patch(
        "tools.delegation_live_log.create_live_transcripts",
        return_value=("id", [], []),
    ), patch(
        "tools.delegation_live_log.update_manifest_statuses",
    ), patch(
        "model_tools._last_resolved_tool_names",
        [],
    ), patch(
        "tools.async_delegation.dispatch_async_delegation_batch",
        side_effect=_dispatch,
    ):
        out = json.loads(
            delegate_task(
                goal="resume the fixer",
                capability="debugger",
                background=True,
                parent_agent=parent,
            )
        )

    assert "error" not in out, out.get("error")
    # Metadata model must be populated from the resolved child, not a
    # None default-chain creds dict.
    assert captured.get("model") == "gpt-5.6-sol"
