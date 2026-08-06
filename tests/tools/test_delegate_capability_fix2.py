"""Fix2: capability resolution must not look like Codex auth failure."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

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
    parent.platform = "cli"
    parent._interrupt_requested = False
    parent._delegate_spinner = None
    return parent


def test_capability_hook_error_reaches_tool_error():
    """Plugin CapabilityResolutionError text must reach the parent tool result."""
    parent = _parent()

    def _boom(hook_name, **_kwargs):
        raise ValueError("Capability references missing skill 'python-debugpy'")

    with patch(
        "tools.delegate_tool._load_config",
        return_value={
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "max_iterations": 5,
        },
    ), patch(
        "tools.delegate_tool._resolve_delegation_credentials"
    ) as mock_creds, patch(
        "hermes_cli.plugins.invoke_hook",
        side_effect=_boom,
    ):
        out = json.loads(
            delegate_task(
                goal="debug me",
                capability="debugger",
                parent_agent=parent,
            )
        )
    assert "error" in out
    assert "python-debugpy" in out["error"]
    assert "Codex" not in out["error"]
    assert "not resolved by an enabled plugin" not in out["error"]
    mock_creds.assert_not_called()


def test_capability_skips_global_codex_preflight():
    """Named capability must not require global openai-codex credentials."""
    parent = _parent()
    kimi_creds = {
        "provider": "kimi-coding",
        "model": "k3",
        "api_key": "sk-kimi-test",
        "base_url": "https://api.kimi.com/coding",
        "api_mode": "anthropic_messages",
    }
    resolved = {
        "provider": "kimi-coding",
        "model": "kimi-coding/k3",
        "context": "You are kimi-coder.",
        "fallback_models": [],
        "toolsets": ["terminal"],
        "workload": "coding",
        "trusted_toolset_elevation": False,
    }
    child = MagicMock()
    child._delegate_role = "leaf"
    child._delegate_saved_tool_names = []
    child.tool_progress_callback = None

    cred_calls = []

    def _creds(cfg, _parent_agent):
        cred_calls.append(dict(cfg))
        if cfg.get("provider") == "openai-codex":
            raise ValueError("No Codex credentials stored")
        return kimi_creds

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
        "tools.delegate_tool._resolve_delegation_credentials", side_effect=_creds
    ), patch(
        "tools.delegate_tool._build_child_agent", return_value=child
    ) as mock_build, patch(
        "tools.delegate_tool._run_single_child",
        return_value={
            "task_index": 0,
            "status": "completed",
            "summary": "ok",
            "error": None,
            "api_calls": 1,
            "duration_seconds": 0.1,
        },
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
        create=True,
    ):
        out = json.loads(
            delegate_task(
                goal="code something",
                capability="kimi-coder",
                parent_agent=parent,
            )
        )

    assert "error" not in out or out.get("error") is None
    # Must never have asked for default Codex
    assert not any(c.get("provider") == "openai-codex" for c in cred_calls)
    assert any(c.get("provider") == "kimi-coding" for c in cred_calls)
    mock_build.assert_called_once()
    kwargs = mock_build.call_args.kwargs
    assert kwargs.get("override_provider") == "kimi-coding"
    assert kwargs.get("model") == "k3"


def test_strict_hook_raises_from_plugin_manager():
    from hermes_cli.plugins import PluginManager

    pm = PluginManager()
    pm._hooks["resolve_delegation_capability"] = [
        lambda **_k: (_ for _ in ()).throw(ValueError("missing skill X"))
    ]
    with pytest.raises(ValueError, match="missing skill X"):
        pm.invoke_hook(
            "resolve_delegation_capability", capability="debugger", role="leaf"
        )

    # Observational hooks still swallow
    pm._hooks["pre_llm_call"] = [
        lambda **_k: (_ for _ in ()).throw(RuntimeError("noise"))
    ]
    assert pm.invoke_hook("pre_llm_call") == []
