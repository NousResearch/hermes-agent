"""``pre_subagent_route`` plugin hook: per-child provider/model/reasoning-effort routing for delegate_task."""

from unittest.mock import MagicMock, patch

import pytest

from tools import delegate_tool
from tools.delegate_tool import _build_children
from tests.tools.test_delegate import _make_mock_parent

BATCH_CREDS = {
    "model": "batch-model", "provider": None, "base_url": None, "api_key": None, "api_mode": None,
    "request_overrides": None,
}


def _build_with_hook(monkeypatch, hook, tasks, routing_cfg):
    """Run _build_children with *hook* registered on a real PluginManager; children are recorded, not run."""
    import hermes_cli.plugins as plugins_mod

    mgr = plugins_mod.PluginManager()
    mgr._discovered = True
    mgr._hooks["pre_subagent_route"] = [hook]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: mgr)
    resolved, built = [], []

    def resolve(cfg, parent):
        resolved.append(dict(cfg))
        if cfg["provider"] != "anthropic":
            raise ValueError(f"Cannot resolve delegation provider '{cfg['provider']}'")
        return {"model": cfg["model"], "provider": "anthropic", "base_url": "https://api.anthropic.example",
                "api_key": "test-only", "api_mode": "anthropic_messages", "request_overrides": {}}

    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", resolve)
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools",
                        lambda **kw: built.append(kw) or MagicMock())
    _, err = _build_children(
        tasks, [None] * len(tasks), dict(BATCH_CREDS), top_role="leaf", max_iterations=10,
        parent_agent=_make_mock_parent(), routing_cfg=routing_cfg, live_deleg_id=None, live_writers=[],
    )
    return err, resolved, built


def test_task_field_routes_child_to_own_provider_or_fails_the_spawn(monkeypatch):
    def hook(child_goal, task=None, **_):
        if (task or {}).get("model_role") == "deep":
            return {"provider": "anthropic", "model": "claude-opus-5-5", "reasoning_effort": "high"}
        if child_goal.startswith("broken"):
            return {"provider": "no-such-provider", "model": "m"}
        return None

    routing_cfg = {"reasoning_effort": "low", "base_url": "https://batch.example/v1", "api_key": "batch-key",
                   "api_mode": "chat_completions", "provider": "openai-codex", "model": "batch-model"}
    err, resolved, built = _build_with_hook(
        monkeypatch, hook, [{"goal": "review the diff", "model_role": "deep"}, {"goal": "list the files"}], routing_cfg)
    assert err is None
    # The batch endpoint never leaks into the new provider's route config.
    assert resolved == [{"reasoning_effort": "low", "provider": "anthropic", "model": "claude-opus-5-5"}]
    assert (built[0]["model"], built[0]["override_provider"], built[0]["override_reasoning_effort"]) == (
        "claude-opus-5-5", "anthropic", "high")
    assert built[0]["override_base_url"] == "https://api.anthropic.example"
    assert built[1]["model"] == "batch-model" and "override_reasoning_effort" not in built[1]

    # An unresolvable provider fails the spawn; no child silently runs on the batch model.
    err, _, built = _build_with_hook(monkeypatch, hook, [{"goal": "broken route"}], routing_cfg)
    assert built == [] and "no-such-provider" in err


@pytest.mark.parametrize(("credentials_cfg", "expected"), [(None, True), ({"model": "pinned-review-model"}, False)])
def test_delegate_task_skips_plugin_routes_for_pinned_credentials(credentials_cfg, expected):
    seen = {}

    def fake_build_children(*args, **kwargs):
        seen.update(kwargs)
        return [], "stop after build"

    with patch.object(delegate_tool, "_load_config", return_value={}), \
            patch.object(delegate_tool, "_resolve_delegation_credentials", return_value=dict(BATCH_CREDS)), \
            patch.object(delegate_tool, "_oneshot_spawn_budget", return_value=None), \
            patch.object(delegate_tool, "_announce_batch"), \
            patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])), \
            patch.object(delegate_tool, "_build_children", side_effect=fake_build_children):
        delegate_tool.delegate_task(goal="review the diff", parent_agent=_make_mock_parent(),
                                    credentials_cfg=credentials_cfg)
    assert seen["plugin_routes"] is expected
