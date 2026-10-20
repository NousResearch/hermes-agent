"""``pre_subagent_route`` plugin hook: per-child provider/model/reasoning-effort routing for delegate_task."""

from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.plugins_subagent_route as route_mod
from tools import delegate_tool
from tools.delegate_tool import _build_children
from tests.tools.test_delegate import _make_mock_parent

BATCH_CREDS = {
    "model": "batch-model", "provider": None, "base_url": None, "api_key": None, "api_mode": None,
    "request_overrides": None,
}


def test_provider_route_resolves_its_own_credentials():
    seen = {}

    def resolve(cfg, parent):
        seen.update(cfg)
        return {"model": cfg["model"], "provider": "anthropic", "base_url": "https://api.anthropic.example",
                "api_key": "test-only", "api_mode": "anthropic_messages", "request_overrides": {}}

    routes = iter([{"provider": "anthropic", "model": "claude-opus-5-5", "reasoning_effort": "high"}, None])
    built = []
    routing_cfg = {"reasoning_effort": "high", "base_url": "https://batch.example/v1", "api_key": "batch-key",
                   "api_mode": "chat_completions", "provider": "openai-codex", "model": "batch-model"}
    with patch.object(delegate_tool, "_build_child_preserving_parent_tools",
                      side_effect=lambda **kw: built.append(kw) or MagicMock()), \
            patch.object(route_mod, "get_plugin_subagent_route", side_effect=lambda **_: next(routes)), \
            patch.object(delegate_tool, "_resolve_delegation_credentials", side_effect=resolve):
        children, err = _build_children(
            [{"goal": "review"}, {"goal": "list"}], [None, None], dict(BATCH_CREDS), top_role="leaf",
            max_iterations=10, parent_agent=_make_mock_parent(), routing_cfg=routing_cfg,
            live_deleg_id=None, live_writers=[],
        )
    assert err is None and len(children) == 2
    # The batch endpoint never leaks into the new provider's route config.
    assert seen == {"reasoning_effort": "high", "provider": "anthropic", "model": "claude-opus-5-5"}
    assert built[0]["model"] == "claude-opus-5-5"
    assert built[0]["override_provider"] == "anthropic"
    assert built[0]["override_base_url"] == "https://api.anthropic.example"
    assert built[0]["override_reasoning_effort"] == "high"
    assert built[1]["model"] == "batch-model" and "override_reasoning_effort" not in built[1]


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
