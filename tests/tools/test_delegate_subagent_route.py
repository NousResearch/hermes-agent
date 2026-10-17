"""``pre_subagent_route`` plugin hook: per-child provider/model/reasoning-effort routing for delegate_task."""

from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.plugins as plugins_mod
import hermes_cli.plugins_subagent_route as route_mod
from tools import delegate_tool
from tools.delegate_tool import _build_child_agent, _build_children
from tests.tools.test_delegate import _make_mock_parent

BATCH_CREDS = {
    "model": "batch-model", "provider": None, "base_url": None, "api_key": None, "api_mode": None,
    "request_overrides": None,
}


def _route_with(results):
    """Run get_plugin_subagent_route with hook callbacks stubbed to return *results*."""
    with patch.object(plugins_mod, "has_hook", return_value=True), \
            patch.object(plugins_mod, "invoke_hook", return_value=list(results)) as invoke:
        route = route_mod.get_plugin_subagent_route(child_goal="Review the auth module", task_index=2)
    return route, invoke


# --- hook contract ---------------------------------------------------------------------------------------------

def test_hook_is_registered_and_shell_unsupported():
    assert "pre_subagent_route" in plugins_mod.VALID_HOOKS
    assert "pre_subagent_route" in plugins_mod.SHELL_UNSUPPORTED_HOOKS


def test_no_registered_hook_skips_invoke():
    with patch.object(plugins_mod, "has_hook", return_value=False), \
            patch.object(plugins_mod, "invoke_hook") as invoke:
        assert route_mod.get_plugin_subagent_route(child_goal="x") is None
    invoke.assert_not_called()


def test_hook_receives_child_context():
    _, invoke = _route_with([None])
    args, kwargs = invoke.call_args
    assert args == ("pre_subagent_route",)
    assert kwargs["child_goal"] == "Review the auth module"
    assert kwargs["task_index"] == 2
    assert {"parent_session_id", "task_count", "child_context", "child_role", "default_provider",
            "default_model", "default_reasoning_effort"} <= set(kwargs)


@pytest.mark.parametrize(("results", "expected"), [
    ([None], None),
    (["not-a-dict", 42], None),
    ([{}], None),
    ([{"model": "claude-opus-5-5"}], {"model": "claude-opus-5-5"}),
    ([{"provider": "anthropic", "model": "claude-opus-5-5", "reasoning_effort": "HIGH"}],
     {"provider": "anthropic", "model": "claude-opus-5-5", "reasoning_effort": "high"}),
    ([{"reasoning_effort": "low"}], {"reasoning_effort": "low"}),
    ([{"reasoning_effort": False}], {"reasoning_effort": False}),
    # provider without a model is dropped; the rest of the route survives
    ([{"provider": "anthropic", "reasoning_effort": "high"}], {"reasoning_effort": "high"}),
    ([{"provider": "anthropic"}], None),
    # unknown effort is dropped
    ([{"model": "m", "reasoning_effort": "hgih"}], {"model": "m"}),
    # oversized / blank strings are rejected
    ([{"model": "x" * 500}], None),
    ([{"model": "   "}], None),
    # run-all-then-pick-first: the first valid route wins
    ([None, {"model": "first"}, {"model": "second"}], {"model": "first"}),
])
def test_route_sanitization(results, expected):
    route, _ = _route_with(results)
    assert route == expected


# --- reasoning override reaches the child --------------------------------------------------------------------

def _child_kwargs(cfg, **overrides):
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch("run_agent.AIAgent") as mock_agent:
        mock_agent.return_value = MagicMock()
        _build_child_agent(task_index=0, goal="route wiring", context=None, toolsets=None, model=None,
                           max_iterations=10, parent_agent=_make_mock_parent(), task_count=1, **overrides)
    return mock_agent.call_args.kwargs


def test_reasoning_override_beats_delegation_config():
    kwargs = _child_kwargs({"reasoning_effort": "high"}, override_reasoning_effort="low")
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_no_override_keeps_delegation_config():
    kwargs = _child_kwargs({"reasoning_effort": "high"})
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_override_false_disables_reasoning():
    kwargs = _child_kwargs({"reasoning_effort": "high"}, override_reasoning_effort=False)
    assert kwargs["reasoning_config"] == {"enabled": False}


# --- per-child routing in _build_children -------------------------------------------------------------------

def _build(tasks, routes, routing_cfg=None, resolve=None, **build_kwargs):
    built = []

    def fake_build(**kwargs):
        built.append(kwargs)
        return MagicMock()

    route_iter = iter(routes)
    patches = [
        patch.object(delegate_tool, "_build_child_preserving_parent_tools", side_effect=fake_build),
        patch.object(route_mod, "get_plugin_subagent_route", side_effect=lambda **_: next(route_iter)),
    ]
    if resolve is not None:
        patches.append(patch.object(delegate_tool, "_resolve_delegation_credentials", side_effect=resolve))
    for p in patches:
        p.start()
    try:
        children, err = _build_children(
            tasks, [None] * len(tasks), dict(BATCH_CREDS), top_role="leaf", max_iterations=10,
            parent_agent=_make_mock_parent(), routing_cfg=routing_cfg or {"reasoning_effort": "high"},
            live_deleg_id=None, live_writers=[], **build_kwargs,
        )
    finally:
        for p in reversed(patches):
            p.stop()
    assert err is None and len(children) == len(tasks)
    return built


def test_no_route_keeps_batch_route():
    built = _build([{"goal": "a"}, {"goal": "b"}], [None, None])
    for kwargs in built:
        assert kwargs["model"] == "batch-model"
        assert kwargs["override_provider"] is None
        assert "override_reasoning_effort" not in kwargs


def test_model_and_effort_route_per_child():
    built = _build([{"goal": "list files"}, {"goal": "review design"}],
                   [{"model": "small-model", "reasoning_effort": "low"}, None])
    assert built[0]["model"] == "small-model"
    assert built[0]["override_reasoning_effort"] == "low"
    assert built[0]["override_provider"] is None  # same provider as the batch
    assert built[1]["model"] == "batch-model"
    assert "override_reasoning_effort" not in built[1]


def test_provider_route_resolves_its_own_credentials():
    seen = {}

    def resolve(cfg, parent):
        seen.update(cfg)
        return {"model": cfg["model"], "provider": "anthropic", "base_url": "https://api.anthropic.example",
                "api_key": "test-only", "api_mode": "anthropic_messages", "request_overrides": {}}

    routing_cfg = {"reasoning_effort": "high", "base_url": "https://batch.example/v1", "api_key": "batch-key",
                   "api_mode": "chat_completions", "provider": "openai-codex", "model": "batch-model"}
    built = _build([{"goal": "review"}],
                   [{"provider": "anthropic", "model": "claude-opus-5-5", "reasoning_effort": "high"}],
                   routing_cfg=routing_cfg, resolve=resolve)
    # The batch endpoint never leaks into the new provider's route config.
    assert seen == {"reasoning_effort": "high", "provider": "anthropic", "model": "claude-opus-5-5"}
    kwargs = built[0]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["override_provider"] == "anthropic"
    assert kwargs["override_base_url"] == "https://api.anthropic.example"
    assert kwargs["override_api_mode"] == "anthropic_messages"
    assert kwargs["override_reasoning_effort"] == "high"
    assert kwargs["routing_cfg"]["provider"] == "anthropic"


def test_unresolvable_provider_falls_back_to_batch_route():
    def resolve(cfg, parent):
        raise ValueError("no credentials for provider")

    built = _build([{"goal": "x"}], [{"provider": "nope", "model": "m"}], resolve=resolve)
    assert built[0]["model"] == "batch-model"
    assert built[0]["override_provider"] is None


def test_hook_exception_falls_back_to_batch_route():
    def boom(**_):
        raise RuntimeError("router down")

    built = []
    with patch.object(delegate_tool, "_build_child_preserving_parent_tools",
                      side_effect=lambda **kw: built.append(kw) or MagicMock()), \
            patch.object(route_mod, "get_plugin_subagent_route", side_effect=boom):
        children, err = _build_children(
            [{"goal": "x"}], [None], dict(BATCH_CREDS), top_role="leaf", max_iterations=10,
            parent_agent=_make_mock_parent(), routing_cfg={}, live_deleg_id=None, live_writers=[],
        )
    assert err is None and len(children) == 1
    assert built[0]["model"] == "batch-model"


# --- caller-pinned routes (/review → auxiliary.review) ---------------------------------------------------------

def test_plugin_routes_disabled_skips_hook():
    with patch.object(route_mod, "get_plugin_subagent_route") as hook:
        built = _build([{"goal": "review this"}], [], plugin_routes=False)
        hook.assert_not_called()
    assert built[0]["model"] == "batch-model"


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
