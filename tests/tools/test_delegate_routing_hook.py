"""Behavior contracts for plugin-resolved per-task delegation routes."""

from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _resolve_delegation_route, delegate_task, DELEGATE_TASK_SCHEMA


def _parent():
    parent = MagicMock()
    parent.provider = "openai-codex"
    parent.model = "gpt-5.6-sol"
    return parent


def test_route_hook_resolves_provider_and_model_without_secrets():
    with patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[
            {
                "provider": "opencode-go",
                "model": "deepseek-v4-flash",
            }
        ],
    ) as invoke:
        resolved = _resolve_delegation_route(
            route="deepseek-flash",
            goal="Review the change",
            context="Read-only review",
            task_index=2,
            parent_agent=_parent(),
        )

    assert resolved == {
        "provider": "opencode-go",
        "model": "deepseek-v4-flash",
    }
    invoke.assert_called_once_with(
        "resolve_subagent_route",
        route="deepseek-flash",
        goal="Review the change",
        context="Read-only review",
        task_index=2,
        parent_provider="openai-codex",
        parent_model="gpt-5.6-sol",
    )


def test_empty_route_preserves_legacy_path_without_invoking_plugins():
    with patch("hermes_cli.plugins.invoke_hook") as invoke:
        assert _resolve_delegation_route(
            route=None,
            goal="Do work",
            context=None,
            task_index=0,
            parent_agent=_parent(),
        ) is None
    invoke.assert_not_called()


def test_route_is_exposed_for_single_and_batch_tasks():
    properties = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert properties["route"]["type"] == "string"
    task_properties = properties["tasks"]["items"]["properties"]
    assert task_properties["route"]["type"] == "string"


def test_single_task_route_reaches_route_resolver():
    parent = _parent()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = MagicMock()

    with (
        patch("tools.delegate_tool._load_config", return_value={"max_iterations": 1}),
        patch(
            "tools.delegate_tool._resolve_delegation_route",
            return_value={"provider": "openai-codex", "model": "gpt-5.6-luna"},
        ) as resolve_route,
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={
                "provider": "openai-codex",
                "model": "gpt-5.6-luna",
                "base_url": "https://example.invalid/v1",
                "api_key": "test-only",
                "api_mode": "codex_responses",
                "request_overrides": None,
                "max_output_tokens": None,
                "command": None,
                "args": [],
            },
        ),
        patch("tools.delegate_tool._build_child_agent", return_value=MagicMock()) as build,
        patch(
            "tools.delegate_tool._run_single_child",
            return_value={"status": "completed", "summary": "ok"},
        ),
    ):
        result = delegate_task(
            goal="Scan inputs",
            context="Read only",
            route="luna",
            parent_agent=parent,
        )

    resolve_route.assert_called_once_with(
        route="luna",
        goal="Scan inputs",
        context="Read only",
        task_index=0,
        parent_agent=parent,
    )
    assert build.call_args.kwargs["model"] == "gpt-5.6-luna"
    assert '"status": "completed"' in result


def test_batch_tasks_build_children_with_independent_routes():
    parent = _parent()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = MagicMock()
    targets = {
        "luna": {"provider": "openai-codex", "model": "gpt-5.6-luna"},
        "deepseek-flash": {"provider": "opencode-go", "model": "deepseek-v4-flash"},
    }

    def route_side_effect(*, route, **_kwargs):
        return targets[route]

    def creds_side_effect(cfg, _parent):
        provider = cfg.get("provider")
        return {
            "provider": provider,
            "model": cfg.get("model"),
            "base_url": "https://example.invalid/v1",
            "api_key": "test-only",
            "api_mode": "codex_responses" if provider == "openai-codex" else "chat_completions",
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": [],
        }

    built_models = []

    def build_side_effect(**kwargs):
        child = MagicMock()
        child.model = kwargs["model"]
        built_models.append(kwargs["model"])
        return child

    with (
        patch("tools.delegate_tool._load_config", return_value={"max_iterations": 1}),
        patch("tools.delegate_tool._resolve_delegation_route", side_effect=route_side_effect),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=creds_side_effect),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=build_side_effect),
        patch(
            "tools.delegate_tool._run_single_child",
            side_effect=lambda task_index, **_kwargs: {
                "task_index": task_index,
                "status": "completed",
                "summary": "ok",
            },
        ),
    ):
        result = delegate_task(
            tasks=[
                {"goal": "scan the repository for security issues", "route": "luna"},
                {"goal": "review the changes for correctness", "route": "deepseek-flash"},
            ],
            parent_agent=parent,
        )

    assert built_models == ["gpt-5.6-luna", "deepseek-v4-flash"]
    parsed = __import__("json").loads(result)
    assert len(parsed["results"]) == 2


def test_route_resolution_fails_closed_through_delegate_task():
    parent = _parent()
    parent._delegate_depth = 0
    parent._active_children = []

    with (
        patch("tools.delegate_tool._load_config", return_value={"max_iterations": 1}),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = delegate_task(
            goal="Do work",
            route="unknown",
            parent_agent=parent,
        )

    parsed = __import__("json").loads(result)
    assert "No enabled plugin resolved delegation route" in parsed["error"]


@pytest.mark.parametrize(
    "hook_results, message",
    [
        ([], "No enabled plugin resolved delegation route"),
        ([{"provider": "openai-codex"}], "must return non-empty 'provider' and 'model'"),
        ([{"provider": "openai-codex", "model": "gpt-5.6-luna", "api_key": "forbidden"}], "unsupported fields"),
        ([
            {"provider": "openai-codex", "model": "gpt-5.6-luna"},
            {"provider": "openai-codex", "model": "gpt-5.6-terra"},
        ], "Multiple plugins resolved delegation route"),
    ],
)
def test_route_resolution_fails_closed(hook_results, message):
    with patch("hermes_cli.plugins.invoke_hook", return_value=hook_results):
        with pytest.raises(ValueError, match=message):
            _resolve_delegation_route(
                route="luna",
                goal="Do work",
                context=None,
                task_index=0,
                parent_agent=_parent(),
            )
