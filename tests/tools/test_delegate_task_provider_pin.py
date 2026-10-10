#!/usr/bin/env python3
"""Per-task provider/model pinning on delegate_task (#107937).

A mixed batch must construct child 0 on the pinned route and child 1 on the
inherited batch/parent route. Blank pins fail-open; unknown providers fail-closed.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as delegate_module
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


def _route_resolver_with_unavailable_default(cfg, _parent_agent):
    provider = str((cfg or {}).get("provider") or "").strip()
    if provider == "unavailable-default":
        raise ValueError("Default provider is unavailable")
    return dict(PIN_CREDS if provider == "lmstudio-x121" else PARENT_CREDS)


def _build_child_and_attach(parent, child):
    parent._active_children.append(child)
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


def test_all_provider_pins_do_not_resolve_an_unused_default_route():
    """A fully provider-pinned public batch must not require default credentials."""
    parent = _make_mock_parent()

    with (
        patch("tools.delegate_tool._load_config", return_value={"provider": "unavailable-default"}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_route_resolver_with_unavailable_default),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=lambda **kwargs: _ok_child()),
        patch("tools.delegate_tool._run_batch", return_value='{"status": "built"}') as run_batch,
    ):
        raw = delegate_task(
            tasks=[{"goal": "Research topic A with enough length", "provider": "lmstudio-x121"}],
            parent_agent=parent,
        )

    assert json.loads(raw) == {"status": "built"}
    assert run_batch.call_args.args[0].children[0][2].provider != "unavailable-default"


def test_all_provider_pins_keep_background_batch_metadata_without_default_creds():
    """Detached dispatch labels the actual child without resolving unused credentials."""
    parent = _make_mock_parent()

    child = _ok_child()
    child.model = PIN_CREDS["model"]
    child.provider = PIN_CREDS["provider"]

    def check_background_batch(batch, background):
        assert background is True
        assert batch.creds == {"model": child.model, "provider": child.provider}
        return '{"status": "built"}'

    with (
        patch("tools.delegate_tool._load_config", return_value={"provider": "unavailable-default"}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_route_resolver_with_unavailable_default),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=child),
        patch("tools.delegate_tool._run_batch", side_effect=check_background_batch),
    ):
        raw = delegate_task(
            tasks=[{"goal": "Research topic A with enough length", "provider": "lmstudio-x121"}],
            background=True, parent_agent=parent,
        )

    assert json.loads(raw) == {"status": "built"}


def test_model_pin_uses_its_effective_route_without_resolving_default_model():
    """A model pin still validates its inherited provider but not its unused model."""
    parent = _make_mock_parent()

    def resolve(cfg, _parent_agent):
        if cfg.get("model") == "unavailable-default-model":
            raise ValueError("Default model is unavailable")
        assert cfg.get("provider") == "nous"
        return dict(PARENT_CREDS, model=cfg.get("model") or PARENT_CREDS["model"])

    with (
        patch("tools.delegate_tool._load_config", return_value={"provider": "nous", "model": "unavailable-default-model"}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=resolve),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=lambda **_kwargs: _ok_child()),
        patch("tools.delegate_tool._run_batch", return_value='{"status": "built"}'),
    ):
        raw = delegate_task(
            tasks=[{"goal": "Research topic A with enough length", "model": "task-model"}],
            parent_agent=parent,
        )

    assert json.loads(raw) == {"status": "built"}


def test_unpinned_sibling_still_requires_an_available_default_route():
    """One inheriting task keeps the whole batch fail-closed on its default route."""
    parent = _make_mock_parent()

    with (
        patch("tools.delegate_tool._load_config", return_value={"provider": "unavailable-default"}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_route_resolver_with_unavailable_default),
        patch("tools.delegate_tool._build_child_preserving_parent_tools") as build_child,
    ):
        raw = delegate_task(
            tasks=[
                {"goal": "Pinned task with enough length", "provider": "lmstudio-x121"},
                {"goal": "Inherited task with enough length"},
            ],
            parent_agent=parent,
        )

    assert "Default provider is unavailable" in json.loads(raw)["error"]
    build_child.assert_not_called()


def test_later_invalid_pin_preflights_before_any_child_is_allocated():
    """A bad later route rejects the complete batch before construction starts."""
    parent = _make_mock_parent()

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_fake_resolve),
        patch("tools.delegate_tool._build_child_preserving_parent_tools") as build_child,
    ):
        raw = delegate_task(
            tasks=[
                {"goal": "Valid task with enough length", "provider": "lmstudio-x121"},
                {"goal": "Invalid task with enough length", "provider": "not-a-real-provider"},
            ],
            parent_agent=parent,
        )

    assert UNRESOLVABLE_PIN_ERROR in json.loads(raw)["error"]
    build_child.assert_not_called()


def test_constructor_failure_closes_and_detaches_previously_built_batch_children():
    """A later construction error cannot orphan a child already attached to its parent."""
    parent = _make_mock_parent()
    first_child = MagicMock()

    def build_child(**_kwargs):
        if parent._active_children:
            raise RuntimeError("second child construction failed")
        return _build_child_and_attach(parent, first_child)

    with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=build_child):
        with pytest.raises(RuntimeError, match="second child construction failed"):
            delegate_module._build_children(
                [{"goal": "First"}, {"goal": "Second"}], [None, None], dict(PARENT_CREDS),
                top_role="leaf", max_iterations=1, parent_agent=parent, routing_cfg={},
                live_deleg_id=None, live_writers=[],
            )

    first_child.close.assert_called_once_with()
    assert parent._active_children == []


def test_setup_failure_rolls_back_the_child_which_triggered_it():
    """Ownership begins immediately after a child returns, before later setup succeeds."""
    parent = _make_mock_parent()

    class SetupFailingChild:
        def __init__(self):
            self.close = MagicMock()

        def __setattr__(self, name, value):
            if name == "_delegation_id":
                raise RuntimeError("progress setup failed")
            object.__setattr__(self, name, value)

    child = SetupFailingChild()

    with patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=lambda **_kwargs: _build_child_and_attach(parent, child),
    ):
        with pytest.raises(RuntimeError, match="progress setup failed"):
            delegate_module._build_children(
                [{"goal": "Only"}], [None], dict(PARENT_CREDS),
                top_role="leaf", max_iterations=1, parent_agent=parent, routing_cfg={},
                live_deleg_id="delegation-id", live_writers=[],
            )

    child.close.assert_called_once_with()
    assert parent._active_children == []


def test_rollback_continues_after_one_child_close_failure():
    """Cleanup errors cannot hide the original constructor failure or skip siblings."""
    parent = _make_mock_parent()
    first_child, second_child = MagicMock(), MagicMock()
    first_child.close.side_effect = RuntimeError("first close failed")
    calls = 0

    def build_child(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _build_child_and_attach(parent, first_child)
        if calls == 2:
            return _build_child_and_attach(parent, second_child)
        raise RuntimeError("third child construction failed")

    with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=build_child):
        with pytest.raises(RuntimeError, match="third child construction failed"):
            delegate_module._build_children(
                [{"goal": "First"}, {"goal": "Second"}, {"goal": "Third"}], [None, None, None],
                dict(PARENT_CREDS), top_role="leaf", max_iterations=1, parent_agent=parent, routing_cfg={},
                live_deleg_id=None, live_writers=[],
            )

    first_child.close.assert_called_once_with()
    second_child.close.assert_called_once_with()
    assert parent._active_children == []


def test_schema_advertises_provider_and_model():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "provider" in props
    assert "model" in props
    assert props["provider"]["type"] == "string"
    assert props["model"]["type"] == "string"
