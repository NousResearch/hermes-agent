"""Delegated children must not inherit a reasoning override that fights the
launch-local reasoning policy (F1).

``_build_child_agent`` applies ``reasoning_config_override`` to the child's
``reasoning_config``, but the child also receives ``request_overrides`` —
inherited from the parent when no provider override is configured, or pinned
via ``override_request_overrides`` for a provider-pinned route.
``chat_completions`` applies those raw overrides AFTER the reasoning derived
from ``reasoning_config``, so a launch-local reasoning DISABLE can silently
reach the wire as enabled/high, and a non-default effort can be replaced.

These tests exercise the real ``AIAgent`` + real ``build_api_kwargs`` with
synthetic credentials and no network, and assert on the OUTGOING kwargs —
constructor-level assertions are precisely why the existing tests miss this.
The launch -> child mapping is covered by the lifecycle tests; this file pins
the child -> wire half of the same path.
"""

import pytest

from agent.chat_completion_helpers import build_api_kwargs
from hermes_constants import parse_reasoning_effort
from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent

MESSAGES = [{"role": "user", "content": "hi"}]
# The conflicting policy a parent may carry in request_overrides (e.g. a stale
# /fast window or an explicit user pin): reasoning forced ON at high effort.
CONFLICTING_OVERRIDES = {"extra_body": {"reasoning": {"enabled": True, "effort": "high"}}}


def _outgoing_reasoning(child) -> dict | None:
    """The reasoning policy that would reach the wire for this child."""
    kwargs = build_api_kwargs(child, MESSAGES, None)
    extra_body = kwargs.get("extra_body") or {}
    return extra_body.get("reasoning")


def _make_parent(**kwargs) -> AIAgent:
    defaults = dict(
        base_url="https://openrouter.ai/api/v1",
        api_key="synthetic-not-a-key",
        model="deepseek/deepseek-r1",
        provider="openrouter",
        quiet_mode=True,
        platform="probe",
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        disabled_toolsets=[],
    )
    defaults.update(kwargs)
    return AIAgent(**defaults)


def _launch_child(parent, **kwargs) -> AIAgent:
    defaults = dict(
        task_index=0,
        goal="probe",
        context=None,
        toolsets=[],
        model=None,
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
    )
    defaults.update(kwargs)
    return _build_child_agent(**defaults)


def _launch_and_probe(parent, **kwargs) -> dict | None:
    child = _launch_child(parent, **kwargs)
    try:
        return _outgoing_reasoning(child)
    finally:
        child.close()


def test_positive_control_launch_effort_reaches_wire():
    """Launch effort high with no conflicting override goes out as high."""
    parent = _make_parent(request_overrides={})
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    outgoing = _launch_and_probe(parent, reasoning_config_override={"enabled": True, "effort": "high"})
    assert outgoing == {"enabled": True, "effort": "high"}


@pytest.mark.parametrize(
    ("launch_value", "expected"),
    [
        (False, None),  # None == reasoning must be disabled or absent
        ("none", None),
        ("low", {"enabled": True, "effort": "low"}),
    ],
    ids=["launch=False", "launch=none", "launch=low"],
)
def test_launch_effort_beats_inherited_reasoning_override(launch_value, expected):
    """The launch-local policy must win over a conflicting reasoning override
    inherited from the parent's request_overrides: an explicit disable stays
    disabled, and a non-default effort is not replaced by the inherited one."""
    parent = _make_parent(request_overrides=dict(CONFLICTING_OVERRIDES))
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    # The production launch path parses the request value before construction;
    # use the same parser so the test covers the real normalized input.
    override = parse_reasoning_effort(launch_value)
    outgoing = _launch_and_probe(parent, reasoning_config_override=override)
    if expected is None:
        assert outgoing in (None, {"enabled": False})
    else:
        assert outgoing == expected


def test_launch_strips_only_reasoning_overrides():
    """Unrelated request overrides must survive an explicit launch effort."""
    overrides = {
        "speed": "fast",
        "extra_body": {
            "reasoning": {"enabled": True, "effort": "high"},
            "provider": {"order": ["ExampleProvider"]},
            "chat_template_kwargs": {"enable_thinking": True, "custom_flag": 1},
        },
    }
    parent = _make_parent(request_overrides=overrides)
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    child = _launch_child(parent, reasoning_config_override={"enabled": False})
    try:
        assert child.request_overrides == {
            "speed": "fast",
            "extra_body": {
                "provider": {"order": ["ExampleProvider"]},
                "chat_template_kwargs": {"custom_flag": 1},
            },
        }
    finally:
        child.close()


def test_launch_strips_top_level_reasoning_controls():
    """Top-level reasoning carriers are stripped too; unrelated keys survive."""
    overrides = {
        "reasoning": {"enabled": True},
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "service_tier": "priority",
    }
    parent = _make_parent(request_overrides=overrides)
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    child = _launch_child(parent, reasoning_config_override={"enabled": False})
    try:
        assert child.request_overrides == {"service_tier": "priority"}
    finally:
        child.close()


def test_no_launch_override_keeps_parent_reasoning_override():
    """Omitted launch effort keeps the existing inheritance path untouched."""
    parent = _make_parent(request_overrides=dict(CONFLICTING_OVERRIDES))
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    child = _launch_child(parent)
    try:
        assert child.request_overrides == CONFLICTING_OVERRIDES
    finally:
        child.close()


def test_launch_disable_beats_provider_pinned_override():
    """A provider-pinned child's override_request_overrides bundle must not
    resurrect reasoning after a launch-local disable either."""
    parent = _make_parent(request_overrides={})
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    outgoing = _launch_and_probe(
        parent,
        override_provider="openrouter",
        override_base_url="https://openrouter.ai/api/v1",
        override_request_overrides=dict(CONFLICTING_OVERRIDES),
        reasoning_config_override={"enabled": False},
    )
    assert outgoing in (None, {"enabled": False})