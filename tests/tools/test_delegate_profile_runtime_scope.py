"""Phase 1 RED contract for target-profile runtime scope."""
from types import SimpleNamespace

import pytest

from tools import delegate_tool as dt


def _creds():
    return {
        "provider": "parent-provider",
        "base_url": "https://parent.invalid/v1",
        "api_key": "parent-secret",
        "api_mode": "chat_completions",
        "request_overrides": {},
        "max_output_tokens": None,
        "command": None,
        "args": None,
        "model": "",
    }


def test_target_profile_scope_applies_profile_tools_not_parent_defaults(monkeypatch):
    seen = []

    def fake_builder(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_builder)
    children, error = dt._build_children(
        [{"goal": "use target profile", "context": "task"}],
        [None],
        _creds(),
        top_role="leaf",
        max_iterations=1,
        parent_agent=SimpleNamespace(),
        live_deleg_id=None,
        live_writers=[],
        profile="target-profile",
        profile_content={
            "name": "target-profile",
            "config": {"model": {"default": "target-model"}, "provider": "target-provider"},
            "soul_md": "target identity",
            "skills": ["target-toolset"],
        },
    )
    assert error is None
    assert children
    assert seen[0]["model"] == "target-model"
    assert seen[0]["toolsets"] == ["target-toolset"]
    assert seen[0]["override_provider"] == "target-provider"


def test_target_profile_credentials_are_not_inherited_from_parent(monkeypatch):
    seen = []

    def fake_builder(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_builder)
    children, error = dt._build_children(
        [{"goal": "use target secret", "context": None}],
        [None],
        _creds(),
        top_role="leaf",
        max_iterations=1,
        parent_agent=SimpleNamespace(),
        live_deleg_id=None,
        live_writers=[],
        profile="target-profile",
        profile_content={
            "name": "target-profile",
            "config": {
                "provider": "target-provider",
                "credentials": {"api_key": "target-secret"},
            },
            "soul_md": None,
            "skills": [],
        },
    )
    assert error is None
    assert children
    assert seen[0]["override_api_key"] == "target-secret"
    assert seen[0]["override_api_key"] != "parent-secret"


def test_profile_scope_is_isolated_per_concurrent_child():
    assert callable(getattr(dt, "build_profile_runtime_scope", None))
    assert callable(getattr(dt, "profile_scope_for_child", None))
