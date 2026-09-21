"""Phase 1 RED contract for delegated target-profile scope.

The frozen baseline has profile handling in the batch wrapper but not in the child-builder seam.
These tests avoid setup-time TypeErrors and fail only on missing target-profile behaviour.
"""
import inspect
from types import SimpleNamespace

from tools import delegate_tool as dt


def _creds(model=""):
    return {
        "provider": "parent-provider",
        "base_url": "https://parent.invalid/v1",
        "api_key": "parent-secret",
        "api_mode": "chat_completions",
        "request_overrides": {},
        "max_output_tokens": None,
        "command": None,
        "args": None,
        "model": model,
    }


def _build(monkeypatch, profile_content, *, model=""):
    seen = []

    def fake_builder(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_builder)
    children, error = dt._build_children(
        [{"goal": "profile task", "context": "task context"}],
        [None],
        _creds(model),
        top_role="leaf",
        max_iterations=1,
        parent_agent=SimpleNamespace(),
        live_deleg_id=None,
        live_writers=[],
        profile=profile_content.get("name", "target-profile"),
        profile_content=profile_content,
    )
    assert error is None
    assert children
    assert seen
    return seen[0], children[0][2]


def test_delegate_task_schema_has_profile_parameter():
    props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "profile" in props


def test_child_builder_accepts_profile_scope_explicitly():
    assert "profile" in inspect.signature(dt._build_child_agent).parameters
    assert "profile_content" in inspect.signature(dt._build_child_agent).parameters


def test_profile_model_is_used_when_no_explicit_model(monkeypatch):
    kwargs, _ = _build(
        monkeypatch,
        {
            "name": "target-profile",
            "config": {"model": {"default": "target-model"}},
            "soul_md": None,
            "skills": [],
        },
    )
    assert kwargs["model"] == "target-model"


def test_explicit_model_wins_over_profile_default(monkeypatch):
    kwargs, _ = _build(
        monkeypatch,
        {
            "name": "target-profile",
            "config": {"model": {"default": "target-model"}},
            "soul_md": None,
            "skills": [],
        },
        model="explicit-model",
    )
    assert kwargs["model"] == "explicit-model"


def test_profile_identity_is_injected_without_parent_identity_leak(monkeypatch):
    kwargs, child = _build(
        monkeypatch,
        {
            "name": "target-profile",
            "config": {},
            "soul_md": "TARGET IDENTITY",
            "skills": [],
        },
    )
    assert "TARGET IDENTITY" in kwargs["context"]
    assert getattr(child, "_delegate_profile_name", None) == "target-profile"


def test_profile_toolsets_are_scoped_to_target_profile(monkeypatch):
    kwargs, _ = _build(
        monkeypatch,
        {
            "name": "target-profile",
            "config": {"toolsets": ["target-toolset"]},
            "soul_md": None,
            "skills": ["target-toolset"],
        },
    )
    assert kwargs["toolsets"] == ["target-toolset"]


def test_profile_provider_and_credentials_do_not_inherit_parent(monkeypatch):
    kwargs, _ = _build(
        monkeypatch,
        {
            "name": "target-profile",
            "config": {
                "provider": "target-provider",
                "credentials": {"api_key": "target-secret"},
            },
            "soul_md": None,
            "skills": [],
        },
    )
    assert kwargs["override_provider"] == "target-provider"
    assert kwargs["override_api_key"] == "target-secret"
    assert kwargs["override_api_key"] != "parent-secret"


def test_profile_fallback_chain_is_scoped_to_target_profile(monkeypatch):
    kwargs, _ = _build(
        monkeypatch,
        {
            "name": "target-profile",
            "config": {
                "fallback_providers": [
                    {"provider": "target-provider", "model": "target-fallback"}
                ]
            },
            "soul_md": None,
            "skills": [],
        },
    )
    assert kwargs.get("override_fallback_providers") == [
        {"provider": "target-provider", "model": "target-fallback"}
    ]


def test_profileless_build_remains_backward_compatible(monkeypatch):
    seen = []
    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", lambda **kwargs: seen.append(kwargs) or SimpleNamespace())
    children, error = dt._build_children(
        [{"goal": "legacy task", "context": None}],
        [None],
        _creds(model="legacy-model"),
        top_role="leaf",
        max_iterations=1,
        parent_agent=SimpleNamespace(),
        live_deleg_id=None,
        live_writers=[],
    )
    assert error is None
    assert children
    assert seen[0]["model"] == "legacy-model"
