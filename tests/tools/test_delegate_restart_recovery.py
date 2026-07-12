"""Shared live/resume adapter coverage for delegate_task."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from gateway.session_context import get_session_env, restore_session_vars, set_session_vars
from agent.runtime_cwd import resolve_agent_cwd
from tools import delegate_tool as dt


def _parent():
    return SimpleNamespace(
        model="parent-model",
        provider="parent-provider",
        base_url="https://parent.invalid/v1",
        api_mode="chat_completions",
        enabled_toolsets=["file", "terminal"],
        reasoning_config={"effort": "high"},
        fallback_model=None,
        service_tier=None,
        provider_preferences=None,
        _delegate_depth=0,
        terminal_cwd="/tmp",
    )


def test_gateway_background_spec_persists_effective_settings_without_credentials(monkeypatch):
    tokens = set_session_vars(
        platform="telegram",
        chat_id="123",
        thread_id="7",
        user_id="u1",
        user_name="Ace",
        session_key="agent:main:telegram:dm:123:7",
        session_id="sess-parent",
        profile="work",
        async_delivery=True,
    )
    try:
        with patch("gateway.status.get_current_boot_id", return_value="100:1.0"):
            spec, boot_id = dt._build_durable_background_spec(
                task_list=[{"goal": "finish", "context": "draft", "role": "leaf"}],
                shared_context="shared",
                top_role="leaf",
                inherit_context=False,
                cfg={"resume_on_restart": True, "provider": "named-provider"},
                creds={
                    "model": "child-model",
                    "provider": "custom",
                    "base_url": "https://child.invalid/v1",
                    "api_mode": "chat_completions",
                    "api_key": "MUST-NOT-PERSIST",
                },
                parent_agent=_parent(),
                session_key="agent:main:telegram:dm:123:7",
                parent_session_id="sess-parent",
                origin_ui_session_id="",
                max_iterations=45,
            )
    finally:
        restore_session_vars(tokens)

    assert boot_id == "100:1.0"
    assert spec is not None
    assert spec["profile"] == "work"
    assert spec["source"]["kind"] == "single"
    assert spec["route"]["chat_type"] == "dm"
    assert spec["route"]["parent_session_id"] == "sess-parent"
    assert spec["execution"]["model"] == "child-model"
    assert spec["execution"]["credential_ref"]["source"] == "provider"
    assert spec["execution"]["credential_ref"]["provider"] == "named-provider"
    assert "MUST-NOT-PERSIST" not in json.dumps(spec)
    assert "api_key" not in json.dumps(spec)


def test_direct_endpoint_spec_persists_credential_reference_not_secret():
    parent = _parent()
    parent.provider = "openrouter"
    tokens = set_session_vars(
        platform="telegram",
        chat_id="123",
        session_key="agent:main:telegram:dm:123",
        session_id="sess-parent",
        profile="work",
    )
    try:
        with patch("gateway.status.get_current_boot_id", return_value="100:1.0"):
            spec, _ = dt._build_durable_background_spec(
                task_list=[{"goal": "finish", "context": None}],
                shared_context=None,
                top_role="leaf",
                inherit_context=False,
                cfg={
                    "resume_on_restart": True,
                    "base_url": "https://direct.invalid/v1",
                    "api_key": "OLD-MUST-NOT-PERSIST",
                },
                creds={
                    "model": "child-model",
                    "provider": "custom",
                    "base_url": "https://direct.invalid/v1",
                    "api_mode": "chat_completions",
                    "api_key": "OLD-MUST-NOT-PERSIST",
                },
                parent_agent=parent,
                session_key="agent:main:telegram:dm:123",
                parent_session_id="sess-parent",
                origin_ui_session_id="",
                max_iterations=45,
            )
    finally:
        restore_session_vars(tokens)

    assert spec is not None
    credential_ref = spec["execution"]["credential_ref"]
    assert credential_ref == {
        "source": "delegation_config",
        "parent_provider": "openrouter",
    }
    assert "OLD-MUST-NOT-PERSIST" not in json.dumps(spec)


def test_recovered_runner_reuses_delegate_task_with_continuation(monkeypatch):
    record = {
        "source": {
            "kind": "batch",
            "tasks": [
                {"goal": "one", "context": "draft one", "role": "leaf"},
                {"goal": "two", "context": None, "role": "leaf"},
            ],
        },
        "execution": {"max_iterations": 33, "workspace_hint": "/tmp"},
        "profile": "work",
        "route": {
            "platform": "telegram",
            "chat_id": "123",
            "session_key": "agent:main:telegram:dm:123",
            "parent_session_id": "sess-parent",
            "profile": "work",
        },
    }
    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        captured["runtime_session_key"] = get_session_env("HERMES_SESSION_KEY", "")
        captured["runtime_profile"] = get_session_env("HERMES_SESSION_PROFILE", "")
        captured["runtime_cwd"] = str(resolve_agent_cwd())
        return json.dumps({"results": [{"task_index": 0}, {"task_index": 1}]})

    monkeypatch.setattr(dt, "delegate_task", fake_delegate_task)
    runner = dt.build_recovered_delegation_runner(
        record,
        "CONTINUE after restart",
        _parent(),
    )
    result = runner()

    assert len(result["results"]) == 2
    assert captured["background"] is False
    assert captured["_recovery_spec"] is record
    assert captured["max_iterations"] == 33
    assert captured["tasks"][0]["context"] == "draft one\n\nCONTINUE after restart"
    assert captured["tasks"][1]["context"] == "CONTINUE after restart"
    assert captured["runtime_session_key"] == "agent:main:telegram:dm:123"
    assert captured["runtime_profile"] == "work"
    assert captured["runtime_cwd"] == "/tmp"


def test_durable_spec_materializes_inherited_parent_context():
    parent = _parent()
    child = SimpleNamespace(
        prefill_messages=[{"role": "user", "content": "folded parent transcript"}]
    )
    tokens = set_session_vars(
        platform="telegram",
        chat_id="123",
        session_key="agent:main:telegram:dm:123",
        session_id="sess-parent",
        profile="work",
    )
    try:
        with patch("gateway.status.get_current_boot_id", return_value="100:1.0"):
            spec, _ = dt._build_durable_background_spec(
                task_list=[{"goal": "finish", "context": None, "inherit_context": True}],
                shared_context=None,
                top_role="leaf",
                inherit_context=True,
                cfg={"resume_on_restart": True},
                creds={"model": "child-model"},
                parent_agent=parent,
                session_key="agent:main:telegram:dm:123",
                parent_session_id="sess-parent",
                origin_ui_session_id="",
                max_iterations=45,
                children=[child],
            )
    finally:
        restore_session_vars(tokens)
    assert spec is not None
    task = spec["source"]["tasks"][0]
    assert task["materialized_prefill_messages"] == child.prefill_messages
