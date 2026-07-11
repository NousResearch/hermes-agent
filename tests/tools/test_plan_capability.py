from types import SimpleNamespace
import json
import threading
import pytest

from agent.chat_completion_helpers import build_api_kwargs
from tools import approval
from tools.registry import registry
from tools.todo_tool import TODO_SCHEMA, TodoStore
from toolsets import resolve_toolset


@pytest.fixture(autouse=True)
def _isolated_local_capability_runtime(monkeypatch):
    """Keep ordinary tests independent of this machine's private Cloud config."""
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "")
    monkeypatch.setattr(approval, "_observed_session_message_id", lambda: "")


def _inserted_receipt() -> str:
    return json.dumps({
        "success": True,
        "inserted": True,
        "readback_verified": True,
        "event_id": "event-1",
    })


def test_exact_plan_capability_is_owner_bound_expiring_and_consumed(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    approval.clear_session("session-1")
    grant = approval.grant_plan_capability(
        session_key="session-1",
        plan_id="plan-1",
        exact_commands=["git status --short"],
        approved_by_user_id="owner-1",
        max_uses_per_command=1,
    )
    assert grant["command_count"] == 1
    assert approval.consume_plan_capability("session-1", "git status --short") == "plan-1"
    assert approval.consume_plan_capability("session-1", "git status --short") is None
    assert approval.consume_plan_capability("session-1", "git status") is None


def test_plan_capability_rejects_non_owner(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    with pytest.raises(PermissionError):
        approval.grant_plan_capability(
            session_key="session-1",
            plan_id="plan-1",
            exact_commands=["git status"],
            approved_by_user_id="teammate",
        )


def test_pending_model_plan_forces_tool_choice_required():
    class _Transport:
        def build_kwargs(self, **kwargs):
            return kwargs

    store = TodoStore()
    store.write([{"id": "1", "content": "finish", "status": "pending"}])
    agent = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "todo"}}],
        _todo_store=store,
        api_mode="codex_responses",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        _base_url_hostname="chatgpt.com",
        _base_url_lower="https://chatgpt.com/backend-api/codex",
        reasoning_config={"effort": "max"},
        session_id="s1",
        max_tokens=None,
        request_overrides=None,
        _prepare_messages_for_non_vision_model=lambda messages: messages,
        _resolved_api_call_timeout=lambda: 60,
        _is_copilot_url=lambda: False,
        _github_models_reasoning_extra_body=lambda: None,
        _get_transport=lambda: _Transport(),
    )
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "do it"}])
    assert kwargs["tool_choice"] == "required"

    store.write([{"id": "1", "content": "finish", "status": "completed"}])
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "do it"}])
    assert kwargs["tool_choice"] is None


def test_todo_schema_and_registry_expose_plan_approval_to_model(monkeypatch):
    properties = TODO_SCHEMA["parameters"]["properties"]
    assert "plan_approval" in properties
    assert "goal_outcome" in properties

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    approval.clear_session("session-registry")
    result = registry.dispatch(
        "todo",
        {
            "todos": [{"id": "1", "content": "approved step", "status": "pending"}],
            "plan_approval": {
                "plan_id": "plan-registry",
                "exact_commands": ["git status --short"],
                "max_uses_per_command": 1,
            },
        },
        store=TodoStore(),
        session_key="session-registry",
        user_id="owner-1",
    )
    data = json.loads(result)
    assert data["plan_capability"]["plan_id"] == "plan-registry"
    assert approval.consume_plan_capability(
        "session-registry", "git status --short"
    ) == "plan-registry"


def test_runtime_observed_user_must_match_approved_owner(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "teammate")
    with pytest.raises(PermissionError, match="runtime-observed user"):
        approval.grant_plan_capability(
            session_key="session-observed-owner",
            plan_id="plan-observed-owner",
            exact_commands=["git status --short"],
            approved_by_user_id="owner-1",
        )


def test_explicit_canonical_config_fails_closed_when_helper_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "approvals": {"plan_owner_user_ids": ["owner-canonical-policy"]},
            "canonical_brain": {"tools_enabled": True},
        },
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.check_canonical_brain_requirements",
        lambda: False,
    )
    assert approval._canonical_brain_required() is True

    monkeypatch.setattr(
        approval,
        "_observed_session_user_id",
        lambda: "owner-canonical-policy",
    )
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: "msg-canonical-policy",
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: False,
    )
    with pytest.raises(PermissionError, match="exact active plan_id"):
        approval.grant_plan_capability(
            session_key="session-canonical-policy",
            plan_id="plan-canonical-policy",
            exact_commands=["git status --short"],
            approved_by_user_id="owner-canonical-policy",
            canonical_case_id="case:canonical-policy",
        )
    assert approval.consume_plan_capability(
        "session-canonical-policy", "git status --short"
    ) is None


def test_gateway_owner_is_discord_bound_and_requires_current_observed_message(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["discord-owner"]}},
    )
    monkeypatch.setattr(
        approval,
        "_observed_session_user_id",
        lambda: "discord-owner",
    )
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "telegram")
    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: "telegram-message",
    )
    with pytest.raises(PermissionError, match="Discord platform"):
        approval.grant_plan_capability(
            session_key="session-platform-binding",
            plan_id="plan-platform-binding",
            exact_commands=["git status --short"],
            approved_by_user_id="discord-owner",
        )

    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(approval, "_observed_session_message_id", lambda: "")
    with pytest.raises(PermissionError, match="runtime-observed owner message_id"):
        approval.grant_plan_capability(
            session_key="session-platform-binding",
            plan_id="plan-platform-binding",
            exact_commands=["git status --short"],
            approved_by_user_id="discord-owner",
            source_refs={"message_id": "model-supplied-old-message"},
        )

    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: "current-discord-message",
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: _inserted_receipt(),
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_capability_check",
        lambda **kwargs: _inserted_receipt(),
    )
    grant = approval.grant_plan_capability(
        session_key="session-platform-binding",
        plan_id="plan-platform-binding",
        exact_commands=["git status --short"],
        approved_by_user_id="discord-owner",
        canonical_case_id="case:platform-binding",
        source_refs={"message_id": "model-supplied-old-message"},
    )
    assert grant["state"] == "granted"
    stored = approval._plan_capabilities["session-platform-binding"][
        "plan-platform-binding"
    ]
    assert stored["source_refs"]["message_id"] == "current-discord-message"
    assert stored["source_refs"].get("message_id") != "model-supplied-old-message"


def test_existing_capability_cannot_be_consumed_by_same_id_on_another_platform(
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["discord-consume-owner"]}},
    )
    session_key = "session-consume-platform-binding"
    command = "git status --short"
    approval.clear_session(session_key)
    monkeypatch.setattr(
        approval,
        "_observed_session_user_id",
        lambda: "discord-consume-owner",
    )
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: "consume-platform-approval",
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: _inserted_receipt(),
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_capability_check",
        lambda **kwargs: _inserted_receipt(),
    )
    approval.grant_plan_capability(
        session_key=session_key,
        plan_id="plan-consume-platform-binding",
        exact_commands=[command],
        approved_by_user_id="discord-consume-owner",
        max_uses_per_command=1,
        canonical_case_id="case:consume-platform-binding",
    )

    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "telegram")
    assert approval.consume_plan_capability(session_key, command) is None

    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    assert approval.consume_plan_capability(session_key, command) == (
        "plan-consume-platform-binding"
    )


def test_observed_approval_message_cannot_mint_fresh_authority_after_restart_or_expiry(
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["discord-replay-owner"]}},
    )
    monkeypatch.setattr(
        approval,
        "_observed_session_user_id",
        lambda: "discord-replay-owner",
    )
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    observed_message = {"id": "replay-message-restart"}
    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: observed_message["id"],
    )
    session_key = "session-replay-ledger"
    plan_id = "plan-replay-ledger"
    approval.clear_session(session_key)
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: _inserted_receipt(),
    )

    first = approval.grant_plan_capability(
        session_key=session_key,
        plan_id=plan_id,
        exact_commands=["git status --short"],
        approved_by_user_id="discord-replay-owner",
        canonical_case_id="case:replay-ledger",
    )
    retry = approval.grant_plan_capability(
        session_key=session_key,
        plan_id=plan_id,
        exact_commands=["git status --short"],
        approved_by_user_id="discord-replay-owner",
        canonical_case_id="case:replay-ledger",
    )
    assert retry["approval_id"] == first["approval_id"]
    assert retry["existing_capability"] is True

    approval.clear_session(session_key)
    with pytest.raises(PermissionError, match="already used"):
        approval.grant_plan_capability(
            session_key=session_key,
            plan_id=plan_id,
            exact_commands=["git status --short"],
            approved_by_user_id="discord-replay-owner",
            canonical_case_id="case:replay-ledger",
        )

    observed_message["id"] = "replay-message-expiry"
    approval.grant_plan_capability(
        session_key=session_key,
        plan_id=plan_id,
        exact_commands=["git status --short"],
        approved_by_user_id="discord-replay-owner",
        canonical_case_id="case:replay-ledger",
    )
    approval._plan_capabilities[session_key][plan_id]["expires_at"] = 0
    with pytest.raises(PermissionError, match="already used"):
        approval.grant_plan_capability(
            session_key=session_key,
            plan_id=plan_id,
            exact_commands=["git status --short"],
            approved_by_user_id="discord-replay-owner",
            canonical_case_id="case:replay-ledger",
        )

    observed_message["id"] = "fresh-owner-message"
    fresh = approval.grant_plan_capability(
        session_key=session_key,
        plan_id=plan_id,
        exact_commands=["git status --short"],
        approved_by_user_id="discord-replay-owner",
        canonical_case_id="case:replay-ledger",
    )
    assert fresh["approval_id"] != first["approval_id"]


def test_canonical_runtime_requires_observed_owner_case_and_active_exact_plan(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    monkeypatch.setattr(approval, "_canonical_brain_required", lambda: True)
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(approval, "_observed_session_message_id", lambda: "msg-canonical")

    with pytest.raises(PermissionError, match="runtime-observed owner"):
        approval.grant_plan_capability(
            session_key="session-canonical",
            plan_id="plan-canonical",
            exact_commands=["git status --short"],
            approved_by_user_id="owner-1",
            canonical_case_id="case:canonical",
        )

    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    with pytest.raises(PermissionError, match="canonical_case_id"):
        approval.grant_plan_capability(
            session_key="session-canonical",
            plan_id="plan-canonical",
            exact_commands=["git status --short"],
            approved_by_user_id="owner-1",
        )

    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: False,
    )
    with pytest.raises(PermissionError, match="exact active plan_id"):
        approval.grant_plan_capability(
            session_key="session-canonical",
            plan_id="plan-canonical",
            exact_commands=["git status --short"],
            approved_by_user_id="owner-1",
            canonical_case_id="case:canonical",
        )

    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: _inserted_receipt(),
    )
    grant = approval.grant_plan_capability(
        session_key="session-canonical",
        plan_id="plan-canonical",
        exact_commands=["git status --short"],
        approved_by_user_id="owner-1",
        canonical_case_id="case:canonical",
    )
    assert grant["canonical_readback_verified"] is True


def test_canonical_grant_requires_new_insert_and_verified_readback(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: "msg-grant-receipt",
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    session_key = "session-grant-receipt"
    approval.clear_session(session_key)
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": False,
            "deduped": True,
            "readback_verified": True,
        }),
    )
    with pytest.raises(RuntimeError, match="not durably verified"):
        approval.grant_plan_capability(
            session_key=session_key,
            plan_id="plan-grant-receipt",
            exact_commands=["git status --short"],
            approved_by_user_id="owner-1",
            canonical_case_id="case:grant-receipt",
        )
    assert approval.consume_plan_capability(session_key, "git status --short") is None
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: _inserted_receipt(),
    )
    retry = approval.grant_plan_capability(
        session_key=session_key,
        plan_id="plan-grant-receipt",
        exact_commands=["git status --short"],
        approved_by_user_id="owner-1",
        canonical_case_id="case:grant-receipt",
    )
    assert retry["canonical_readback_verified"] is True


def test_canonical_consume_revalidates_active_plan_and_rejects_deduped_receipt(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: "msg-consume-receipt",
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    session_key = "session-consume-receipt"
    case_id = "case:consume-receipt"
    plan_id = "plan-consume-receipt"
    command = "git status --short"
    approval.clear_session(session_key)
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: _inserted_receipt(),
    )
    approval.grant_plan_capability(
        session_key=session_key,
        plan_id=plan_id,
        exact_commands=[command],
        approved_by_user_id="owner-1",
        canonical_case_id=case_id,
        max_uses_per_command=1,
    )

    monkeypatch.setattr(approval, "_canonical_brain_required", lambda: True)
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: False,
    )
    check_calls = []
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_capability_check",
        lambda **kwargs: check_calls.append(kwargs) or _inserted_receipt(),
    )
    assert approval.consume_plan_capability(session_key, command) is None
    assert check_calls == []

    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_capability_check",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": False,
            "deduped": True,
            "readback_verified": True,
        }),
    )
    assert approval.consume_plan_capability(session_key, command) is None

    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_capability_check",
        lambda **kwargs: _inserted_receipt(),
    )
    assert approval.consume_plan_capability(session_key, command) == plan_id
    assert approval.consume_plan_capability(session_key, command) is None


def test_concurrent_canonical_receipt_failure_cannot_overcredit_counter(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        approval,
        "_observed_session_message_id",
        lambda: "msg-concurrent-receipt",
    )
    monkeypatch.setattr(
        "tools.canonical_brain_tool.canonical_active_plan_matches",
        lambda **kwargs: True,
    )
    session_key = "session-concurrent-receipt"
    plan_id = "plan-concurrent-receipt"
    command = "git status --short"
    approval.clear_session(session_key)
    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_approval_receipt",
        lambda **kwargs: _inserted_receipt(),
    )
    approval.grant_plan_capability(
        session_key=session_key,
        plan_id=plan_id,
        exact_commands=[command],
        approved_by_user_id="owner-1",
        canonical_case_id="case:concurrent-receipt",
        max_uses_per_command=2,
    )

    first_entered = threading.Event()
    allow_first_failure = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def _record_check(**kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_entered.set()
            assert allow_first_failure.wait(timeout=2)
            return json.dumps({
                "success": False,
                "inserted": False,
                "readback_verified": False,
            })
        return _inserted_receipt()

    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_capability_check",
        _record_check,
    )
    results = []
    first = threading.Thread(
        target=lambda: results.append(
            approval.consume_plan_capability(session_key, command)
        )
    )
    second = threading.Thread(
        target=lambda: results.append(
            approval.consume_plan_capability(session_key, command)
        )
    )
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    allow_first_failure.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive() and not second.is_alive()
    assert sorted(value or "" for value in results) == ["", plan_id]

    monkeypatch.setattr(
        "tools.canonical_brain_tool.record_plan_capability_check",
        lambda **kwargs: _inserted_receipt(),
    )
    assert approval.consume_plan_capability(session_key, command) == plan_id
    assert approval.consume_plan_capability(session_key, command) is None


def test_canonical_toolset_exposes_receipt_coupled_route_back_executor():
    assert "route_back_execute" in resolve_toolset("canonical_brain")
    assert "route_back_execute" in resolve_toolset("hermes-cli")
