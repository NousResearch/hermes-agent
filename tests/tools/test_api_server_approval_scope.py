"""Unit witnesses for the resolver-backed approval boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from gateway.platforms.api_server_runs import _unregister_approval_notify
from tools import approval as approval_module
from tools import approval_context


@pytest.fixture(autouse=True)
def isolate_approval_state(monkeypatch):
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(
        approval_context, "_get_unattended_approval_mode", lambda: "deny"
    )
    yield
    with approval_module._lock:
        approval_module._gateway_queues.clear()
        approval_module._gateway_notify_cbs.clear()
        approval_module._pending.clear()


@pytest.mark.parametrize("platform", ["webhook", "msgraph_webhook"])
def test_notifier_only_unattended_platform_stays_fail_closed(platform):
    """An outbound notifier does not prove that a client can resolve approval."""
    key = f"notifier-only-{platform}"
    tokens = set_session_vars(platform=platform, session_key=key)
    session_token = approval_context.set_current_session_key(key)
    notified: list[dict] = []
    approval_module.register_gateway_notify(key, notified.append)
    try:
        result = approval_module.check_execute_code_guard(
            "print('unattended probe')", "local"
        )
        assert approval_context._is_gateway_approval_context() is False
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert notified == []
        assert key not in approval_module._gateway_queues
    finally:
        approval_module.unregister_gateway_notify(key)
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)


def test_api_server_listenerless_execute_code_stays_fail_closed():
    """An API session without the explicit run capability remains unattended."""
    key = "listenerless-api"
    tokens = set_session_vars(platform="api_server", session_key=key)
    session_token = approval_context.set_current_session_key(key)
    try:
        result = approval_module.check_execute_code_guard(
            "print('listenerless probe')", "local"
        )
        assert approval_context._is_gateway_approval_context() is False
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert key not in approval_module._gateway_queues
    finally:
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)


def test_api_server_notifier_only_does_not_grant_resolver_capability():
    """The shared outbound callback registry is not an inbound run resolver."""
    key = "notifier-only-api-server"
    tokens = set_session_vars(platform="api_server", session_key=key)
    session_token = approval_context.set_current_session_key(key)
    notified: list[dict] = []
    approval_module.register_gateway_notify(key, notified.append)
    try:
        result = approval_module.check_execute_code_guard(
            "print('notifier-only probe')", "local"
        )
        assert approval_context._is_gateway_approval_context() is False
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert notified == []
        assert key not in approval_module._gateway_queues
    finally:
        approval_module.unregister_gateway_notify(key)
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)


def test_resolver_capability_requires_exact_api_session_key():
    """The capability cannot bleed into a different concurrent session."""
    key = "resolver-api-session"
    tokens = set_session_vars(platform="api_server", session_key=key)
    session_token = approval_context.set_current_session_key(key)
    resolver = approval_context.create_approval_resolver(key)
    resolver.activate()
    resolver_token = approval_context.set_current_approval_resolver(resolver)
    try:
        assert approval_context._is_unattended_platform_approval_context() is False
        assert approval_context._is_gateway_approval_context() is True

        other_session_token = approval_context.set_current_session_key("other-session")
        try:
            assert approval_context._is_unattended_platform_approval_context() is True
            assert approval_context._is_gateway_approval_context() is False
        finally:
            approval_context.reset_current_session_key(other_session_token)
    finally:
        resolver.revoke()
        approval_context.reset_current_approval_resolver(resolver_token)
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)


def test_revoked_run_resolver_fails_closed_before_notifier_teardown(monkeypatch):
    """A cancelled run cannot leave a late tool call with an unanswerable waiter."""
    key = "revoked-api-run"
    tokens = set_session_vars(platform="api_server", session_key=key)
    session_token = approval_context.set_current_session_key(key)
    resolver = approval_context.create_approval_resolver(key)
    resolver.activate()
    resolver_token = approval_context.set_current_approval_resolver(resolver)
    notified: list[dict] = []
    approval_module.register_gateway_notify(key, notified.append)
    try:
        _unregister_approval_notify(key, resolver)
        # Model the race where the gate computed ``is_gateway=True`` just
        # before cancellation revoked the capability and removed the notifier.
        monkeypatch.setattr(
            approval_module,
            "_presence",
            lambda *_args, **_kwargs: (None, False, True, False),
        )
        monkeypatch.setattr(approval_module, "_unattended_contexts", lambda: [])
        result = approval_module.check_execute_code_guard(
            "print('late cancellation probe')", "local"
        )
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert result.get("status") != "pending_approval"
        assert notified == []
        assert key not in approval_module._gateway_queues
    finally:
        approval_module.unregister_gateway_notify(key)
        approval_context.reset_current_approval_resolver(resolver_token)
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)


def test_gateway_wait_rechecks_revocable_api_resolver_before_enqueuing(monkeypatch):
    """A late guard must not enqueue after cancellation already released the resolver."""
    from tools import approval_gateway_wait

    key = "revoked-before-gateway-wait"
    tokens = set_session_vars(platform="api_server", session_key=key)
    session_token = approval_context.set_current_session_key(key)
    resolver = approval_context.create_approval_resolver(key)
    resolver.activate()
    resolver_token = approval_context.set_current_approval_resolver(resolver)
    notified: list[dict] = []
    approval_module.register_gateway_notify(key, notified.append)
    monkeypatch.setattr(
        approval_gateway_wait, "_poll_event", lambda *_args, **_kwargs: "timeout"
    )
    try:
        _unregister_approval_notify(key, resolver)
        result = approval_gateway_wait._await_gateway_decision(
            key,
            notified.append,
            {"command": "late", "description": "late", "pattern_key": "execute_code"},
        )
        assert result["resolved"] is False
        assert result["reason"] == "approval resolver unavailable"
        assert notified == []
        assert key not in approval_module._gateway_queues
    finally:
        approval_module.unregister_gateway_notify(key)
        approval_context.reset_current_approval_resolver(resolver_token)
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)


def test_execute_code_entrypoint_propagates_run_resolver_to_kernel_worker(monkeypatch):
    """The real execute_code entry point retains the run capability across its worker hop."""
    from tools import code_execution_tool, code_kernel, terminal_tool, terminal_scope
    from tools import process_registry
    from tools.thread_context import propagate_context_to_thread

    key = "execute-code-worker-api-run"
    tokens = set_session_vars(platform="api_server", session_key=key)
    session_token = approval_context.set_current_session_key(key)
    resolver = approval_context.create_approval_resolver(key)
    resolver.activate()
    resolver_token = approval_context.set_current_approval_resolver(resolver)
    with approval_module._lock:
        approval_module._permanent_approved.discard("execute_code")
    seen: dict[str, bool] = {}

    def resolve_once(data: dict):
        approval_module.resolve_gateway_approval(
            key, "once", request_id=data["request_id"]
        )

    approval_module.register_gateway_notify(key, resolve_once)
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _config: False)
    monkeypatch.setattr(terminal_scope, "enforce_no_refusal", lambda: None)
    monkeypatch.setattr(
        process_registry, "_is_supervised_gateway_process", lambda: False
    )
    monkeypatch.setattr(
        code_execution_tool, "_load_config", lambda: {"timeout": 1, "max_tool_calls": 1}
    )
    monkeypatch.setattr(code_execution_tool, "_get_execution_mode", lambda: "project")

    def fake_kernel(_code, **_kwargs):
        seen["resolver"] = approval_context._has_current_approval_resolver()
        return "kernel-result"

    monkeypatch.setattr(code_kernel, "execute_in_session_kernel", fake_kernel)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                propagate_context_to_thread(
                    lambda: code_execution_tool.execute_code(
                        "print('worker approval probe')", task_id=key
                    )
                )
            ).result(timeout=5)
        assert result == "kernel-result"
        assert seen["resolver"] is True
    finally:
        approval_module.unregister_gateway_notify(key)
        approval_context.reset_current_approval_resolver(resolver_token)
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)


def test_hardline_floor_precedes_any_approval_transport():
    """A representative hardline command remains unconditionally blocked."""
    key = "hardline-api"
    tokens = set_session_vars(platform="api_server", session_key=key)
    session_token = approval_context.set_current_session_key(key)
    notified: list[dict] = []
    approval_module.register_gateway_notify(key, notified.append)
    try:
        result = approval_module.check_all_command_guards("rm -rf /", "local")
        assert result["approved"] is False
        assert result["hardline"] is True
        assert notified == []
        assert key not in approval_module._gateway_queues
    finally:
        approval_module.unregister_gateway_notify(key)
        approval_context.reset_current_session_key(session_token)
        clear_session_vars(tokens)
