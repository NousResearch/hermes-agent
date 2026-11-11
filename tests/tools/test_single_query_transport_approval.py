"""Single-query (-q) workers: an explicitly selected plugin approval transport answers the
dangerous-command and execute_code gates before ``approvals.single_query_mode`` resolves them.
Builtin transport, cron, never-presented transports (unavailable/declined) and non-dangerous
commands keep the old behavior; any other transport failure is denied."""

import pytest

import tools.approval as approval_module
from gateway.session_context import reset_session_vars
from tools import approval_context, approval_prompt
from tools.approval import check_all_command_guards, check_execute_code_guard

SESSION = "sq-transport-session"
DANGEROUS = "rm -rf /tmp/sq-transport-fixture"


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    approval_module._permanent_approved.clear()
    approval_module.clear_session(SESSION)
    approval_module._denial_tally.pop(SESSION, None)
    reset_session_vars()
    for var in ("HERMES_CRON_SESSION", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_context, "_get_approval_transport_timeout", lambda: 5, raising=False)
    token = approval_context.set_current_session_key(SESSION)
    yield
    approval_context.reset_current_session_key(token)
    approval_module._permanent_approved.clear()
    approval_module.clear_session(SESSION)
    reset_session_vars()


def _select(monkeypatch, present, name="phone"):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    manifest = PluginManifest(name="sq-approval", version="1.0.0", description="fixture",
                              source="user", key="sq-approval")
    PluginContext(manifest, manager).register_approval_transport(name, present)
    monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: (name, "builtin"))


def _answer(choice, seen):
    return lambda request: seen.append(request) or request.respond(choice)


def test_transport_once_approves_dangerous_command(monkeypatch):
    seen = []
    _select(monkeypatch, _answer("once", seen))
    result = check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is True
    assert len(seen) == 1
    assert seen[0].surface == "cli"


def test_transport_deny_blocks(monkeypatch):
    seen = []
    _select(monkeypatch, _answer("deny", seen))
    result = check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is False
    assert "selected approval transport" in result["message"]
    assert result["outcome"] == "denied"
    assert "failed" not in result["message"]
    assert len(seen) == 1


def _declines(seen):
    from hermes_cli.approval_transport import ApprovalTransportDeclined

    def present(request):
        seen.append(request)
        raise ApprovalTransportDeclined("not an autonomy worker")
    return present


def test_transport_decline_falls_back_to_single_query_mode(monkeypatch):
    seen = []
    _select(monkeypatch, _declines(seen))
    result = check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is False
    assert "single-query" in result["message"]
    assert len(seen) == 1


def test_transport_decline_keeps_configured_single_query_mode(monkeypatch):
    _select(monkeypatch, _declines([]))
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "approve")
    assert check_all_command_guards(DANGEROUS, "local")["approved"] is True
    assert check_execute_code_guard("print(1)", "local")["approved"] is True


def test_unavailable_transport_keeps_configured_single_query_mode(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("missing", None))
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "approve")
    assert check_all_command_guards(DANGEROUS, "local")["approved"] is True


def test_transport_resolution_error_is_denied_even_in_approve_mode(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("phone", None))
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "approve")

    def fail_resolution():
        raise RuntimeError("plugin discovery failed")

    monkeypatch.setattr(approval_prompt, "get_plugin_manager", fail_resolution)
    result = check_all_command_guards(DANGEROUS, "local")

    assert result["approved"] is False
    assert result["outcome"] == "transport_error"
    assert "Selected approval transport failed (error)" in result["message"]
    assert approval_module._denial_tally.get(SESSION) == 1


def _assert_denied_in_approve_mode(monkeypatch, failure):
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "approve")
    for result in (check_all_command_guards(DANGEROUS, "local"), check_execute_code_guard("print(1)", "local")):
        assert result["approved"] is False
        assert f"Selected approval transport failed ({failure})" in result["message"]
        assert "NOT consented" in result["message"]
        assert result["outcome"] == f"transport_{failure}"
    assert not approval_module.is_approved(SESSION, "execute_code")
    assert approval_module._denial_tally.get(SESSION, 0) >= 2


def test_error_after_send_is_denied_even_in_approve_mode(monkeypatch):
    """A generic exception may follow delivery to the phone: silence is not consent."""
    sent = []

    def sends_then_raises(request):
        sent.append(request.request_id)
        raise RuntimeError("connection reset after send")

    _select(monkeypatch, sends_then_raises)
    _assert_denied_in_approve_mode(monkeypatch, "error")
    assert len(sent) == 2


def test_slow_transport_times_out_and_is_denied_even_in_approve_mode(monkeypatch):
    import threading

    release = threading.Event()
    monkeypatch.setattr(approval_context, "_get_approval_transport_timeout", lambda: 0.2, raising=False)
    _select(monkeypatch, lambda request: release.wait(5) and request.respond("once"))
    try:
        _assert_denied_in_approve_mode(monkeypatch, "timeout")
    finally:
        release.set()


def test_busy_transport_is_denied_even_in_approve_mode(monkeypatch):
    import threading

    from hermes_cli import approval_transport

    seen = []
    monkeypatch.setattr(approval_transport, "_transport_worker_slots", threading.BoundedSemaphore(1))
    approval_transport._transport_worker_slots.acquire()
    _select(monkeypatch, _answer("once", seen))
    _assert_denied_in_approve_mode(monkeypatch, "busy")
    assert seen == []


def test_request_build_failure_is_denied_without_invoking_transport(monkeypatch):
    from agent import redact

    seen = []
    _select(monkeypatch, _answer("once", seen))
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "approve")

    def fail_redaction(*args, **kwargs):
        raise RuntimeError("redaction unavailable")

    monkeypatch.setattr(redact, "redact_sensitive_text", fail_redaction)
    result = check_all_command_guards(DANGEROUS, "local")

    assert result["approved"] is False
    assert result["outcome"] == "transport_error"
    assert "Selected approval transport failed (error)" in result["message"]
    assert approval_module._denial_tally.get(SESSION) == 1
    assert seen == []


@pytest.mark.parametrize("failure", ["interrupted", "stale", "invalid"])
def test_presented_but_unanswered_is_denied_even_in_approve_mode(monkeypatch, failure):
    _select(monkeypatch, _answer("once", []))
    monkeypatch.setattr(approval_module, "_present_with_selected_transport", lambda **kw: {
        "selected": True, "choice": "deny", "failure": failure, "fallback": None, "name": "phone"})
    _assert_denied_in_approve_mode(monkeypatch, failure)


def test_stale_answer_from_real_transport_is_denied(monkeypatch):
    from hermes_cli.approval_transport import ApprovalDecision

    _select(monkeypatch, lambda request: ApprovalDecision("other", request.digest, "once"))
    _assert_denied_in_approve_mode(monkeypatch, "stale")


def test_decline_outside_single_query_keeps_existing_transport_failure_handling(monkeypatch):
    """Non -q gates treat ``declined`` exactly like any other transport failure."""
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    _select(monkeypatch, _declines([]))
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("phone", None))
    attempt = approval_prompt._present_with_selected_transport(
        command=DANGEROUS, description="d", pattern_key="k", pattern_keys=["k"], session_key=SESSION,
        surface="cli", allow_session=True, allow_permanent=True)
    assert attempt["failure"] == "declined"
    choice, denied = approval_prompt._transport_choice(attempt, pattern_key="k", description="d")
    assert choice is None and denied["approved"] is False and denied["outcome"] == "transport_declined"
    choice, denied = approval_prompt._transport_choice({**attempt, "fallback": "builtin"},
                                                       pattern_key="k", description="d")
    assert choice is None and denied is None


def test_builtin_transport_keeps_single_query_deny(monkeypatch):
    calls = []
    monkeypatch.setattr(approval_module, "_present_with_selected_transport",
                        lambda **kw: calls.append(kw) or {"selected": False})
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("builtin", None))
    result = check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is False
    assert "single-query" in result["message"]
    assert calls == []


def test_cron_context_never_presents(monkeypatch):
    seen = []
    _select(monkeypatch, _answer("once", seen))
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny", raising=False)
    result = check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is False
    assert seen == []


def test_non_dangerous_command_never_presents(monkeypatch):
    seen = []
    _select(monkeypatch, _answer("deny", seen))
    assert check_all_command_guards("ls -la", "local")["approved"] is True
    assert seen == []


def test_execute_code_uses_transport(monkeypatch):
    seen = []
    _select(monkeypatch, _answer("once", seen))
    assert check_execute_code_guard("print(1)", "local")["approved"] is True
    assert len(seen) == 1 and seen[0].pattern_key == "execute_code"

    denied = []
    _select(monkeypatch, _answer("deny", denied))
    result = check_execute_code_guard("print(2)", "local")
    assert result["approved"] is False
    assert "selected approval transport" in result["message"]
    assert len(denied) == 1


def test_session_choice_persists(monkeypatch):
    seen = []
    _select(monkeypatch, _answer("session", seen))
    assert check_all_command_guards(DANGEROUS, "local")["approved"] is True
    assert check_all_command_guards(DANGEROUS, "local")["approved"] is True
    assert len(seen) == 1
