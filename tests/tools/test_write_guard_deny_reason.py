"""Deny-with-reason relay through the protected-instruction write gate.

A deny that carries the user's free text — the CLI panel's ``Denied`` choice
(``== "deny"``, ``.reason`` set) on the CLI surface, ``reason`` in the gateway
decision on the gateway surface — must reach the BLOCKED message the gate
returns, so the agent can adapt instead of only hearing "denied". Same addendum
shape as ``tools/approval.py``'s ``reason_addendum`` (parity with the gateway's
``/deny <reason>``, which relays its free text the same way).
"""

from tools.approval_prompt import Denied
from tools.file_tools_write_guards import _check_protected_instruction_write

TARGET = "/probe/.hermes/config.yaml"
ADDENDUM = ' Reason given by the user: "{}".'


def _gate_on(monkeypatch):
    """Pin the gate's config: a read failure already fails safe to enabled,
    but pinning keeps the test independent of the developer's config.yaml."""
    monkeypatch.setattr("tools.file_tools_write_guards._protected_instruction_config",
                        lambda: (True, []))


def test_gateway_deny_reason_reaches_blocked_message(monkeypatch):
    """A gateway decision carrying a reason appends it to the denied message."""
    from tools import approval as _approval
    from tools.approval_context import get_current_session_key

    _gate_on(monkeypatch)
    session_key = get_current_session_key()
    monkeypatch.setattr(_approval, "_gateway_notify_cbs", {session_key: lambda data: None})
    monkeypatch.setattr("tools.approval_gateway_wait._await_gateway_decision",
                        lambda key, cb, data, surface="gateway":
                        {"resolved": True, "choice": "deny", "reason": "not in prod"})
    result = _check_protected_instruction_write([TARGET])
    assert result is not None, "the write must be denied, not approved"
    assert "was denied by the user." in result
    assert ADDENDUM.format("not in prod") in result


def test_cli_deny_reason_reaches_blocked_message(monkeypatch):
    """A CLI panel deny with reason (Denied choice) appends the reason."""
    from tools import approval as _approval

    _gate_on(monkeypatch)
    monkeypatch.setattr(_approval, "_gateway_notify_cbs", {})  # no gateway surface: CLI branch
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback",
                        lambda: (lambda *args, **kwargs: "deny"))
    monkeypatch.setattr("tools.approval_context._no_user_can_answer", lambda: False)
    monkeypatch.setattr("tools.approval_prompt.prompt_dangerous_approval",
                        lambda command, description, **kwargs: Denied("too risky"))
    result = _check_protected_instruction_write([TARGET])
    assert result is not None, "the write must be denied, not approved"
    assert "was denied by the user." in result
    assert ADDENDUM.format("too risky") in result
