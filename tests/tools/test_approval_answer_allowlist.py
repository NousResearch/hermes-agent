"""Only once/session/always count as consent.

The approval gate used to be a deny-list: it refused "deny", ``None`` and "timeout" and treated
every other value as a yes. The local CLI prompt is safe either way because
``prompt_dangerous_approval`` normalizes keystrokes through a lookup table, but two paths hand
their answer straight through to ``grant``:

* the gateway round-trip (``_await_gateway_decision``), an inter-process message, and
* third-party approval transport plugins.

So a client that starts sending ``{"choice": "approve"}`` after an upgrade, a truncated payload, or
a plugin bug read as consent and the command ran.

This test pins the allow-list in ``tools.approval._human_decision.grant``.
"""

from __future__ import annotations

import pytest

import tools.approval as approval_module
from tools import approval_context
from tools.approval import check_all_command_guards
from tools.terminal_tool import set_approval_callback


DANGEROUS = "rm -rf /tmp/approval-allowlist-testdir"

# Values a gateway client or transport plugin could plausibly send that are NOT consent. Every one
# of these executed the command while the gate was a deny-list.
NOT_CONSENT = [
    "yes",
    "ok",
    "true",
    "accept",
    "",             # empty payload
    "null",
    "1",
    "denied",       # near-miss on the refusal word
]

@pytest.fixture(autouse=True)
def _clean_approval_env(monkeypatch):
    """Neutral approval environment: no yolo, manual mode, tirith quiet, no leftover grants."""
    for key in ("HERMES_EXEC_ASK", "HERMES_GATEWAY_SESSION", "HERMES_SESSION_PLATFORM",
                "HERMES_CRON_SESSION", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
    )
    approval_module._session_approved.clear()
    approval_module._permanent_approved.clear()
    approval_module._pending.clear()
    approval_module._denial_tally.clear()
    set_approval_callback(None)
    yield
    approval_module._session_approved.clear()
    approval_module._permanent_approved.clear()
    approval_module._pending.clear()
    approval_module._denial_tally.clear()
    set_approval_callback(None)


def _drive_gateway(monkeypatch, answer):
    """Run the command gate through the gateway branch with ``answer`` as the user's reply."""
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval_module, "_gateway_notify_cb", lambda _sk: (lambda *a, **k: True))
    monkeypatch.setattr(
        approval_module, "_await_gateway_decision",
        lambda *a, **k: {"resolved": True, "choice": answer, "reason": None},
    )
    return check_all_command_guards(DANGEROUS, "local")


class TestGatewayAnswersMustBeConsentWords:
    @pytest.mark.parametrize("answer", NOT_CONSENT)
    def test_unrecognized_gateway_answer_is_refused(self, monkeypatch, answer):
        result = _drive_gateway(monkeypatch, answer)

        assert result.get("approved") is False, f"{answer!r} was treated as consent"
        assert result.get("outcome") == "unrecognized_answer"
        assert result.get("user_consent") is False
        assert "not a recognized decision" in (result.get("message") or "")
