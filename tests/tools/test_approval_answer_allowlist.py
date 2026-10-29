"""Only once/session/always count as consent.

The approval gate used to be a deny-list: it refused "deny", ``None`` and "timeout" and treated
every other value as a yes. The local CLI prompt is safe either way because
``prompt_dangerous_approval`` normalizes keystrokes through a lookup table, but two paths hand
their answer straight through to ``grant``:

* the gateway round-trip (``_await_gateway_decision``), an inter-process message, and
* third-party approval transport plugins.

So a client that starts sending ``{"choice": "approve"}`` after an upgrade, a truncated payload, or
a plugin bug read as consent and the command ran.

These tests pin the allow-list in ``tools.approval._human_decision.grant`` and the boundary
normalization in ``_canonical_choice`` that keeps real platform answers working.
"""

from __future__ import annotations

import pytest

import tools.approval as approval_module
from tools import approval_context
from tools.approval import CONSENT_ANSWERS, check_all_command_guards
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

# Spellings normalized onto the canonical vocabulary BEFORE the allow-list runs, rather than
# widening it. Case and whitespace are cosmetic. The affirmative aliases are the set the API server
# already applies (``gateway/platforms/api_server_runs.py::_APPROVAL_CHOICE_ALIASES``); the
# WhatsApp Cloud Approve button sends "approve" verbatim.
ALIASES_TO_CANONICAL = {
    "ONCE": "once",
    "once ": "once",
    "  Session  ": "session",
    "ALWAYS": "always",
    "approve": "once",      # WhatsApp Cloud button payload
    "Approve": "once",
    "approved": "once",
    "allow": "once",
}


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

    @pytest.mark.parametrize("answer", sorted(CONSENT_ANSWERS))
    def test_the_three_consent_words_still_approve(self, monkeypatch, answer):
        result = _drive_gateway(monkeypatch, answer)

        assert result.get("approved") is True, f"{answer!r} should still approve"
        assert result.get("user_approved") is True

    def test_deny_still_reports_denied_not_unrecognized(self, monkeypatch):
        """"deny" has its own branch ahead of grant(); it keeps its own outcome label."""
        result = _drive_gateway(monkeypatch, "deny")

        assert result.get("approved") is False
        assert result.get("outcome") == "denied"

    def test_unrecognized_answer_grants_nothing_for_later_calls(self, monkeypatch):
        """A refused answer must not leave a grant behind that approves the next call."""
        _drive_gateway(monkeypatch, "always")  # the word that WOULD persist, spelled correctly
        approval_module._session_approved.clear()
        approval_module._permanent_approved.clear()

        _drive_gateway(monkeypatch, "yes")
        assert not approval_module._session_approved, "an unrecognized answer left a session grant"
        assert not approval_module._permanent_approved, "an unrecognized answer left a permanent grant"


class TestAliasesNormalizeRatherThanWidenTheAllowList:
    """Normalization happens before the allow-list, so the allow-list itself stays exact."""

    @pytest.mark.parametrize("sent,canonical", sorted(ALIASES_TO_CANONICAL.items()))
    def test_alias_is_accepted(self, monkeypatch, sent, canonical):
        result = _drive_gateway(monkeypatch, sent)

        assert result.get("approved") is True, f"{sent!r} should normalize to {canonical!r}"
        assert result.get("user_approved") is True

    def test_whatsapp_approve_tap_still_works(self, monkeypatch):
        """The WhatsApp Cloud adapter builds an ``appr:<id>:approve`` button payload and hands the
        literal word "approve" to resolve_gateway_approval, which stores it verbatim as the waiting
        agent's choice. An allow-list with no normalization refuses it — and the adapter still
        replies "✅ Approved." to the user, so a refused command looks approved.
        """
        result = _drive_gateway(monkeypatch, "approve")

        assert result.get("approved") is True
        assert result.get("outcome") != "unrecognized_answer"

    def test_approve_maps_to_the_narrowest_scope(self, monkeypatch):
        """An Approve tap is ONE operation. It must never persist a session or permanent grant."""
        _drive_gateway(monkeypatch, "approve")

        assert not any(approval_module._session_approved.values()), \
            "'approve' persisted a session grant — it must map to 'once', not 'session'"
        assert not approval_module._permanent_approved, "'approve' persisted a permanent grant"

    def test_session_spelling_variant_still_persists(self, monkeypatch):
        """The mirror of the test above: a real 'session' answer must still persist."""
        _drive_gateway(monkeypatch, "  Session  ")

        assert any(approval_module._session_approved.values()), \
            "a normalized 'session' answer did not persist a session grant"


class TestCliCallbackAnswersMustBeConsentWords:
    """The interactive prompt normalizes keystrokes, but the callback seam it sits behind does
    not — an embedding client or plugin supplies that function directly."""

    def test_unrecognized_callback_answer_is_refused(self, monkeypatch):
        set_approval_callback(lambda command, description, **kwargs: "yes")

        result = check_all_command_guards(DANGEROUS, "local")

        assert result.get("approved") is False
        assert result.get("outcome") == "unrecognized_answer"

    def test_none_from_callback_is_refused(self, monkeypatch):
        """``None`` reaches grant() unguarded on the CLI path; it used to read as a yes."""
        set_approval_callback(lambda command, description, **kwargs: None)

        result = check_all_command_guards(DANGEROUS, "local")

        assert result.get("approved") is False
        assert result.get("outcome") == "unrecognized_answer"

    def test_once_from_callback_still_approves(self, monkeypatch):
        set_approval_callback(lambda command, description, **kwargs: "once")

        result = check_all_command_guards(DANGEROUS, "local")

        assert result.get("approved") is True
