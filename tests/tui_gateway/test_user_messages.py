"""User-facing copy for assistant-start failures must match the failure's actual cause.

When init dies waiting for a cross-process auth lock — the profile auth-store lock or the shared
Nous store lock, both on the ``resolve_nous_access_token`` init path (#124533) — the cause is
contention with another hermes process (a dashboard or a slow credential refresh), so the generic
/model / `hermes setup` hints would send the user re-checking credentials that are fine.
"""

from __future__ import annotations

import pytest

from tui_gateway.user_messages import agent_init_failed_message


@pytest.mark.parametrize("exc_text", [
    "Timed out waiting for auth store lock (/home/u/.hermes/profiles/coder/auth.lock); "
    "another hermes process (pid 4242) probably still holds it "
    "(e.g. a dashboard or a slow credential refresh)",
    "Timed out waiting for auth store lock (/home/u/.hermes/profiles/coder/auth.lock)",
    "Timed out waiting for shared Nous auth lock (/home/u/.hermes/shared/nous.lock)",
], ids=["auth-store-with-holder", "auth-store-no-holder", "shared-nous-store"])
def test_auth_lock_timeout_contention_gets_the_wait_copy(exc_text):
    message = agent_init_failed_message(TimeoutError(exc_text))
    assert "/model" not in message
    assert "hermes setup" not in message
    assert "lock" in message and "dashboard" in message  # actionable: what holds it, what to do


def test_generic_timeout_keeps_the_model_setup_hints():
    # A TimeoutError that is NOT an auth lock (e.g. a network connect timeout) must not
    # be misread as lock contention.
    message = agent_init_failed_message(TimeoutError("connect timed out"))
    assert "/model" in message and "hermes setup" in message


def test_other_init_failures_keep_the_model_setup_hints():
    message = agent_init_failed_message(RuntimeError("provider bootstrap failed"))
    assert "/model" in message and "hermes setup" in message


# --- Content-policy copy must offer the new-session way out (issue #132504) ------------------

def _content_policy_ts_surface_copy():
    """Desktop and terminal-TUI English copy for the content-policy block, read from
    their i18n sources so the invariant covers the TypeScript surfaces too."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    desktop = (root / "apps/desktop/src/i18n/en.ts").read_text(encoding="utf-8")
    tui = (root / "ui-tui/src/i18n/en/userMessages.ts").read_text(encoding="utf-8")
    desktop_match = re.search(r"content_policy_blocked: \{.*?body: provider => `(.*?)`", desktop, re.DOTALL)
    tui_match = re.search(r"contentPolicyBlocked: \{.*?hint: '(.*?)'", tui, re.DOTALL)
    assert desktop_match is not None, "content_policy_blocked body not found in desktop en.ts"
    assert tui_match is not None, "contentPolicyBlocked hint not found in ui-tui userMessages.ts"
    return desktop_match.group(1), tui_match.group(1)


def test_content_policy_copy_points_to_new_session_not_rewording():
    """The blocked trigger often sits in an EARLIER tool result (a skill_view/read_file/
    session_search page that tripped the provider's guardrail) that is re-sent with every
    turn, so rewording the latest message cannot clear the block. Every surface's copy must
    name the real exits — a new session (/new, or a new chat on Desktop) or another model —
    and must not lead with reword/edit-again advice (#132504)."""
    from agent.turn_failure_copy import CONTENT_POLICY_NEXT_STEPS
    from tui_gateway.user_messages import turn_error_hint

    surface = {"code": "content_policy_blocked", "provider": "openrouter", "retryable": False}
    desktop_body, tui_hint = _content_policy_ts_surface_copy()
    surfaces = (
        ("agent CONTENT_POLICY_NEXT_STEPS", CONTENT_POLICY_NEXT_STEPS),
        ("tui_gateway turn_error_hint", turn_error_hint(surface)),
        ("desktop en.ts body", desktop_body),
        ("ui-tui en hint", tui_hint),
    )
    for label, text in surfaces:
        lowered = text.lower()
        # (a) the new-session way out is named, in this surface's own vocabulary
        assert "/new" in text or "new chat" in lowered or "new session" in lowered, label
        # (b) rewording is not the advice: a reword cannot clear a trigger that lives in
        # an earlier tool result and is re-sent every turn
        assert not any(
            w in lowered for w in ("reword", "rephrase", "edit it", "edit your message", "edit the message")
        ), label
