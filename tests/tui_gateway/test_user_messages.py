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
