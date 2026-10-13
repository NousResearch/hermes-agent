"""Approval hook payloads must fall back to HERMES_SESSION_ID when no observability context is bound.

#137558: every approval fire site passes the routing ``session_key`` but not
``session_id``, while every other hook on the same plugin boundary hands plugins
``session_id``.  When no observability context was bound, the dispatch layer
omitted ``session_id`` entirely, so an observer plugin keying per-session state
on ``session_id`` could not attribute an approval event to any session — the
handler returned early and the event was lost with no error and no warning.

Contract: bound observability contextvar first, then the session-scoped
``HERMES_SESSION_ID`` (contextvar-first via ``gateway.session_context`` so
concurrent gateway turns resolve their own id), then omitted when no identity
exists at all (legacy behavior).  An explicitly passed ``session_id`` kwarg is
never clobbered.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools import approval_context


def _capture_hook(captured):
    def _invoke(hook_name, **kwargs):
        captured.append((hook_name, kwargs))
    return _invoke


@pytest.fixture(autouse=True)
def _no_ambient_session_id(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)


class TestApprovalHookSessionIdEnvFallback:
    def test_env_fallback_when_context_unbound(self, monkeypatch):
        """CLI/legacy paths without a bound observability context still identify the session."""
        monkeypatch.setenv("HERMES_SESSION_ID", "20261013_024646_envsrc")
        captured = []
        with patch("hermes_cli.lifecycle.invoke_hook", side_effect=_capture_hook(captured)):
            approval_context._fire_approval_hook(
                "pre_approval_request",
                command="rm -rf /tmp/hermes-tests",
                description="dangerous",
                session_key="routing-key",
                surface="cli",
            )
        assert captured, "hook must dispatch"
        _, kwargs = captured[0]
        assert kwargs.get("session_id") == "20261013_024646_envsrc"
        assert kwargs.get("session_key") == "routing-key"

    def test_bound_contextvar_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_ID", "env-session")
        captured = []
        tokens = approval_context.set_current_observability_context(session_id="ctx-session")
        try:
            with patch("hermes_cli.lifecycle.invoke_hook", side_effect=_capture_hook(captured)):
                approval_context._fire_approval_hook("post_approval_response", choice="approved")
        finally:
            approval_context.reset_current_observability_context(tokens)
        _, kwargs = captured[0]
        assert kwargs.get("session_id") == "ctx-session"

    def test_explicit_kwarg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_ID", "env-session")
        captured = []
        with patch("hermes_cli.lifecycle.invoke_hook", side_effect=_capture_hook(captured)):
            approval_context._fire_approval_hook(
                "pre_approval_request", session_id="explicit-session", command="x", description="y",
            )
        _, kwargs = captured[0]
        assert kwargs.get("session_id") == "explicit-session"

    def test_session_context_var_wins_over_process_env(self, monkeypatch):
        """Concurrent gateway turns: a session-scoped HERMES_SESSION_ID binding must not read
        another session's process-env value."""
        from gateway import session_context

        monkeypatch.setenv("HERMES_SESSION_ID", "first-session-env")
        captured = []
        # scoped_current_session_id binds task-local only (never touches os.environ),
        # mirroring how a second gateway turn overlays its own id.
        with session_context.scoped_current_session_id("second-session-ctx"):
            with patch("hermes_cli.lifecycle.invoke_hook", side_effect=_capture_hook(captured)):
                approval_context._fire_approval_hook(
                    "pre_approval_request", command="x", description="y", session_key="second-session-ctx",
                )
        _, kwargs = captured[0]
        assert kwargs.get("session_id") == "second-session-ctx"

    def test_absent_when_no_identity_anywhere(self):
        """No bound context and no HERMES_SESSION_ID: omit the key (legacy behavior —
        the observer's own fallback owns that decision)."""
        captured = []
        with patch("hermes_cli.lifecycle.invoke_hook", side_effect=_capture_hook(captured)):
            approval_context._fire_approval_hook(
                "pre_approval_request", command="x", description="y", session_key="k",
            )
        _, kwargs = captured[0]
        assert "session_id" not in kwargs
