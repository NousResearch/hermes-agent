"""Tests for tools.local_toast — the loopback antenna toast dispatcher.

Covers:
- fires-once: mocked urllib captures the POST body when notify_input_needed is called
- coalesces: repeat calls in the coalesce window collapse to a single dispatch
- kill-switch: HERMES_LOCAL_TOAST_DISABLE=1 makes the call a no-op
- fail-open: a broken antenna (URLError/timeout/etc.) never raises to caller
- persona resolution: HERMES_ACTIVE_PROFILE wins, HERMES_PROFILE fallback, "hermes" last
- endpoint override: HERMES_LOCAL_TOAST_URL is honored

The dispatch runs on a daemon thread; tests join it via a short timeout on a
threading.Event captured in the mocked _post_toast.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
from unittest.mock import patch

import pytest

import tools.local_toast as local_toast


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Clear coalesce state + env var overrides between tests."""
    # Ensure kill-switch and URL override are clean unless the test sets them.
    monkeypatch.delenv("HERMES_LOCAL_TOAST_DISABLE", raising=False)
    monkeypatch.delenv("HERMES_LOCAL_TOAST_URL", raising=False)
    monkeypatch.delenv("HERMES_ACTIVE_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    with local_toast._lock:
        local_toast._last_fired.clear()
    yield
    with local_toast._lock:
        local_toast._last_fired.clear()


def _wait_for_dispatch(mock_post, expected: int = 1, timeout: float = 2.0) -> None:
    """Poll until mock_post has been called `expected` times or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mock_post.call_count >= expected:
            return
        time.sleep(0.02)
    # let final call quiesce
    time.sleep(0.05)


class TestFiresOnce:
    def test_notify_dispatches_post(self, monkeypatch):
        """A single notify_input_needed call should invoke the POST helper once."""
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "test-persona")
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed("clarify", detail="which color?")
            _wait_for_dispatch(mock_post, expected=1)
            assert mock_post.call_count == 1
            title, message, level = mock_post.call_args.args
            assert title == "test-persona: needs input"
            assert "which color?" in message
            assert level == "info"

    def test_notify_carries_approval_level_warn(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "test-persona")
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed(
                "approval", detail="rm -rf /tmp/foo", level="warn"
            )
            _wait_for_dispatch(mock_post, expected=1)
            title, message, level = mock_post.call_args.args
            assert level == "warn"
            assert "Command awaiting approval" in message
            assert "rm -rf /tmp/foo" in message

    def test_detail_truncated_at_200_chars(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "p")
        with patch.object(local_toast, "_post_toast") as mock_post:
            long_detail = "x" * 500
            local_toast.notify_input_needed("clarify", detail=long_detail)
            _wait_for_dispatch(mock_post, expected=1)
            _, message, _ = mock_post.call_args.args
            # The message contains prefix + colon + <=200 chars of detail
            assert "x" * 200 in message
            assert "x" * 201 not in message

    def test_detail_newlines_stripped(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "p")
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed("clarify", detail="line1\nline2\nline3")
            _wait_for_dispatch(mock_post, expected=1)
            _, message, _ = mock_post.call_args.args
            assert "\n" not in message
            assert "line1 line2 line3" in message


class TestCoalesce:
    def test_second_call_within_window_is_dropped(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "p")
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed("clarify", detail="first")
            local_toast.notify_input_needed("clarify", detail="second")
            _wait_for_dispatch(mock_post, expected=1, timeout=0.5)
            # Only ONE dispatch — the second was coalesced.
            assert mock_post.call_count == 1
            _, first_msg, _ = mock_post.call_args.args
            assert "first" in first_msg

    def test_different_kinds_do_not_coalesce(self, monkeypatch):
        """Coalesce key is (persona, kind) — different kinds fire independently."""
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "p")
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed("clarify", detail="q")
            local_toast.notify_input_needed("approval", detail="rm -rf")
            _wait_for_dispatch(mock_post, expected=2)
            assert mock_post.call_count == 2

    def test_call_after_window_expiry_fires_again(self, monkeypatch):
        """After the coalesce window elapses the same key can fire again."""
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "p")
        # Squash the window so we don't sleep in CI.
        monkeypatch.setattr(local_toast, "_COALESCE_WINDOW_SEC", 0.05)
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed("clarify", detail="first")
            _wait_for_dispatch(mock_post, expected=1)
            time.sleep(0.1)  # exceed window
            local_toast.notify_input_needed("clarify", detail="second")
            _wait_for_dispatch(mock_post, expected=2)
            assert mock_post.call_count == 2


class TestDisabled:
    def test_env_kill_switch_prevents_dispatch(self, monkeypatch):
        monkeypatch.setenv("HERMES_LOCAL_TOAST_DISABLE", "1")
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed("clarify", detail="quiet")
            _wait_for_dispatch(mock_post, expected=1, timeout=0.3)
            assert mock_post.call_count == 0

    def test_blank_env_var_does_not_disable(self, monkeypatch):
        """Only a non-empty value counts as a kill-switch."""
        monkeypatch.setenv("HERMES_LOCAL_TOAST_DISABLE", "")
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "p")
        with patch.object(local_toast, "_post_toast") as mock_post:
            local_toast.notify_input_needed("clarify", detail="noisy")
            _wait_for_dispatch(mock_post, expected=1)
            assert mock_post.call_count == 1


class TestFailOpen:
    def test_urlopen_urlerror_never_raises(self, monkeypatch):
        """A dead antenna must be swallowed inside _post_toast."""
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "p")

        def _boom(*a, **kw):
            raise urllib.error.URLError("connection refused")

        with patch.object(local_toast.urllib.request, "urlopen", side_effect=_boom):
            # Call the raw helper directly — must NOT raise.
            local_toast._post_toast("title", "message", "info")

    def test_notify_input_needed_swallows_persona_lookup_failure(self, monkeypatch):
        """Even if _resolve_persona blows up, notify_input_needed returns None."""
        def _boom():
            raise RuntimeError("no profile")
        monkeypatch.setattr(local_toast, "_resolve_persona", _boom)
        # Must NOT raise
        result = local_toast.notify_input_needed("clarify", detail="x")
        assert result is None

    def test_endpoint_override_env_var(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_LOCAL_TOAST_URL",
            "http://127.0.0.1:9999/custom",
        )
        assert local_toast._endpoint() == "http://127.0.0.1:9999/custom"


class TestPersonaResolution:
    def test_active_profile_env_wins(self, monkeypatch):
        monkeypatch.setenv("HERMES_ACTIVE_PROFILE", "daddy")
        monkeypatch.setenv("HERMES_PROFILE", "legacy-value")
        assert local_toast._resolve_persona() == "daddy"

    def test_legacy_profile_env_fallback(self, monkeypatch):
        monkeypatch.delenv("HERMES_ACTIVE_PROFILE", raising=False)
        # Force the hermes_cli.profiles branch to fail cleanly.
        import sys
        # Even if the module exists, it won't return a value when nothing is set.
        # If it does return something we still fall through to legacy env.
        monkeypatch.setenv("HERMES_PROFILE", "legacy-persona")
        # Patch the intermediate resolver to force fall-through.
        try:
            from hermes_cli import profiles as _p
            monkeypatch.setattr(_p, "get_active_profile_name", lambda: "")
        except Exception:
            pass
        assert local_toast._resolve_persona() == "legacy-persona"

    def test_default_fallback_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("HERMES_ACTIVE_PROFILE", raising=False)
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        try:
            from hermes_cli import profiles as _p
            monkeypatch.setattr(_p, "get_active_profile_name", lambda: "")
        except Exception:
            pass
        assert local_toast._resolve_persona() == "hermes"


class TestPostBodyShape:
    def test_post_body_is_valid_json_with_expected_fields(self, monkeypatch):
        """Capture the raw HTTP request payload built by _post_toast."""
        captured = {}

        class _FakeResp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def read(self_inner):
                return b'{"ok":true}'

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["body"] = req.data
            captured["content_type"] = req.get_header("Content-type")
            return _FakeResp()

        monkeypatch.setattr(local_toast.urllib.request, "urlopen", _fake_urlopen)
        local_toast._post_toast("t", "m", "info")

        assert captured["url"] == local_toast._DEFAULT_URL
        assert captured["method"] == "POST"
        assert captured["content_type"] == "application/json"
        body = json.loads(captured["body"].decode("utf-8"))
        assert body == {"title": "t", "message": "m", "level": "info"}
