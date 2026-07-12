"""Regression test for #63250: Discord "Always" approval should persist to command_allowlist.

The bug was in check_all_command_guards gateway path: save_permanent_allowlist()
was called inside the warnings loop, but should only be called once after
processing all keys (matching the CLI path behavior).
"""

import pytest
import threading
import time

from tools.approval import (
    check_all_command_guards,
    load_permanent_allowlist,
    _permanent_approved,
    _session_approved,
    set_current_session_key,
    reset_current_session_key,
)


class TestGatewayAlwaysPersistenceBug63250:
    """Test that gateway "always" approval correctly persists patterns."""

    def setup_method(self):
        """Reset approval state before each test."""
        _permanent_approved.clear()
        _session_approved.clear()
        load_permanent_allowlist()

    def test_gateway_always_persists_single_pattern(self, monkeypatch):
        """Test that a single dangerous pattern approved with "always" is persisted."""
        session_key = "test-session-gateway-always-single"

        # Mock gateway context
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        # Simulate a dangerous command (avoid hardline blocks)
        command = "rm -rf /tmp/test"

        # Thread that will resolve the approval
        def resolve_thread():
            # Wait a tiny bit for the approval to be queued
            time.sleep(0.01)
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_key, "always")

        # Mock the notify callback to start the resolve thread
        def mock_notify(approval_data):
            t = threading.Thread(target=resolve_thread)
            t.start()

        monkeypatch.setattr(
            "tools.approval._gateway_notify_cbs",
            {session_key: mock_notify},
        )

        token = set_current_session_key(session_key)
        try:
            result = check_all_command_guards(command, env_type="local")
        finally:
            reset_current_session_key(token)

        # Command should be approved
        assert result["approved"] is True

        # Pattern should be in permanent allowlist
        # The actual pattern key depends on the guard implementation
        # We verify that something was persisted
        assert len(_permanent_approved) > 0

    def test_gateway_always_persists_after_processing_all_keys(self, monkeypatch):
        """Test that save_permanent_allowlist is called once after processing all keys.

        This is the regression test for #63250: the bug was that save was called
        inside the loop, potentially multiple times and with inconsistent state.
        """
        session_key = "test-session-gateway-always-save-once"

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        # Use a command that triggers dangerous pattern but not hardline
        command = "rm -rf /tmp/test"

        # Track how many times save is called
        save_count = 0
        original_save = None

        def mock_save_permanent_allowlist(patterns):
            nonlocal save_count
            save_count += 1
            if original_save:
                original_save(patterns)

        # Get the original save function
        import tools.approval as approval_module
        original_save = approval_module.save_permanent_allowlist

        # Mock the save function
        monkeypatch.setattr(
            "tools.approval.save_permanent_allowlist",
            mock_save_permanent_allowlist,
        )

        # Thread that will resolve the approval
        def resolve_thread():
            time.sleep(0.01)
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_key, "always")

        def mock_notify(approval_data):
            t = threading.Thread(target=resolve_thread)
            t.start()

        monkeypatch.setattr(
            "tools.approval._gateway_notify_cbs",
            {session_key: mock_notify},
        )

        token = set_current_session_key(session_key)
        try:
            result = check_all_command_guards(command, env_type="local")
        finally:
            reset_current_session_key(token)

        assert result["approved"] is True

        # save_permanent_allowlist should be called exactly once after processing all keys
        # NOT inside the loop for each key
        assert save_count == 1

    def test_gateway_session_approval_does_not_save_permanent(self, monkeypatch):
        """Test that "session" scope does not trigger save_permanent_allowlist."""
        session_key = "test-session-gateway-session-no-save"

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        command = "rm -rf /tmp/test"
        initial_allowlist_count = len(_permanent_approved)

        # Track save calls
        save_count = 0

        def mock_save_permanent_allowlist(patterns):
            nonlocal save_count
            save_count += 1

        monkeypatch.setattr(
            "tools.approval.save_permanent_allowlist",
            mock_save_permanent_allowlist,
        )

        # Thread that will resolve the approval
        def resolve_thread():
            time.sleep(0.01)
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_key, "session")

        def mock_notify(approval_data):
            t = threading.Thread(target=resolve_thread)
            t.start()

        monkeypatch.setattr(
            "tools.approval._gateway_notify_cbs",
            {session_key: mock_notify},
        )

        token = set_current_session_key(session_key)
        try:
            result = check_all_command_guards(command, env_type="local")
        finally:
            reset_current_session_key(token)

        assert result["approved"] is True

        # save_permanent_allowlist should NOT be called for session scope
        assert save_count == 0

    def test_gateway_once_approval_does_not_save_permanent(self, monkeypatch):
        """Test that "once" scope does not trigger save_permanent_allowlist."""
        session_key = "test-session-gateway-once-no-save"

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        command = "rm -rf /tmp/test"

        # Track save calls
        save_count = 0

        def mock_save_permanent_allowlist(patterns):
            nonlocal save_count
            save_count += 1

        monkeypatch.setattr(
            "tools.approval.save_permanent_allowlist",
            mock_save_permanent_allowlist,
        )

        # Thread that will resolve the approval
        def resolve_thread():
            time.sleep(0.01)
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_key, "once")

        def mock_notify(approval_data):
            t = threading.Thread(target=resolve_thread)
            t.start()

        monkeypatch.setattr(
            "tools.approval._gateway_notify_cbs",
            {session_key: mock_notify},
        )

        token = set_current_session_key(session_key)
        try:
            result = check_all_command_guards(command, env_type="local")
        finally:
            reset_current_session_key(token)

        assert result["approved"] is True

        # save_permanent_allowlist should NOT be called for once scope
        assert save_count == 0