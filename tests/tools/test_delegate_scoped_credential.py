"""Tests for ScopedCredentialView -- credential isolation for subagents.

Verifies that child agents receive a restricted view of the credential pool
that only exposes the single leased credential, blocking pool-wide iteration,
status resets, and credential mutations.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import ScopedCredentialView, _run_single_child


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _make_mock_pool_entry(entry_id="cred-a", label="test-key"):
    """Create a mock PooledCredential entry."""
    entry = MagicMock()
    entry.id = entry_id
    entry.label = label
    entry.runtime_api_key = f"sk-{entry_id}"
    entry.runtime_base_url = "https://example.com/v1"
    return entry


class TestScopedCredentialViewSelect:
    """select() returns only the leased credential."""

    def test_select_returns_leased_entry(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        result = view.select()
        assert result is entry

    def test_current_returns_leased_entry(self):
        entry = _make_mock_pool_entry("cred-b")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        result = view.current()
        assert result is entry

    def test_peek_returns_leased_entry(self):
        entry = _make_mock_pool_entry("cred-c")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        result = view.peek()
        assert result is entry


class TestScopedCredentialViewIteration:
    """Pool-wide iteration is blocked -- only the leased entry is visible."""

    def test_entries_returns_single_entry_list(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        result = view.entries()
        assert result == [entry]
        assert len(result) == 1

    def test_entries_with_none_entry_returns_empty(self):
        pool = MagicMock()
        view = ScopedCredentialView(None, pool)

        result = view.entries()
        assert result == []

    def test_has_credentials_true_when_entry_present(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        assert view.has_credentials() is True

    def test_has_credentials_false_when_no_entry(self):
        pool = MagicMock()
        view = ScopedCredentialView(None, pool)

        assert view.has_credentials() is False

    def test_has_available_true_when_entry_present(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        assert view.has_available() is True

    def test_has_available_false_when_no_entry(self):
        pool = MagicMock()
        view = ScopedCredentialView(None, pool)

        assert view.has_available() is False


class TestScopedCredentialViewRotation:
    """rotate() is a no-op that returns the same entry."""

    def test_rotate_returns_same_entry(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        result = view.rotate()
        assert result is entry

    def test_mark_exhausted_and_rotate_returns_same_entry(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        result = view.mark_exhausted_and_rotate(status_code=429)
        assert result is entry

    def test_rotate_does_not_call_pool(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        view.rotate()
        pool.mark_exhausted_and_rotate.assert_not_called()
        pool.select.assert_not_called()


class TestScopedCredentialViewBlocked:
    """Pool-wide mutation operations are blocked."""

    def test_reset_statuses_raises(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        with pytest.raises(PermissionError, match="statuses"):
            view.reset_statuses()

    def test_remove_index_raises(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        with pytest.raises(PermissionError, match="remove"):
            view.remove_index(1)

    def test_add_entry_raises(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        with pytest.raises(PermissionError, match="add"):
            view.add_entry(MagicMock())


class TestScopedCredentialViewLease:
    """Lease operations delegate to the underlying pool."""

    def test_acquire_lease_delegates_to_pool(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        pool.acquire_lease.return_value = "cred-a"
        view = ScopedCredentialView(entry, pool)

        result = view.acquire_lease()
        pool.acquire_lease.assert_called_once_with(credential_id="cred-a")
        assert result == "cred-a"

    def test_acquire_lease_with_explicit_id_delegates(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        pool.acquire_lease.return_value = "cred-x"
        view = ScopedCredentialView(entry, pool)

        result = view.acquire_lease(credential_id="cred-x")
        pool.acquire_lease.assert_called_once_with(credential_id="cred-x")
        assert result == "cred-x"

    def test_release_lease_delegates_to_pool(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        view = ScopedCredentialView(entry, pool)

        view.release_lease("cred-a")
        pool.release_lease.assert_called_once_with("cred-a")

    def test_provider_delegates_to_pool(self):
        entry = _make_mock_pool_entry("cred-a")
        pool = MagicMock()
        pool.provider = "openrouter"
        view = ScopedCredentialView(entry, pool)

        assert view.provider == "openrouter"


class TestScopedViewDoesNotExposeOtherCredentials:
    """End-to-end: a scoped view over a pool with multiple entries only
    exposes the single leased credential."""

    def test_multi_credential_pool_scoped_to_one(self):
        cred_a = _make_mock_pool_entry("cred-a", "key-alpha")
        cred_b = _make_mock_pool_entry("cred-b", "key-beta")
        cred_c = _make_mock_pool_entry("cred-c", "key-gamma")

        pool = MagicMock()
        pool.entries.return_value = [cred_a, cred_b, cred_c]
        pool.provider = "openrouter"

        # Scope to cred_b only
        view = ScopedCredentialView(cred_b, pool)

        # select/current/peek all return cred_b
        assert view.select() is cred_b
        assert view.current() is cred_b
        assert view.peek() is cred_b

        # entries() shows only cred_b, not the full pool
        assert view.entries() == [cred_b]
        assert cred_a not in view.entries()
        assert cred_c not in view.entries()

        # rotate returns same
        assert view.rotate() is cred_b


class TestRunSingleChildUsesScoped:
    """_run_single_child wraps the pool in ScopedCredentialView."""

    def test_child_gets_scoped_view_after_lease(self):
        """After leasing, child._credential_pool should be a ScopedCredentialView,
        not the original full pool."""
        leased_entry = _make_mock_pool_entry("cred-b")

        full_pool = MagicMock()
        full_pool.acquire_lease.return_value = "cred-b"
        full_pool.current.return_value = leased_entry

        child = MagicMock()
        child._credential_pool = full_pool
        child.tool_progress_callback = None

        # Track what _credential_pool is set to during run_conversation
        captured_pool = {}

        def capture_pool(**kwargs):
            captured_pool["pool"] = child._credential_pool
            return {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [],
            }

        child.run_conversation.side_effect = capture_pool

        _run_single_child(
            task_index=0,
            goal="Test scoped view assignment",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        pool_during_run = captured_pool["pool"]
        assert isinstance(pool_during_run, ScopedCredentialView)
        assert pool_during_run.select() is leased_entry
        assert pool_during_run.current() is leased_entry
        assert pool_during_run.entries() == [leased_entry]

    def test_lease_released_on_real_pool_after_scoped(self):
        """release_lease should be called on the real pool, not lost
        when child._credential_pool is replaced with scoped view."""
        leased_entry = _make_mock_pool_entry("cred-a")

        full_pool = MagicMock()
        full_pool.acquire_lease.return_value = "cred-a"
        full_pool.current.return_value = leased_entry

        child = MagicMock()
        child._credential_pool = full_pool
        child.tool_progress_callback = None
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

        _run_single_child(
            task_index=0,
            goal="Test release after scoped",
            child=child,
            parent_agent=_make_mock_parent(),
        )

        # The real pool's release_lease must be called
        full_pool.release_lease.assert_called_once_with("cred-a")
