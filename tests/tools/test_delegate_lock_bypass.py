"""Tests for delegate_tool lock discipline in _active_children tracking.

Both _build_child_agent (append) and _run_single_child's finally block
(remove) must always hold _active_children_lock when mutating
_active_children.  The current code has an unlocked fallback branch
that runs when the lock is None, bypassing thread-safety.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestDelegateLockBypass:
    def test_append_always_under_lock_when_children_exist(self):
        """Simulate _build_child_agent's lock path: append must always be
        inside the lock context, not in an unlocked else branch."""
        parent = MagicMock()
        parent._active_children = []
        lock = threading.Lock()
        parent._active_children_lock = lock

        # Simulate the current (buggy) code path
        from tools.delegate_tool import _build_child_agent

        # We'll patch to intercept just the lock usage
        acquired = []
        released = []

        class TrackingLock:
            """A lock wrapper that records acquire/release calls."""
            def __init__(self, real_lock):
                self._real = real_lock

            def __enter__(self):
                self._real.__enter__()
                acquired.append(True)
                return self

            def __exit__(self, *args):
                released.append(True)
                return self._real.__exit__(*args)

        tracking = TrackingLock(lock)
        parent._active_children_lock = tracking

        # Set concrete attributes so comparisons inside _build_child_agent work
        parent._delegate_depth = 0
        parent._subagent_id = None
        parent.enabled_toolsets = None
        parent.valid_tool_names = []
        parent.model = "test-model"
        parent.base_url = "http://localhost"
        parent.api_key = "test-key"
        parent.provider = "test"
        parent.api_mode = None
        parent.acp_command = None
        parent.acp_args = []
        parent.reasoning_config = None
        parent.prefill_messages = None
        parent.platform = None
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent.max_tokens = None
        parent._print_fn = None

        with patch("tools.delegate_tool._build_child_system_prompt", return_value=""), \
             patch("tools.delegate_tool._strip_blocked_tools", return_value=[]), \
             patch("tools.delegate_tool._resolve_child_credential_pool", return_value=None), \
             patch("tools.delegate_tool._resolve_workspace_hint", return_value=None), \
             patch("tools.delegate_tool._build_child_progress_callback", return_value=None), \
             patch("tools.delegate_tool._load_config", return_value={}), \
             patch("tools.delegate_tool._get_max_spawn_depth", return_value=5), \
             patch("tools.delegate_tool._get_orchestrator_enabled", return_value=False), \
             patch("tools.delegate_tool._get_inherit_mcp_toolsets", return_value=False), \
             patch("tools.delegate_tool.AIAgent", create=True) as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child

            _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                task_count=1,
                parent_agent=parent,
            )

        # Lock should have been acquired for the append
        assert len(acquired) > 0, "Lock should be acquired for _active_children append"
        assert len(released) > 0, "Lock should be released after _active_children append"

    def test_no_unlocked_fallback_path(self):
        """When _active_children exists but _active_children_lock is None,
        the code must NOT fall through to an unprotected append."""
        # Read the source to verify no unlocked else branch exists
        import inspect
        from tools.delegate_tool import _build_child_agent

        source = inspect.getsource(_build_child_agent)

        # The buggy pattern: after `if lock:`, an `else:` that does append
        # We check that the lock path doesn't have an unlocked else fallback
        # that accesses _active_children

        # Find the lock-related code
        lines = source.split('\n')
        in_lock_block = False
        found_unlocked_append = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if '_active_children_lock' in stripped:
                in_lock_block = True
            if in_lock_block and stripped.startswith('else:'):
                # Check if the else block contains _active_children.append
                for j in range(i + 1, min(i + 5, len(lines))):
                    if '_active_children' in lines[j] and 'append' in lines[j]:
                        found_unlocked_append = True
                        break

        assert not found_unlocked_append, (
            "Found unlocked append in else branch — "
            "_active_children.append should only happen inside the lock"
        )
