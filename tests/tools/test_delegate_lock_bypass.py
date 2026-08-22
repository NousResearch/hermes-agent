"""Tests for delegate_tool lock discipline in _active_children tracking.

Both _build_child_agent (append) and _run_single_child's finally block
(remove) must always hold _active_children_lock when mutating
_active_children.  These tests verify:

1. The happy path: lock is present and acquired for every mutation.
2. The None-lock path: when the lock is absent the child is NOT registered
   (safe skip) rather than mutating the list without protection.
3. Concurrency: two concurrent workers cannot corrupt _active_children when
   both append/remove under the same lock.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parent_with_lock():
    """Return a stub parent agent that has _active_children and a real lock."""
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
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
    return parent


_COMMON_PATCHES = dict(
    _build_child_system_prompt="",
    _strip_blocked_tools=[],
    _resolve_child_credential_pool=None,
    _resolve_workspace_hint=None,
    _build_child_progress_callback=None,
    _load_config={},
    _get_max_spawn_depth=5,
    _get_orchestrator_enabled=False,
    _get_inherit_mcp_toolsets=False,
)


def _patch_all(extra=None):
    """Return a list of patch objects for the common _build_child_agent deps."""
    patches = []
    for name, retval in _COMMON_PATCHES.items():
        val = retval() if callable(retval) else retval
        patches.append(patch(f"tools.delegate_tool.{name}", return_value=val))
    if extra:
        for name, retval in extra.items():
            patches.append(patch(f"tools.delegate_tool.{name}", return_value=retval))
    return patches


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestDelegateLockBypass:

    # ------------------------------------------------------------------
    # 1. Happy path: lock is present and used
    # ------------------------------------------------------------------

    def test_append_always_under_lock_when_lock_present(self):
        """Lock must be acquired for the _active_children append."""
        from tools.delegate_tool import _build_child_agent

        parent = _make_parent_with_lock()
        acquired = []
        released = []

        class TrackingLock:
            def __init__(self, real_lock):
                self._real = real_lock

            def __enter__(self):
                self._real.__enter__()
                acquired.append(True)
                return self

            def __exit__(self, *args):
                released.append(True)
                return self._real.__exit__(*args)

        parent._active_children_lock = TrackingLock(threading.Lock())

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
            MockAgent.return_value = MagicMock()
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

        assert len(acquired) > 0, "Lock must be acquired for _active_children append"
        assert len(released) > 0, "Lock must be released after _active_children append"

    # ------------------------------------------------------------------
    # 2. None-lock path: child is NOT registered (safe skip)
    # ------------------------------------------------------------------

    def test_none_lock_does_not_append_to_active_children(self):
        """When _active_children_lock is absent the list must not be mutated.

        The original buggy code appended without a lock.  The first fix
        created a new Lock() on the fly (TOCTOU risk).  The correct fix
        skips registration entirely and logs a debug message.
        """
        from tools.delegate_tool import _build_child_agent

        parent = _make_parent_with_lock()
        # Simulate a stub that has _active_children but no lock
        del parent._active_children_lock
        # Ensure getattr returns None (MagicMock would return a truthy mock)
        parent._active_children_lock = None

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
            MockAgent.return_value = MagicMock()
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

        assert parent._active_children == [], (
            "When _active_children_lock is None the list must NOT be mutated; "
            "got unexpected entries: %r" % parent._active_children
        )

    def test_none_lock_does_not_create_new_lock_on_parent(self):
        """The fix must NOT write a new Lock() onto parent._active_children_lock.

        Creating a per-call lock is a TOCTOU hazard: two concurrent workers
        each create a distinct lock and then mutate the list simultaneously.
        """
        from tools.delegate_tool import _build_child_agent

        parent = _make_parent_with_lock()
        parent._active_children_lock = None

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
            MockAgent.return_value = MagicMock()
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

        assert parent._active_children_lock is None, (
            "The fix must not overwrite _active_children_lock with a new Lock(); "
            "doing so creates divergent lock objects under concurrent load. "
            "Got: %r" % parent._active_children_lock
        )

    # ------------------------------------------------------------------
    # 3. No unlocked else-branch in source (structural guard)
    # ------------------------------------------------------------------

    def test_no_unlocked_fallback_path(self):
        """_active_children.append must not appear in an unlocked else branch."""
        import inspect
        from tools.delegate_tool import _build_child_agent

        source = inspect.getsource(_build_child_agent)
        lines = source.split('\n')
        in_lock_block = False
        found_unlocked_append = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if '_active_children_lock' in stripped:
                in_lock_block = True
            if in_lock_block and stripped.startswith('else:'):
                for j in range(i + 1, min(i + 5, len(lines))):
                    if '_active_children' in lines[j] and 'append' in lines[j]:
                        found_unlocked_append = True
                        break

        assert not found_unlocked_append, (
            "Found unlocked append in else branch — "
            "_active_children.append should only happen inside the lock"
        )

    def test_no_unlocked_remove_path(self):
        """_active_children.remove must not appear in an unlocked else branch
        in _run_single_child (the unregister path mirrors the register path)."""
        import inspect
        from tools.delegate_tool import _run_single_child

        source = inspect.getsource(_run_single_child)
        lines = source.split('\n')
        in_lock_block = False
        found_unlocked_remove = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if '_active_children_lock' in stripped:
                in_lock_block = True
            if in_lock_block and stripped.startswith('else:'):
                for j in range(i + 1, min(i + 5, len(lines))):
                    if '_active_children' in lines[j] and 'remove' in lines[j]:
                        found_unlocked_remove = True
                        break

        assert not found_unlocked_remove, (
            "Found unlocked remove in else branch of _run_single_child — "
            "_active_children.remove should only happen inside the lock"
        )

    # ------------------------------------------------------------------
    # 4. Concurrency: two threads cannot corrupt the list
    # ------------------------------------------------------------------

    def test_concurrent_append_and_remove_are_serialised(self):
        """Two threads appending/removing under the same lock must not corrupt
        the list.  This is the actual thread-safety property the PR claims to
        fix.  We use a real threading.Lock and verify no list corruption occurs
        across 200 concurrent iterations."""
        import threading as _threading

        parent = MagicMock()
        parent._active_children = []
        lock = _threading.Lock()
        parent._active_children_lock = lock

        errors = []
        iterations = 200

        def worker(i):
            child = object()
            # Simulate what _build_child_agent does
            acquired_lock = getattr(parent, "_active_children_lock", None)
            if acquired_lock is not None:
                with acquired_lock:
                    parent._active_children.append(child)
            # Small yield to increase interleaving probability
            time.sleep(0)
            # Simulate what _run_single_child's finally block does
            acquired_lock2 = getattr(parent, "_active_children_lock", None)
            if acquired_lock2 is not None:
                with acquired_lock2:
                    try:
                        parent._active_children.remove(child)
                    except ValueError:
                        errors.append(f"worker {i}: child not found for remove")

        threads = [_threading.Thread(target=worker, args=(i,)) for i in range(iterations)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, "Concurrent list mutations produced errors: %s" % errors
        assert parent._active_children == [], (
            "After all workers complete _active_children should be empty; "
            "got %d stale entries (list corruption)" % len(parent._active_children)
        )
