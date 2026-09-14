"""Tests for ``delegation.suppress_memory_notify``: gating the parent memory-provider
``on_delegation`` notifications that ``_notify_memory_manager`` fires for delegate_task
child results. Default must keep firing (the hook is a deliberate provider contract);
a truthy flag opts the parent out. Malformed config fails OPEN (children still notify).
"""

from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool_results import _notify_memory_manager


def _make_parent():
    parent = MagicMock()
    parent._memory_manager = MagicMock()
    return parent


def _make_result(task_index=0, summary="did the work"):
    return {"task_index": task_index, "summary": summary, "status": "completed"}


def _notify_with_config(parent, results, cfg, *, load_error=None):
    """Run _notify_memory_manager with ``tools.delegate_tool._load_config`` patched —
    the seam production reads (function-local import)."""
    task_list = [{"goal": f"goal-{i}"} for i in range(len(results))]
    children = {i: MagicMock(session_id=f"child-{i}") for i in range(len(results))}
    target = {"return_value": cfg} if load_error is None else {"side_effect": load_error}
    with patch("tools.delegate_tool._load_config", **target):
        _notify_memory_manager(results, task_list, children, parent)


def test_on_delegation_fires_per_entry_by_default():
    """No flag configured: every child result still reaches the parent's provider with
    task text, summary text, and the child's session id (hook contract invariant)."""
    parent = _make_parent()
    results = [_make_result(0, "alpha done"), _make_result(1, "beta done")]

    _notify_with_config(parent, results, {})

    calls = parent._memory_manager.on_delegation.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs == {"task": "goal-0", "result": "alpha done", "child_session_id": "child-0"}
    assert calls[1].kwargs == {"task": "goal-1", "result": "beta done", "child_session_id": "child-1"}


@pytest.mark.parametrize("flag_value", [True, "true", "yes", "1"])
def test_suppress_flag_truthy_skips_on_delegation(flag_value):
    """A truthy delegation.suppress_memory_notify short-circuits before any per-entry work."""
    parent = _make_parent()

    _notify_with_config(parent, [_make_result(0)], {"suppress_memory_notify": flag_value})

    parent._memory_manager.on_delegation.assert_not_called()


@pytest.mark.parametrize("flag_value", [False, "false", "no", "off", 0, None])
def test_suppress_flag_falsy_still_notifies(flag_value):
    """Falsy values (including explicit false and unknown strings) keep the notify contract."""
    parent = _make_parent()

    _notify_with_config(parent, [_make_result(0)], {"suppress_memory_notify": flag_value})

    parent._memory_manager.on_delegation.assert_called_once()


def test_missing_delegation_key_still_notifies():
    """Absent flag in a populated delegation section: notify (default-on contract)."""
    parent = _make_parent()

    _notify_with_config(parent, [_make_result(0)], {"max_summary_chars": 24000})

    parent._memory_manager.on_delegation.assert_called_once()


def test_notify_loop_isolates_raising_provider():
    """A provider raising inside on_delegation must not break result processing,
    and the hook is still invoked (isolation, not skipping)."""
    parent = _make_parent()
    parent._memory_manager.on_delegation.side_effect = RuntimeError("provider down")

    _notify_with_config(parent, [_make_result(0)], {})

    parent._memory_manager.on_delegation.assert_called_once()


def test_malformed_config_fails_open():
    """A raising _load_config must not crash finalization: children still notify."""
    parent = _make_parent()

    _notify_with_config(parent, [_make_result(0)], None, load_error=TypeError("bad config"))

    parent._memory_manager.on_delegation.assert_called_once()
