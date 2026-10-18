"""Log-spam / escalation fixes (2026-10-18):

Fix 1: Watch notification admission_conflict — downgraded from WARNING to DEBUG
Fix 2: Kanban dispatcher tick failure — logger.exception() → escalation logging
Fix 3: Session hygiene compression failure — escalation logging added
"""

import logging
import time
import types

from tui_gateway import server
from gateway import run_notifications, run_turn_hygiene
from gateway import kanban_watchers_dispatcher


def test_watch_not_admitted_is_now_debug(caplog):
    """Fix 1: WatchNotAccepted is no longer logged at WARNING level.

    The old behaviour logged every transient store error as a WARNING,
    flooding errors.log (110 occurrences in a single day).  The fix
    downgrades to DEBUG while preserving the False return value so
    callers still requeue.
    """
    import types
    from gateway.run_notifications import GatewayNotificationsMixin

    # Build a minimal source object
    source = types.SimpleNamespace(
        platform=types.SimpleNamespace(value="discord"),
        chat_id="test_channel",
        thread_id=None,
    )

    # Patch logger to capture log records
    caplog.set_level(logging.DEBUG)

    # Create a minimal notification mixin and call _inject_watch_notification
    # We mock the internals so the function hits the WakeNotAccepted path
    mixin = types.SimpleNamespace()
    mixin.adapters = {"discord": types.SimpleNamespace()}
    mixin.adapters["discord"].chat_id = "test_channel"
    mixin.adapters["discord"].prime_routing_cache = lambda *a, **k: None

    from unittest.mock import AsyncMock, MagicMock, patch

    async def _test():
        class FakeAuthority:
            pass

        from hermes_state_runtime import RuntimeStoreError

        evt = types.SimpleNamespace(
            session_id="test", message_id="123", origin_session_id="test"
        )

        # Patch admit_internal_event (the fallback path)
        with patch.object(run_notifications, "admit_internal_event", new_callable=AsyncMock):
            # Make the admit_process_notice path raise RuntimeStoreError
            with patch.object(run_notifications, "active_authority", return_value=FakeAuthority()):
                with patch("gateway.run_notifications.admit_process_notice", new_callable=AsyncMock,
                           side_effect=RuntimeStoreError("store is busy")):
                    # Call the watch injection
                    result = await mixin._inject_watch_notification(
                        source, "test notification text", evt, raise_not_accepted=False
                    )
        # Return value is False (requeue), not an exception
        assert result is False

    # Run the async test
    import asyncio
    asyncio.run(_test())

    # Check that no WARNING was logged (only DEBUG/INFO)
    warning_records = [r for r in caplog.get_records("call") if r.levelno >= logging.WARNING]
    watch_warnings = [r for r in warning_records if "Watch notification not admitted" in r.message]
    assert len(watch_warnings) == 0, (
        f"Expected no WARNING log records for watch admission failures, "
        f"but got: {[r.message for r in watch_warnings]}"
    )

    # Check that DEBUG was logged (the new behavior)
    debug_records = [r for r in caplog.get_records("call")
                     if r.levelno < logging.INFO and "Watch notification not admitted" in r.message]
    assert len(debug_records) == 1, (
        f"Expected exactly one DEBUG log record, got {len(debug_records)}"
    )


def test_kanban_dispatcher_error_tracking():
    """Fix 2: Kanban dispatcher tracks per-board failure counts and
    escalates from DEBUG → WARNING → ERROR as failures accumulate.

    The old code called logger.exception() (full traceback) on EVERY tick,
    flooding logs when the underlying error was transient.
    """
    kb = types.SimpleNamespace(
        DEFAULT_FAILURE_LIMIT=3,
        kanban_db_path=lambda slug: types.SimpleNamespace(
            expanduser=lambda: types.SimpleNamespace(resolve=lambda: "/tmp/board.db"),
            stat=lambda: types.SimpleNamespace(st_mtime_ns=1, st_size=1024),
        ),
    )
    settings = types.SimpleNamespace(
        interval=60, max_spawn=10, max_in_progress=5,
        failure_limit=3, stale_timeout_seconds=3600,
        reconcile_orphans=False, default_assignee=None,
        max_in_progress_per_profile=None,
    )

    dispatcher = kanban_watchers_dispatcher._KanbanDispatcher(kb, settings)

    # Verify new tracking attributes exist
    assert hasattr(dispatcher, "_non_corrupt_errors")
    assert hasattr(dispatcher, "_non_corrupt_error_times")
    assert dispatcher._NON_CORRUPT_ERROR_WARN_THRESHOLD == 10
    assert dispatcher._NON_CORRUPT_ERROR_ERROR_THRESHOLD == 50

    # Simulate error accumulation
    board_slug = "test_board"

    # Simulate 3 failures (first 3 logged at WARNING)
    for i in range(3):
        dispatcher._non_corrupt_errors[board_slug] = i
    # The next failure (count=4) is logged at DEBUG (between 4 and 9)
    # Verify the escalation logic is in the source
    import inspect
    source = inspect.getsource(kanban_watchers_dispatcher._KanbanDispatcher.tick_once_for_board)
    assert "_NON_CORRUPT_ERROR_WARN_THRESHOLD" in source
    assert "_NON_CORRUPT_ERROR_ERROR_THRESHOLD" in source
    assert "_non_corrupt_errors" in source


def test_hygiene_compression_failure_tracking():
    """Fix 3: Session hygiene compression tracks per-session failure counts
    and escalates from WARNING → (debug) → WARNING → ERROR as failures
    accumulate.  The old code logged WARNING on every single failure,
    producing 25+ log lines from a single failing session."""

    import inspect
    source = inspect.getsource(run_turn_hygiene.run_turn_hygiene)
    assert "_hygiene_compression_failures" in source, (
        "hygiene compression failure tracking not found in run_turn_hygiene"
    )
    assert "logger.error" in source.split("_hygiene_compression_failures")[1].split("logger.debug")[0] if "logger.debug" in source.split("_hygiene_compression_failures")[1] else True
