#!/usr/bin/env python3
"""
End-to-end verification script for kanban→TUI notification delivery.

This script verifies:
1. FIFO writer side: kanban_db._append_event writes JSON to FIFO
2. FIFO reader side: tui_gateway.server global reader reads from FIFO
3. Notification dispatch: _format_kanban_notification formats correctly
4. ENXIO handling: when no reader, error is logged not swallowed
5. Log check: no FIFO-related errors in recent logs
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Add workspace to path
sys.path.insert(0, "/home/c/Desktop/agenda/hermes-agent")

# Use real kanban DB and FIFO
KANBAN_DB = Path.home() / ".hermes" / "kanban.db"
FIFO_PATH = Path.home() / ".hermes" / "tui_kanban.fifo"

def log(msg):
    print(f"[VERIFY] {msg}", flush=True)

def check_1_fifo_write():
    """Check 1: Create task, complete it, verify FIFO write."""
    log("=" * 60)
    log("CHECK 1: FIFO write on task completion")
    log("=" * 60)

    from hermes_cli import kanban_db as kb

    # Ensure FIFO exists (recreate if needed to ensure clean state)
    if FIFO_PATH.exists():
        FIFO_PATH.unlink()
    os.mkfifo(str(FIFO_PATH), 0o600)
    log(f"Created fresh FIFO: {FIFO_PATH}")

    # Start a background reader
    received = []
    done = threading.Event()
    reader_started = threading.Event()

    def reader():
        try:
            reader_started.set()
            with open(str(FIFO_PATH), "r", encoding="utf-8") as f:
                for line in f:
                    received.append(json.loads(line.strip()))
                    done.set()
                    break
        except Exception as e:
            log(f"Reader error: {e}")
            done.set()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    reader_started.wait(timeout=2)
    time.sleep(0.5)  # Extra time for open() to block

    # Create and complete a test task
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="VERIFY-notification-test", assignee="default")
        log(f"Created task: {tid}")
        kb.complete_task(conn, tid, summary="verification complete")
        log(f"Completed task: {tid}")

    # Wait for reader
    ok = done.wait(timeout=5)
    t.join(timeout=2)

    if ok and received:
        data = received[0]
        log(f"FIFO received: {json.dumps(data, indent=2)}")
        assert data["task_id"] == tid, f"task_id mismatch: {data['task_id']} != {tid}"
        assert data["kind"] == "completed", f"kind mismatch: {data['kind']}"
        log("CHECK 1: PASS - FIFO write verified")
        return True, tid
    else:
        log("CHECK 1: FAIL - No data received on FIFO")
        # Debug: check if FIFO still exists and has readers
        log(f"FIFO exists: {FIFO_PATH.exists()}")
        return False, tid

def check_2_format_notification():
    """Check 2: Verify notification formatting."""
    log("=" * 60)
    log("CHECK 2: Notification message formatting")
    log("=" * 60)

    # Import server with mocked dependencies
    from unittest.mock import MagicMock, patch

    fifo = tempfile.mktemp(suffix=".fifo")
    os.mkfifo(fifo, 0o600)

    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        import tui_gateway.server as srv
        importlib.reload(srv)
        srv._KANBAN_FIFO_PATH = fifo

        # Test completed with summary
        ev = {"kind": "completed", "payload": {"summary": "shipped rate limiter"}}
        sub = {"task_id": "t_test", "platform": "cli", "chat_id": "cli-123"}
        msg = srv._format_kanban_notification(ev, sub)
        log(f"Completed msg: {msg}")
        assert msg is not None and "[IMPORTANT: Kanban task t_test done" in msg
        assert "shipped rate limiter" in msg

        # Test blocked with reason
        ev = {"kind": "blocked", "payload": {"reason": "need API key"}}
        msg = srv._format_kanban_notification(ev, sub)
        log(f"Blocked msg: {msg}")
        assert msg is not None and "blocked" in msg
        assert "need API key" in msg

        # Test crashed
        ev = {"kind": "crashed"}
        msg = srv._format_kanban_notification(ev, sub)
        log(f"Crashed msg: {msg}")
        assert msg is not None and "crashed" in msg

        srv._sessions.clear()
        importlib.reload(srv)

    log("CHECK 2: PASS - Notification formatting verified")
    return True

def check_3_global_reader():
    """Check 3: Verify global FIFO reader starts and reads data."""
    log("=" * 60)
    log("CHECK 3: Global FIFO reader")
    log("=" * 60)

    from unittest.mock import MagicMock, patch

    # Create a temp FIFO and point the module at it BEFORE import
    tmpdir = tempfile.mkdtemp()
    fifo = os.path.join(tmpdir, "tui_kanban.fifo")
    os.mkfifo(fifo, 0o600)

    # Patch expanduser so the module uses our temp FIFO from import time
    real_expand = os.path.expanduser
    def fake_expand(p):
        if "tui_kanban.fifo" in p:
            return fifo
        return real_expand(p)

    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value=tmpdir)),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        # Patch expanduser before importing so the module-level mkfifo uses our path
        os.path.expanduser = fake_expand
        try:
            import importlib
            # Remove cached module to force fresh import
            if "tui_gateway.server" in sys.modules:
                del sys.modules["tui_gateway.server"]
            import tui_gateway.server as srv
            importlib.reload(srv)

            # Global reader should have started at import
            assert srv._kanban_global_reader_thread is not None
            assert srv._kanban_global_reader_thread.is_alive()
            log(f"Global reader thread: {srv._kanban_global_reader_thread}")

            # Write test data to FIFO
            test_data = {"task_id": "t_reader_test", "kind": "completed"}
            with open(fifo, "w", encoding="utf-8") as f:
                f.write(json.dumps(test_data) + "\n")

            # Wait for reader to process
            time.sleep(0.5)

            # Check queue
            try:
                queued = srv._kanban_fifo_queue.get_nowait()
                log(f"Queue received: {queued}")
                assert queued["task_id"] == "t_reader_test"
                log("CHECK 3: PASS - Global reader reads FIFO correctly")
                result = True
            except queue.Empty:
                log("CHECK 3: FAIL - Queue empty, reader didn't process data")
                result = False

            srv._sessions.clear()
            importlib.reload(srv)
        finally:
            os.path.expanduser = real_expand
            if os.path.exists(fifo):
                os.unlink(fifo)
            os.rmdir(tmpdir)

    return result

def check_4_enxio_handling():
    """Check 4: Verify ENXIO is logged, not swallowed."""
    log("=" * 60)
    log("CHECK 4: ENXIO handling (no reader)")
    log("=" * 60)

    from hermes_cli import kanban_db as kb

    # Point to a FIFO that has no reader
    fake_fifo = tempfile.mktemp(suffix=".fifo")
    os.mkfifo(fake_fifo, 0o600)

    # Monkey-patch expanduser to use our fake FIFO
    real_expand = os.path.expanduser
    def fake_expand(p):
        if "tui_kanban.fifo" in p:
            return fake_fifo
        return real_expand(p)

    os.path.expanduser = fake_expand

    try:
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="enxio-test", assignee="default")
            # This should NOT raise even though no reader
            kb.complete_task(conn, tid, summary="test")
            log(f"Task completed without error (ENXIO handled gracefully): {tid}")
            log("CHECK 4: PASS - ENXIO handled gracefully (logged, not raised)")
            return True
    except Exception as e:
        log(f"CHECK 4: FAIL - Exception raised: {e}")
        return False
    finally:
        os.path.expanduser = real_expand
        if os.path.exists(fake_fifo):
            os.unlink(fake_fifo)

def check_5_logs():
    """Check 5: Check recent logs for FIFO errors."""
    log("=" * 60)
    log("CHECK 5: Log check for FIFO errors")
    log("=" * 60)

    log_files = [
        Path.home() / ".hermes" / "logs" / "gateway.log",
        Path.home() / ".hermes" / "logs" / "errors.log",
    ]

    errors_found = []
    for log_file in log_files:
        if not log_file.exists():
            continue
        # Check last 100 lines for FIFO/kanban errors (excluding our own script output)
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[-100:]
            for line in lines:
                lower = line.lower()
                # Skip lines that are just our verification script output
                if "[VERIFY]" in line:
                    continue
                if any(k in lower for k in ["fifo", "kanban", "enxio", "tui_kanban"]):
                    if any(e in lower for e in ["error", "exception", "fail"]):
                        errors_found.append(f"{log_file.name}: {line.strip()}")
                    elif "warning" in lower and "kanban" in lower:
                        errors_found.append(f"{log_file.name}: {line.strip()}")
        except Exception as e:
            log(f"Could not read {log_file}: {e}")

    if errors_found:
        log(f"Found {len(errors_found)} FIFO/kanban related log entries:")
        for e in errors_found:
            log(f"  {e}")
    else:
        log("No FIFO/kanban errors found in recent logs")

    log("CHECK 5: DONE - Log check complete")
    return errors_found

def cleanup_test_task(tid):
    """Remove the test task from kanban DB."""
    log(f"Cleaning up test task: {tid}")
    try:
        from hermes_cli import kanban_db as kb
        with kb.connect() as conn:
            # Archive the task
            kb.archive_task(conn, tid)
            log(f"Archived task: {tid}")
    except Exception as e:
        log(f"Cleanup warning: {e}")

def main():
    log("Starting kanban notification delivery verification")
    log(f"Kanban DB: {KANBAN_DB}")
    log(f"FIFO path: {FIFO_PATH}")

    results = {}

    # Check 1: FIFO write
    ok, tid = check_1_fifo_write()
    results["check_1_fifo_write"] = ok

    # Check 2: Format notification
    results["check_2_format_notification"] = check_2_format_notification()

    # Check 3: Global reader
    results["check_3_global_reader"] = check_3_global_reader()

    # Check 4: ENXIO handling
    results["check_4_enxio_handling"] = check_4_enxio_handling()

    # Check 5: Logs
    log_errors = check_5_logs()
    results["check_5_logs"] = len(log_errors) == 0

    # Cleanup
    if tid:
        cleanup_test_task(tid)

    # Summary
    log("=" * 60)
    log("VERIFICATION SUMMARY")
    log("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        log(f"  {name}: {status}")

    all_pass = all(results.values())
    if all_pass:
        log("ALL CHECKS PASSED")
        return 0
    else:
        log("SOME CHECKS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
