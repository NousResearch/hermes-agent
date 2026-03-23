"""Test that _run_single_child sets the subagent context flag."""
import threading
from unittest.mock import MagicMock, patch
from tools.approval import is_subagent_context, set_subagent_context


class TestDelegateSubagentContext:
    def test_run_single_child_sets_subagent_context(self):
        """_run_single_child should set is_subagent_context()=True during child execution."""
        # Track what is_subagent_context() returns during child.run_conversation()
        context_during_run = {}

        mock_child = MagicMock()
        def fake_run_conversation(user_message):
            context_during_run["is_subagent"] = is_subagent_context()
            return {"final_response": "done", "messages": []}
        mock_child.run_conversation = fake_run_conversation
        mock_child.tool_progress_callback = None

        from tools.delegate_tool import _run_single_child

        # Ensure clean state
        set_subagent_context(False)
        assert is_subagent_context() is False

        result = _run_single_child(0, "test goal", child=mock_child, parent_agent=MagicMock())

        assert context_during_run.get("is_subagent") is True, \
            "_run_single_child should set subagent context during child execution"
        assert is_subagent_context() is False, \
            "subagent context should be cleaned up after _run_single_child"

    def test_subagent_context_cleaned_up_on_error(self):
        """Even if child.run_conversation raises, subagent context should be False afterward."""
        mock_child = MagicMock()
        mock_child.run_conversation.side_effect = RuntimeError("boom")
        mock_child.tool_progress_callback = None

        from tools.delegate_tool import _run_single_child

        set_subagent_context(False)
        result = _run_single_child(0, "test goal", child=mock_child, parent_agent=MagicMock())

        assert is_subagent_context() is False, \
            "subagent context should be cleaned up even after errors"


class TestSubmitPendingBlockingUsesHandles:
    """Verify submit_pending_blocking delegates to the handle-based system."""

    def test_submit_pending_blocking_creates_waiter(self):
        """submit_pending_blocking should create a blocking waiter visible via has_blocking_waiters."""
        from tools.approval import (
            submit_pending_blocking,
            has_blocking_waiters,
            get_blocking_waiter_details,
            resolve_pending,
        )
        import threading

        session_key = "test_spb_creates_waiter"
        approval = {"command": "rm -rf /", "pattern_key": "rm_recursive", "description": "recursive delete"}

        # Start submit_pending_blocking in a thread (it blocks)
        result_holder = {}
        def blocking_call():
            result_holder["result"] = submit_pending_blocking(session_key, approval, timeout=10)

        t = threading.Thread(target=blocking_call)
        t.start()

        import time
        # Give thread time to register
        time.sleep(0.5)

        try:
            # The waiter should be visible via the handle-based system
            assert has_blocking_waiters(session_key), \
                "submit_pending_blocking should create a waiter visible via has_blocking_waiters"

            details = get_blocking_waiter_details(session_key)
            assert len(details) >= 1, "Should have at least one waiter detail"
            assert details[0]["command"] == "rm -rf /"

            # Resolve it
            rid = details[0]["request_id"]
            resolve_pending(rid, approved=True)
        finally:
            t.join(timeout=5)

        assert result_holder.get("result", {}).get("approved") is True

    def test_submit_pending_blocking_timeout(self):
        """submit_pending_blocking should return timeout result when not resolved."""
        from tools.approval import submit_pending_blocking

        session_key = "test_spb_timeout"
        approval = {"command": "rm -rf /", "pattern_key": "rm_recursive", "description": "recursive delete"}

        result = submit_pending_blocking(session_key, approval, timeout=0.5)

        assert result["approved"] is False
        assert "timed out" in result.get("message", "").lower() or "timeout" in result.get("message", "").lower()

    def test_parallel_submit_pending_blocking(self):
        """Two concurrent submit_pending_blocking calls on the same session should both work."""
        from tools.approval import (
            submit_pending_blocking,
            has_blocking_waiters,
            get_blocking_waiter_details,
            resolve_pending,
        )
        import threading
        import time

        session_key = "test_parallel_spb"
        approval_1 = {"command": "cmd1", "pattern_key": "pk1", "description": "desc1"}
        approval_2 = {"command": "cmd2", "pattern_key": "pk2", "description": "desc2"}

        results = {}
        def call_1():
            results["r1"] = submit_pending_blocking(session_key, approval_1, timeout=10)
        def call_2():
            results["r2"] = submit_pending_blocking(session_key, approval_2, timeout=10)

        t1 = threading.Thread(target=call_1)
        t2 = threading.Thread(target=call_2)
        t1.start()
        t2.start()

        time.sleep(0.5)

        try:
            # Both should be visible as blocking waiters
            assert has_blocking_waiters(session_key), "Both waiters should be visible"
            details = get_blocking_waiter_details(session_key)
            assert len(details) == 2, f"Expected 2 waiters, got {len(details)}"

            # Resolve both
            for d in details:
                resolve_pending(d["request_id"], approved=True)
        finally:
            t1.join(timeout=5)
            t2.join(timeout=5)

        assert results.get("r1", {}).get("approved") is True
        assert results.get("r2", {}).get("approved") is True
