"""Regression tests for request-local session context in execute_code."""

import json
import os
import queue
import threading
from contextvars import Context
from unittest.mock import patch

from gateway.session_context import get_session_env, set_current_session_id
from tools.code_execution_tool import execute_code
from tools.env_passthrough import register_env_passthrough


def test_execute_code_prefers_request_context_over_global_session_mirror():
    """A concurrent session must not replace the caller's child session ID.

    ``AIAgent`` initialization calls ``set_current_session_id()``, which binds
    a request-local ContextVar and also updates a process-global compatibility
    mirror. This test controls that same interleaving without constructing
    full agents or requiring model credentials.
    """
    let_request_b_run = threading.Event()
    request_b_finished = threading.Event()
    results = queue.Queue()
    errors = queue.Queue()

    def request_b():
        try:
            if not let_request_b_run.wait(timeout=5):
                raise AssertionError("request B was not released")
            set_current_session_id("session-B")
        except BaseException as exc:
            errors.put(exc)
        finally:
            request_b_finished.set()

    def request_a():
        try:
            set_current_session_id("session-A")
            register_env_passthrough(["HERMES_SESSION_ID"])

            let_request_b_run.set()
            if not request_b_finished.wait(timeout=5):
                raise AssertionError("request B did not update the global mirror")

            assert os.environ["HERMES_SESSION_ID"] == "session-B"
            assert get_session_env("HERMES_SESSION_ID") == "session-A"

            raw_result = execute_code(
                'import os\nprint(os.environ.get("HERMES_SESSION_ID", "MISSING"))',
                task_id="session-context-regression",
                enabled_tools=[],
            )
            results.put(json.loads(raw_result))
        except BaseException as exc:
            errors.put(exc)

    with (
        patch.dict(os.environ, {}, clear=False),
        patch(
            "tools.approval.check_execute_code_guard",
            return_value={"approved": True},
        ),
        patch("tools.terminal_tool._docker_has_host_access", return_value=False),
        patch(
            "tools.terminal_tool._get_env_config",
            return_value={"env_type": "local"},
        ),
        patch(
            "tools.code_execution_tool._load_config",
            return_value={"timeout": 15, "max_tool_calls": 50},
        ),
    ):
        thread_b = threading.Thread(
            target=lambda: Context().run(request_b),
            name="session-context-request-b",
        )
        thread_a = threading.Thread(
            target=lambda: Context().run(request_a),
            name="session-context-request-a",
        )

        thread_b.start()
        thread_a.start()
        thread_a.join(timeout=30)
        thread_b.join(timeout=30)

        assert not thread_a.is_alive(), "request A did not finish"
        assert not thread_b.is_alive(), "request B did not finish"

    if not errors.empty():
        raise errors.get()

    result = results.get_nowait()
    assert result["status"] == "success", result
    assert result["output"].strip() == "session-A"
