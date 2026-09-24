"""Operation scopes and guest shell syntax preserve their outer contracts."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tools.test_core_routing_review import shell_route, dispatch


def test_terminal_nested_scopes_preserve_fence_and_thread_isolation():
    from hermes_cli import session_execution as execution
    from tools.terminal_targets import terminal_operation, check_terminal_operation, _operation
    execution.register_session_execution_context("scope-control", execution.SessionExecutionContext(terminal_access_epoch=lambda: 0))
    lease = execution.resolve_session_execution_context(session_id="scope-control")
    barrier = threading.Barrier(2)
    valid = [True, True]
    def operation(i):
        previous = _operation.get()
        def check():
            if not valid[i]:
                raise execution.SessionExecutionError("scope revoked")
        try:
            with terminal_operation(lease, 0, check):
                with terminal_operation(lease, 0):
                    barrier.wait(5)
                    valid[i] = i == 1
                    if i == 0:
                        with pytest.raises(execution.SessionExecutionError):
                            check_terminal_operation(lease)
                    else:
                        check_terminal_operation(lease)
                assert _operation.get()[2] is check
        finally:
            assert _operation.get() is previous
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(operation, range(2)))
    finally:
        execution.remove_session_execution_context("scope-control")


@pytest.mark.parametrize("mode", ["pipe", "pty", "promoted"])
def test_guest_quoted_cwd_metadata_and_parent_are_preserved(shell_route, mode):
    from tools.process_registry import process_registry
    tt, home, nested, lease, epoch, paused = shell_route
    quoted = home / "a 'quoted' directory"
    quoted.mkdir()
    parent = dispatch("pwd")["output"]
    command = "printf '%s' \"$PWD\" # trailing comment"
    options = {"background": True, "pty": mode == "pty"} if mode != "promoted" else {"timeout": tt.FOREGROUND_MAX_TIMEOUT + 1}
    import shlex
    assert dispatch("cd " + shlex.quote(str(quoted)), target="review")["exit_code"] == 0
    result = dispatch(command, target="review", **options)
    assert "session_id" in result, result
    completed = process_registry.wait(result["session_id"], timeout=10)
    assert completed["exit_code"] == 0
    assert completed["output"].strip() == str(quoted)
    proc = process_registry.get(result["session_id"])
    assert proc.command == command and proc.cwd == str(quoted)
    assert result["target_cwd"] == str(quoted) and result["execution_target"] == "review"
    assert dispatch("pwd")["output"] == parent
