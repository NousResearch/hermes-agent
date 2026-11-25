"""Prompt callbacks and immutable policy travel together across reused workers."""
from concurrent.futures import ThreadPoolExecutor

from hermes_cli.plugins import clear_thread_tool_whitelist, get_pre_tool_call_directive, set_thread_tool_whitelist
from tools import terminal_tool
from tools.thread_context import capture_thread_callbacks, propagate_context_to_thread


def test_whitelist_migration_keeps_approval_callback_capture_and_replay():
    def callback(*args, **kwargs):
        return True
    previous = terminal_tool._get_approval_callback()
    terminal_tool.set_approval_callback(callback)
    set_thread_tool_whitelist({"memory"}, "blocked {tool_name}")
    try:
        captured = capture_thread_callbacks()
        assert any(value is callback for _, value in captured)
        with ThreadPoolExecutor(max_workers=1) as pool:
            seen, directive = pool.submit(propagate_context_to_thread(
                lambda: (terminal_tool._get_approval_callback(), get_pre_tool_call_directive("terminal", {}))
            )).result(timeout=10)
            assert seen is callback
            assert directive == ("block", "blocked terminal")
            assert pool.submit(terminal_tool._get_approval_callback).result(timeout=10) is None
    finally:
        terminal_tool.set_approval_callback(previous)
        clear_thread_tool_whitelist()
