"""A background fork's dispatch-side tool whitelist reaches parallel tool workers.

The background review (and every fork built like it) restricts its tools with a thread-local
``pre_tool_call`` whitelist (``hermes_cli.plugins.set_thread_tool_whitelist``). A batch of
parallel-safe calls runs on pool workers started through ``propagate_context_to_thread``; when the
whitelist stayed on the fork's own thread, every call of such a batch escaped it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from hermes_cli.plugins import clear_thread_tool_whitelist, get_pre_tool_call_directive, set_thread_tool_whitelist
from tools.thread_context import propagate_context_to_thread


def _directive_on_worker(pool: ThreadPoolExecutor, tool_name: str):
    return pool.submit(propagate_context_to_thread(lambda: get_pre_tool_call_directive(tool_name, {}))).result()


def test_parallel_worker_enforces_the_dispatching_threads_whitelist():
    set_thread_tool_whitelist({"skill_manage"}, "only skill_manage; {tool_name} denied")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert _directive_on_worker(pool, "web_search") == ("block", "only skill_manage; web_search denied")
            assert _directive_on_worker(pool, "skill_manage") == (None, None)
    finally:
        clear_thread_tool_whitelist()


def test_a_reused_worker_does_not_keep_a_previous_whitelist():
    with ThreadPoolExecutor(max_workers=1) as pool:
        set_thread_tool_whitelist({"skill_manage"})
        try:
            assert _directive_on_worker(pool, "web_search")[0] == "block"
        finally:
            clear_thread_tool_whitelist()
        # Same pool thread, now dispatched from a live (unrestricted) turn.
        assert _directive_on_worker(pool, "web_search") == (None, None)
