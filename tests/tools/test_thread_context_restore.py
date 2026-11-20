"""Prompt callbacks survive nested dispatch and remain isolated on reused threads."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tools import terminal_tool
from tools.thread_context import propagate_context_to_thread


def test_inline_dispatch_restores_approval_callback_after_exception():
    callback = object()
    terminal_tool.set_approval_callback(callback)
    try:
        def raises():
            assert terminal_tool._get_approval_callback() is callback
            raise ValueError("probe")

        with pytest.raises(ValueError, match="probe"):
            propagate_context_to_thread(raises)()
        assert terminal_tool._get_approval_callback() is callback
    finally:
        terminal_tool.set_approval_callback(None)


def test_empty_parent_masks_reused_worker_callback_and_restores_it():
    sentinel = object()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(terminal_tool.set_approval_callback, sentinel).result()
        terminal_tool.set_approval_callback(None)
        wrapped = propagate_context_to_thread(terminal_tool._get_approval_callback)
        assert pool.submit(wrapped).result() is None
        assert pool.submit(terminal_tool._get_approval_callback).result() is sentinel
        pool.submit(terminal_tool.set_approval_callback, None).result()
