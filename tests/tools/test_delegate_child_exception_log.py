"""A child that raises is logged with its own traceback, not ``NoneType: None``.

The warning is emitted after the ``except`` block, where ``exc_info=True`` reads no active exception.
"""
import logging

import pytest

from tests.tools.test_delegate_output_schema import _StubChild, _StubParent
from tools.delegate_tool import _run_single_child


def test_raised_child_logs_its_traceback(caplog: pytest.LogCaptureFixture) -> None:
    child = _StubChild([])

    def boom(user_message, task_id=None, **_kw):
        raise KeyError("local-parent-session")
    child.run_conversation = boom
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool_child_run"):
        entry = _run_single_child(0, "triage", child, _StubParent())
    assert entry["status"] == "error"
    (record,) = [r for r in caplog.records if "raised KeyError" in r.getMessage()]
    assert record.exc_info is not None and record.exc_info[0] is KeyError
    assert "local-parent-session" in caplog.text and "NoneType: None" not in caplog.text
