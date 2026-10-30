"""Search failures retain their result while logs expose only diagnostic metadata."""
import errno
import json
import logging
from unittest.mock import Mock

import pytest

from tools import file_tools


@pytest.mark.parametrize("stage", ["backend_init", "backend_search"])
def test_search_process_failure_logs_stage_and_identity_without_content(monkeypatch, caplog, stage):
    private = "private-query-and-document-name"
    error = ProcessLookupError(errno.ESRCH, private, "/private/private-path")
    monkeypatch.setattr(file_tools, "_resolve_path_for_task", lambda *args: "/safe-fixture")
    monkeypatch.setattr(file_tools, "get_read_block_error", lambda *args: None)
    ops = Mock()
    if stage == "backend_init":
        get_ops = Mock(side_effect=error)
    else:
        ops.search.side_effect = error
        get_ops = Mock(return_value=ops)
    monkeypatch.setattr(file_tools, "_get_file_ops", get_ops)
    with caplog.at_level(logging.WARNING, logger="tools.file_tools"):
        result = file_tools._handle_search_files(
            {"pattern": private, "path": "/private/private-path"},
            task_id=f"task-{stage}", session_id="session-fixture")
    assert json.loads(result) == {"error": str(error)}
    records = [r for r in caplog.records if "search_files failed" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert f"stage={stage}" in message
    assert "ProcessLookupError" in message
    assert f"errno={errno.ESRCH}" in message
    assert "session-fixture" in message and f"task-{stage}" in message
    assert private not in message and "/private/" not in message
    assert records[0].exc_info is None
