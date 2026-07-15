from __future__ import annotations

import json

import tools.code_execution_tool as code_execution_tool
import tools.file_tools as file_tools
import tools.read_extract as read_extract
from tools.file_operations import ReadResult, WriteResult


class _FakeFileOperations:
    def __init__(self):
        self.writes: list[tuple[str, str]] = []

    def read_file(self, path, offset, limit):
        assert path == "notes.txt"
        assert (offset, limit) == (1, 500)
        return ReadResult(
            content="1|lease-only content",
            total_lines=1,
            file_size=18,
            truncated=False,
        )

    def write_file(self, path, content):
        self.writes.append((path, content))
        return WriteResult(bytes_written=len(content.encode("utf-8")))


def _unexpected_host_io(*_args, **_kwargs):
    raise AssertionError("isolated worker file operation touched host filesystem")


def test_isolated_file_reads_and_writes_never_use_host_paths(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "isolated_worker")
    monkeypatch.setenv("TERMINAL_CWD", "/workspace")
    fake = _FakeFileOperations()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: fake)
    monkeypatch.setattr(file_tools.os.path, "getmtime", _unexpected_host_io)
    monkeypatch.setattr(file_tools, "get_read_block_error", _unexpected_host_io)
    monkeypatch.setattr(file_tools.file_state, "record_read", _unexpected_host_io)
    monkeypatch.setattr(file_tools.file_state, "check_stale", _unexpected_host_io)
    monkeypatch.setattr(file_tools.file_state, "note_write", _unexpected_host_io)

    read_result = json.loads(
        file_tools.read_file_tool("notes.txt", task_id="conversation-42")
    )
    assert read_result["content"] == "1|lease-only content"

    write_result = json.loads(
        file_tools.write_file_tool(
            "notes.txt",
            "updated",
            task_id="conversation-42",
        )
    )
    assert write_result["bytes_written"] == 7
    assert fake.writes == [("/workspace/notes.txt", "updated")]


def test_isolated_document_extraction_is_not_attempted_on_host(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "isolated_worker")
    monkeypatch.setenv("TERMINAL_CWD", "/workspace")
    monkeypatch.setattr(read_extract, "extract_document_text", _unexpected_host_io)
    result = json.loads(
        file_tools.read_file_tool("report.docx", task_id="conversation-42")
    )
    assert "Cannot read binary file" in result["error"]


def test_file_handlers_prefer_exact_session_only_for_isolated_worker(
    monkeypatch,
) -> None:
    kwargs = {"task_id": "delegate-1", "session_id": "conversation-42"}
    monkeypatch.setenv("TERMINAL_ENV", "isolated_worker")
    assert file_tools._handler_task_id(kwargs) == "conversation-42"

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    assert file_tools._handler_task_id(kwargs) == "delegate-1"


def test_execute_code_prefers_exact_session_only_for_isolated_worker(
    monkeypatch,
) -> None:
    kwargs = {"task_id": "delegate-1", "session_id": "conversation-42"}
    monkeypatch.setenv("TERMINAL_ENV", "isolated_worker")
    assert code_execution_tool._execution_task_id(kwargs) == "conversation-42"

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    assert code_execution_tool._execution_task_id(kwargs) == "delegate-1"
