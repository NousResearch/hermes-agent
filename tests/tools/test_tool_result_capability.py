"""Opaque, same-session recovery for persisted tool results."""

from __future__ import annotations

import json
import os
import re
import stat
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.file_tools import _handle_read_file, read_file_tool
from tools.tool_result_storage import (
    PERSISTED_OUTPUT_TAG,
    SpillCapabilityError,
    maybe_persist_tool_result,
    get_spillover_dir,
    resolve_spill_capability,
)

_URI_RE = re.compile(
    r"hermes-spill://v1/[0-9a-f]{64}/[0-9a-f]{64}/[0-9a-f]{32}"
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


def _spill(content: str, session_id: str) -> tuple[str, str]:
    notice = maybe_persist_tool_result(
        content=content,
        tool_name="terminal",
        tool_use_id="call/reused-across-sessions",
        env=None,
        threshold=1,
        session_id=session_id,
    )
    match = _URI_RE.search(notice)
    assert PERSISTED_OUTPUT_TAG in notice
    assert match is not None
    return notice, match.group(0)


def test_capability_round_trip_is_exact_and_host_path_is_hidden():
    payload = "head\n" + ("middle\n" * 200) + "tail"
    notice, uri = _spill(payload, "owner-session")

    assert resolve_spill_capability(uri, session_id="owner-session") == payload
    assert "cache/spillover" not in notice
    assert "/Users/" not in notice


def test_capability_payload_is_not_plaintext_at_rest_or_named_with_bearer_token():
    payload = "private-secret-payload\n" + ("x" * 2_000)
    _notice, uri = _spill(payload, "owner-session")
    capability = uri.rsplit("/", 1)[-1]
    path = next(get_spillover_dir().glob("spill_*"))

    assert payload.encode("utf-8") not in path.read_bytes()
    assert capability not in path.name


def test_cross_session_and_tampered_capabilities_fail_closed():
    payload = "private\n" + ("x" * 2_000)
    _notice, uri = _spill(payload, "owner-session")

    with pytest.raises(SpillCapabilityError):
        resolve_spill_capability(uri, session_id="other-session")

    tampered = uri[:-1] + ("0" if uri[-1] != "0" else "1")
    with pytest.raises(SpillCapabilityError):
        resolve_spill_capability(tampered, session_id="owner-session")


def test_reused_tool_call_id_cannot_collide_across_sessions():
    _notice_a, uri_a = _spill("payload-a", "session-a")
    _notice_b, uri_b = _spill("payload-b", "session-b")

    assert uri_a != uri_b
    assert resolve_spill_capability(uri_a, session_id="session-a") == "payload-a"
    assert resolve_spill_capability(uri_b, session_id="session-b") == "payload-b"


def test_scoped_write_failure_never_downgrades_to_host_path(monkeypatch):
    from tools import tool_result_storage

    monkeypatch.setattr(
        tool_result_storage,
        "_write_capability_spillover",
        lambda *_args, **_kwargs: None,
    )
    content = "private\n" + ("x" * 5_000)

    result = maybe_persist_tool_result(
        content=content,
        tool_name="terminal",
        tool_use_id="predictable-id",
        threshold=1,
        session_id="scoped-session",
    )

    assert "hermes-spill://" not in result
    assert "cache/spillover" not in result
    assert "predictable-id" not in result
    assert "no recovery path was emitted" in result
    assert len(result) < len(content)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_symlink_swap_and_payload_tampering_fail_closed():
    _notice, uri = _spill("original-payload", "tamper-session")
    path = next(get_spillover_dir().glob("spill_*"))
    backup = path.with_suffix(".backup")
    path.rename(backup)
    path.symlink_to(backup)

    with pytest.raises(SpillCapabilityError):
        resolve_spill_capability(uri, session_id="tamper-session")

    path.unlink()
    backup.rename(path)
    path.write_text("changed-payload")
    os.chmod(path, 0o600)
    with pytest.raises(SpillCapabilityError):
        resolve_spill_capability(uri, session_id="tamper-session")


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO semantics")
def test_fifo_replacement_fails_closed_without_blocking():
    _notice, uri = _spill("original-payload", "fifo-session")
    path = next(get_spillover_dir().glob("spill_*"))
    path.unlink()
    os.mkfifo(path, mode=0o600)

    with pytest.raises(SpillCapabilityError):
        resolve_spill_capability(uri, session_id="fifo-session")


def test_reparse_metadata_is_rejected_before_open(monkeypatch):
    from tools import tool_result_storage

    _notice, uri = _spill("original-payload", "reparse-session")
    real = next(get_spillover_dir().glob("spill_*")).lstat()
    fake = SimpleNamespace(
        st_mode=real.st_mode,
        st_dev=real.st_dev,
        st_ino=real.st_ino,
        st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
    )
    monkeypatch.setattr(tool_result_storage.os, "lstat", lambda _path: fake)
    monkeypatch.setattr(
        tool_result_storage.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reparse point was opened")
        ),
    )

    with pytest.raises(SpillCapabilityError):
        resolve_spill_capability(uri, session_id="reparse-session")


def test_read_file_paginates_capability_before_backend_routing():
    payload = "\n".join(f"row-{index}" for index in range(20))
    _notice, uri = _spill(payload, "page-session")

    with (
        patch("tools.file_tools._get_file_ops", side_effect=AssertionError("backend called")),
        patch("tools.file_tools._resolve_path_for_task", side_effect=AssertionError("path resolver called")),
        patch("tools.file_tools._is_blocked_device", side_effect=AssertionError("device guard called")),
    ):
        raw = read_file_tool(
            uri,
            offset=3,
            limit=4,
            session_id="page-session",
        )

    result = json.loads(raw)
    assert result["content"].splitlines() == [
        "3|row-2",
        "4|row-3",
        "5|row-4",
        "6|row-5",
    ]
    assert result["total_lines"] == 20
    assert result["truncated"] is True


def test_handler_forwards_hidden_session_scope():
    _notice, uri = _spill("line-1\nline-2\nline-3", "handler-session")

    result = json.loads(
        _handle_read_file(
            {"path": uri, "offset": 2, "limit": 1},
            task_id="task",
            session_id="handler-session",
        )
    )

    assert result["content"] == "2|line-2"
    assert "error" not in result


def test_capability_read_obeys_existing_char_budget(monkeypatch):
    from tools import file_tools

    _notice, uri = _spill("x" * 5_000, "budget-session")
    monkeypatch.setattr(file_tools, "_max_read_chars_cached", 120)

    result = json.loads(
        read_file_tool(uri, session_id="budget-session")
    )

    assert len(result["content"]) <= 120
    assert result["truncated"] is True
    assert result["truncated_by"] == "bytes"


def test_central_dispatch_spills_direct_transport_result(monkeypatch):
    import model_tools

    payload = "direct-transport\n" + ("z" * 120_000)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *_a, **_k: payload)

    notice = model_tools.handle_function_call(
        "web_search",
        {"query": "test"},
        task_id="direct-task",
        tool_call_id="direct-call",
        session_id="direct-session",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    match = _URI_RE.search(notice)
    assert match is not None
    assert resolve_spill_capability(
        match.group(0), session_id="direct-session",
    ) == payload


def test_central_dispatch_fails_open_when_persistence_raises(monkeypatch):
    import model_tools
    from tools import tool_result_storage

    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *_a, **_k: "result")

    def fail_persistence(**_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        tool_result_storage,
        "maybe_persist_tool_result",
        fail_persistence,
    )

    assert model_tools.handle_function_call(
        "web_search",
        {"query": "test"},
        session_id="direct-session",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    ) == "result"
