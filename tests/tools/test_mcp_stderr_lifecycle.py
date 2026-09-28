"""MCP shutdown releases only the selected profile's cached stderr handle."""

import io
from pathlib import Path

from hermes_constants import (
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.logs import _parse_line_timestamp
from tools.mcp_tool_config import _StderrTee, _get_mcp_stderr_log, _write_stderr_log_header
from tools.mcp_tool_lifecycle import shutdown_mcp_servers


def test_scoped_shutdown_releases_log_and_preserves_other_profile(tmp_path):
    handles = []
    for name in ("first", "second"):
        token = set_hermes_home_override(tmp_path / name)
        try:
            handles.append(_get_mcp_stderr_log())
        finally:
            reset_hermes_home_override(token)
    first, second = handles
    try:
        shutdown_mcp_servers(scope=hermes_home_key(tmp_path / "first"))
        assert first.closed
        second.write("other profile remains usable\n")
        second.flush()
        token = set_hermes_home_override(tmp_path / "first")
        try:
            reopened = _get_mcp_stderr_log()
            handles.append(reopened)
            reopened.write("reload can reopen the log\n")
            reopened.flush()
        finally:
            reset_hermes_home_override(token)
        shutdown_mcp_servers()
        assert second.closed and reopened.closed
    finally:
        for handle in handles:
            handle.close()

def test_rename_profile_releases_cached_log_handle(tmp_path, monkeypatch):
    from hermes_cli import profiles
    from tools import mcp_tool_config

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    monkeypatch.setattr(profiles, "_live_default_multiplexer", lambda: False)
    old_dir = profiles.create_profile("mcp-log-old", no_alias=True)
    token = set_hermes_home_override(old_dir)
    try:
        handle = _get_mcp_stderr_log()
    finally:
        reset_hermes_home_override(token)
    try:
        new_dir = profiles.rename_profile("mcp-log-old", "mcp-log-new")
        # The old home's handle must be released before the directory moves (Windows refuses
        # to rename a directory holding an open file) and must not linger under the old key.
        assert handle.closed
        assert hermes_home_key(old_dir) not in mcp_tool_config._mcp_stderr_log_fh
        assert new_dir.is_dir() and not old_dir.exists()
    finally:
        handle.close()


def test_every_mcp_log_line_carries_a_stamp_hermes_logs_since_reads(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        _write_stderr_log_header("probe")
        log = io.StringIO()
        tee = _StderrTee(log)
        # A line split across writes, a CRLF line and a last line with no newline.
        for chunk in (b"first half ", b"second half\n", b"crlf line\r\n", b"no newline at exit"):
            tee.sink.write(chunk)
        assert tee.close() == "first half second half\ncrlf line\r\nno newline at exit"
        banner = (tmp_path / "logs" / "mcp-stderr.log").read_text().strip()
    finally:
        reset_hermes_home_override(token)
        shutdown_mcp_servers(scope=hermes_home_key(tmp_path))
    lines = log.getvalue().splitlines()
    assert [line[24:] for line in lines] == ["first half second half", "crlf line", "no newline at exit"]
    for line in [banner, *lines]:
        assert _parse_line_timestamp(line) is not None, line
    assert banner.endswith("===== starting MCP server 'probe' =====")
