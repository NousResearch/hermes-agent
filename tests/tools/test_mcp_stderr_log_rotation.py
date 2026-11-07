"""The shared MCP stderr log is rotated when its handle is opened.

Every stdio MCP child's stderr lands in ``<HERMES_HOME>/logs/mcp-stderr.log``,
which the app's ``logging.max_size_mb`` rotation does not cover. The handle is
cached for the life of the process (per profile home), so the size check runs
when it is (re)opened: an oversized file moves to ``mcp-stderr.log.1`` (one
archived generation) and a fresh file starts.
"""

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import mcp_tool_config
from tools.mcp_tool_config import _close_mcp_stderr_logs, _get_mcp_stderr_log


CAP = 1024


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    # A small cap keeps the fixtures tiny; raising=False lets the test run red on a base without the cap.
    monkeypatch.setattr(mcp_tool_config, "_MCP_STDERR_LOG_MAX_BYTES", CAP, raising=False)
    token = set_hermes_home_override(tmp_path)
    _close_mcp_stderr_logs()
    try:
        logs = tmp_path / "logs"
        logs.mkdir()
        yield logs
    finally:
        _close_mcp_stderr_logs()
        reset_hermes_home_override(token)


def test_oversized_log_rotates_to_one_generation(log_dir):
    live, archived = log_dir / "mcp-stderr.log", log_dir / "mcp-stderr.log.1"
    archived.write_text("older generation\n")
    live.write_bytes(b"x" * (CAP + 1))

    fh = _get_mcp_stderr_log()
    fh.write("fresh line\n")
    fh.flush()

    assert archived.stat().st_size == CAP + 1  # replaced the older .1, not appended to it
    assert live.read_text() == "fresh line\n"
    assert sorted(p.name for p in log_dir.iterdir()) == ["mcp-stderr.log", "mcp-stderr.log.1"]


def test_log_under_cap_keeps_appending(log_dir):
    live = log_dir / "mcp-stderr.log"
    live.write_text("existing content\n")

    fh = _get_mcp_stderr_log()
    fh.write("appended\n")
    fh.flush()

    assert not (log_dir / "mcp-stderr.log.1").exists()
    assert live.read_text() == "existing content\nappended\n"


def test_failed_rotation_still_opens_the_log(log_dir, monkeypatch):
    live = log_dir / "mcp-stderr.log"
    live.write_bytes(b"x" * (CAP + 1))

    def refuse(self, target):
        raise PermissionError("held open elsewhere")

    monkeypatch.setattr(type(live), "replace", refuse)
    fh = _get_mcp_stderr_log()
    fh.write("still logged\n")
    fh.flush()

    assert fh.name == str(live)
    assert live.read_bytes().endswith(b"still logged\n")
