"""Test TUI gateway stdin EINVAL handling (#92284)."""
import errno
import io
import os
from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.entry as entry


class _FakeStdin:
    def __init__(self, error=None, line=""):
        self._error = error
        self._line = line

    def readline(self):
        if self._error:
            raise self._error
        return self._line


def test_einval_returns_none():
    """EINVAL from a PTY-less launcher maps to clean EOF (None)."""
    fake = _FakeStdin(error=OSError(errno.EINVAL, "Invalid argument"))
    with patch.object(entry.sys, "stdin", fake):
        result = entry._read_stdin_line()
    assert result is None


def test_eio_propagates():
    """EIO (master-side close) should propagate, not be swallowed."""
    fake = _FakeStdin(error=OSError(errno.EIO, "I/O error"))
    with patch.object(entry.sys, "stdin", fake):
        with pytest.raises(OSError) as exc_info:
            entry._read_stdin_line()
    assert exc_info.value.errno == errno.EIO


def test_normal_line_passthrough():
    fake = _FakeStdin(line='{"jsonrpc": "2.0"}\n')
    with patch.object(entry.sys, "stdin", fake):
        result = entry._read_stdin_line()
    assert result == '{"jsonrpc": "2.0"}\n'


def test_eof_returns_empty_string():
    """readline() returns '' at EOF — passthrough, not None."""
    fake = _FakeStdin(line="")
    with patch.object(entry.sys, "stdin", fake):
        result = entry._read_stdin_line()
    assert result == ""
