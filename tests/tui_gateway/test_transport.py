"""Tests for tui_gateway.transport -- Transport protocol, ContextVar helpers,
StdioTransport, and TeeTransport."""

import errno
import json
import threading

import pytest
from unittest.mock import MagicMock, patch

from tui_gateway.transport import (
    StdioTransport,
    TeeTransport,
    current_transport,
    bind_transport,
    reset_transport,
)


@pytest.fixture(autouse=True)
def _clean_contextvar():
    """Reset the transport contextvar before and after each test."""
    token = bind_transport(None)
    yield
    reset_transport(token)


@pytest.fixture()
def mock_stream():
    return MagicMock()


@pytest.fixture()
def stdio_transport(mock_stream):
    return StdioTransport(lambda: mock_stream, threading.Lock())


# -- ContextVar helpers -------------------------------------------------------

def test_current_transport_default_none():
    assert current_transport() is None


def test_bind_and_current():
    transport = MagicMock()
    bind_transport(transport)
    assert current_transport() is transport


def test_reset_transport():
    transport = MagicMock()
    token = bind_transport(transport)
    assert current_transport() is transport
    reset_transport(token)
    assert current_transport() is None


# -- StdioTransport.write: success path ----------------------------------------

def test_write_success(stdio_transport, mock_stream):
    payload = {"jsonrpc": "2.0", "method": "test"}
    result = stdio_transport.write(payload)
    assert result is True
    expected_line = json.dumps(payload, ensure_ascii=False) + "\n"
    mock_stream.write.assert_called_once_with(expected_line)
    mock_stream.flush.assert_called_once()


# -- StdioTransport.write: write-side errors -----------------------------------

def test_write_broken_pipe_on_write(stdio_transport, mock_stream):
    mock_stream.write.side_effect = BrokenPipeError()
    assert stdio_transport.write({"a": 1}) is False


def test_write_closed_file_valueerror(stdio_transport, mock_stream):
    mock_stream.write.side_effect = ValueError("I/O operation on closed file")
    assert stdio_transport.write({"a": 1}) is False


def test_write_unicode_encode_error_reraises(stdio_transport, mock_stream):
    mock_stream.write.side_effect = UnicodeEncodeError("utf-8", "", 0, 1, "test")
    with pytest.raises(UnicodeEncodeError):
        stdio_transport.write({"a": 1})


def test_write_other_valueerror_reraises(stdio_transport, mock_stream):
    mock_stream.write.side_effect = ValueError("something else entirely")
    with pytest.raises(ValueError, match="something else entirely"):
        stdio_transport.write({"a": 1})


def test_write_peer_gone_oserror(stdio_transport, mock_stream):
    err = OSError()
    err.errno = errno.EPIPE
    mock_stream.write.side_effect = err
    assert stdio_transport.write({"a": 1}) is False


def test_write_non_peer_oserror_reraises(stdio_transport, mock_stream):
    err = OSError()
    err.errno = errno.ENOSPC
    mock_stream.write.side_effect = err
    with pytest.raises(OSError):
        stdio_transport.write({"a": 1})


# -- StdioTransport.write: flush-side errors -----------------------------------

def test_flush_broken_pipe(stdio_transport, mock_stream):
    mock_stream.flush.side_effect = BrokenPipeError()
    assert stdio_transport.write({"a": 1}) is False


def test_flush_peer_gone_oserror(stdio_transport, mock_stream):
    err = OSError()
    err.errno = errno.EPIPE
    mock_stream.flush.side_effect = err
    assert stdio_transport.write({"a": 1}) is False


# -- StdioTransport: flush skip ------------------------------------------------

def test_write_skips_flush_when_disabled(stdio_transport, mock_stream):
    with patch("tui_gateway.transport._DISABLE_FLUSH", True):
        result = stdio_transport.write({"a": 1})
    assert result is True
    mock_stream.write.assert_called_once()
    mock_stream.flush.assert_not_called()


# -- StdioTransport.close -----------------------------------------------------

def test_close_noop(stdio_transport):
    stdio_transport.close()


# -- TeeTransport -------------------------------------------------------------

def test_tee_write_primary_success():
    primary = MagicMock()
    primary.write.return_value = True
    secondary = MagicMock()
    tee = TeeTransport(primary, secondary)
    assert tee.write({"x": 1}) is True
    primary.write.assert_called_once_with({"x": 1})
    secondary.write.assert_called_once_with({"x": 1})


def test_tee_write_primary_false():
    primary = MagicMock()
    primary.write.return_value = False
    secondary = MagicMock()
    tee = TeeTransport(primary, secondary)
    assert tee.write({"x": 1}) is False
    secondary.write.assert_called_once_with({"x": 1})


def test_tee_write_secondary_exception_swallowed():
    primary = MagicMock()
    primary.write.return_value = True
    secondary = MagicMock()
    secondary.write.side_effect = RuntimeError("sidecar crash")
    tee = TeeTransport(primary, secondary)
    assert tee.write({"x": 1}) is True


def test_tee_close_all():
    primary = MagicMock()
    secondary = MagicMock()
    tee = TeeTransport(primary, secondary)
    tee.close()
    primary.close.assert_called_once()
    secondary.close.assert_called_once()


def test_tee_close_primary_raises_secondaries_still_close():
    primary = MagicMock()
    primary.close.side_effect = RuntimeError("primary close failed")
    secondary = MagicMock()
    tee = TeeTransport(primary, secondary)
    with pytest.raises(RuntimeError, match="primary close failed"):
        tee.close()
    secondary.close.assert_called_once()
