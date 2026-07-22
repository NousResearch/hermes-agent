"""Tests for tui_gateway.transport — _PEER_GONE_ERRNOS and StdioTransport."""

import errno
import json
import io
import pytest

from tui_gateway.transport import _PEER_GONE_ERRNOS, StdioTransport


class TestPeerGoneErrnos:
    """Verify that _PEER_GONE_ERRNOS contains the expected errno values."""

    def test_einval_in_peer_gone_errnos(self):
        """EINVAL (22) must be in the set for Windows detached-stdout."""
        assert errno.EINVAL in _PEER_GONE_ERRNOS

    def test_epipe_in_peer_gone_errnos(self):
        """EPIPE must be in the set for POSIX closed-pipe."""
        assert errno.EPIPE in _PEER_GONE_ERRNOS

    def test_peer_gone_errnos_is_frozenset(self):
        """_PEER_GONE_ERRNOS must be immutable."""
        assert isinstance(_PEER_GONE_ERRNOS, frozenset)

    def test_connreset_in_peer_gone_errnos(self):
        """ECONNRESET must be in the set."""
        assert errno.ECONNRESET in _PEER_GONE_ERRNOS

    def test_ebadf_in_peer_gone_errnos(self):
        """EBADF must be in the set."""
        assert errno.EBADF in _PEER_GONE_ERRNOS

    def test_eshutdown_in_peer_gone_errnos(self):
        """ESHUTDOWN must be in the set."""
        assert errno.ESHUTDOWN in _PEER_GONE_ERRNOS


class TestStdioTransportWrite:
    """Verify StdioTransport.write() handles peer-gone errnos correctly."""

    def test_write_einval_returns_false(self):
        """OSError(22) should return False (peer gone), not raise."""
        import threading
        stream = io.StringIO()
        lock = threading.Lock()
        transport = StdioTransport(lambda: stream, lock)

        def raise_einval(line):
            raise OSError(errno.EINVAL, "Invalid argument")

        stream.write = raise_einval
        result = transport.write({"test": "data"})
        assert result is False

    def test_write_epipe_returns_false(self):
        """OSError(EPIPE) should return False (peer gone), not raise."""
        import threading
        stream = io.StringIO()
        lock = threading.Lock()
        transport = StdioTransport(lambda: stream, lock)

        def raise_epipe(line):
            raise OSError(errno.EPIPE, "Broken pipe")

        stream.write = raise_epipe
        result = transport.write({"test": "data"})
        assert result is False

    def test_write_enospc_raises(self):
        """OSError(ENOSPC) should raise (real host problem), not return False."""
        import threading
        stream = io.StringIO()
        lock = threading.Lock()
        transport = StdioTransport(lambda: stream, lock)

        def raise_enospc(line):
            raise OSError(errno.ENOSPC, "No space left on device")

        stream.write = raise_enospc
        with pytest.raises(OSError) as exc_info:
            transport.write({"test": "data"})
        assert exc_info.value.errno == errno.ENOSPC

    def test_write_success_returns_true(self):
        """Successful write should return True."""
        import threading
        stream = io.StringIO()
        lock = threading.Lock()
        transport = StdioTransport(lambda: stream, lock)
        result = transport.write({"test": "data"})
        assert result is True
