"""POSIX fd-readiness helper shared by the stdout drains and the PTY bridge.

``select.select()`` raises ``ValueError: filedescriptor out of range in select()``
once an fd reaches FD_SETSIZE (1024). A long-lived gateway/desktop process holds
that many open fds, and every drain swallowed that ValueError as "fd already
closed", silently discarding output (issue #94928).

``selectors.DefaultSelector`` (epoll on Linux, kqueue on macOS) has no FD_SETSIZE
ceiling. ``poll()`` is not the default: macOS ``poll()`` does not support device
fds and reports a PTY master as POLLNVAL, i.e. "ready" at once, so a PTY read loop
would spin a core. It stays the fallback because, unlike epoll/kqueue, it needs no
fd of its own.
"""

import errno
import os
import selectors
import stat
import sys


class FdWaiter:
    """Readability wait on one fd. Owns a selector (epoll/kqueue are fds themselves):
    create it once per reader lifetime and ``close()`` it when the reader is done."""

    def __init__(self, fd: int, selector_cls):
        self._selector = selector_cls()
        try:
            self._selector.register(fd, selectors.EVENT_READ)
        except BaseException:
            self._selector.close()
            raise

    def wait(self, timeout_s: float) -> bool:
        """True as soon as the fd is readable (EOF and hangup included), False when
        *timeout_s* elapses idle."""
        return bool(self._selector.select(max(timeout_s, 0)))

    def close(self) -> None:
        self._selector.close()


def open_fd_waiter(fd: int):
    """Return an :class:`FdWaiter` for *fd*, or None only when *fd* is closed/invalid
    and there is nothing to drain. Raises OSError when no safe wait exists."""
    try:
        return FdWaiter(fd, selectors.DefaultSelector)
    except ValueError:
        return None
    except OSError as exc:
        if exc.errno == errno.EBADF:
            return None
        # The selector itself could not be set up (EMFILE: the process is at its fd limit)
        # while the fd is still live. poll() needs no extra fd; never report that as EOF.
        # On macOS it is no wait at all for a device (PTY): raise rather than spin.
        if sys.platform == "darwin" and stat.S_ISCHR(os.fstat(fd).st_mode):
            raise
    return FdWaiter(fd, selectors.PollSelector)
