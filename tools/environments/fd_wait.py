"""POSIX fd-readiness helper shared by the stdout drains.

``select.select()`` raises ``ValueError: filedescriptor out of range in select()``
the moment a pipe fd reaches FD_SETSIZE (1024); a long-lived gateway/desktop
process holds that many open fds, and both drains swallowed that ValueError as
"fd already closed", silently discarding output with no error signal (issue
#94928). ``poll()`` has no FD_SETSIZE ceiling, and POLLHUP/POLLERR count as
readable so the caller's read observes the real EOF or broken descriptor.
"""

import select


def make_fd_waiter(fd: int):
    """Return a ``wait(timeout_ms) -> bool`` readability waiter for *fd*.

    Returns None when *fd* is already closed and there is nothing to drain.
    ``wait`` returns True as soon as *fd* is readable (POLLHUP/POLLERR included,
    so the caller's read observes EOF), False when *timeout_ms* elapses idle.
    """
    poller = select.poll()
    try:
        poller.register(fd, select.POLLIN)
    except (ValueError, OSError):
        return None

    def wait(timeout_ms: int) -> bool:
        return bool(poller.poll(timeout_ms))

    return wait
