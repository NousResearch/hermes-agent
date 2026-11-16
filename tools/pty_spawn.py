"""Start the child of a ``terminal(background=true, pty=true)`` job without relaying its signals to us.

ptyprocess forks in Python, so until exec the child runs this interpreter's signal handlers, and
until asyncio's own ``after_in_child`` fork hook clears it, the child also keeps our event loop's
wakeup fd. A signal that hits the child in that post-fork window is written to OUR loop, which
handles it as its own: in the gateway a SIGTERM aimed at the child (an OOM killer picks it, since
it carries our RSS) shut the whole gateway down, and SIGUSR1 restarted it. After the hook the
child still has asyncio's no-op handler, so it swallows the signal and keeps running until exec.

The spawning thread blocks every asynchronous signal across the fork; in ``preexec_fn`` the child
puts the handlers it inherited back to SIG_DFL, drops the wakeup fd and restores the mask, so a
signal that arrived meanwhile kills the child with its default action. Synchronous fault signals
stay unblocked: blocking one the thread itself raises would kill the process outright.
"""

import signal
import sys
from contextlib import suppress

_IS_WINDOWS = sys.platform == "win32"
_FAULT_SIGNALS = ("SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL")
_BLOCKED_ACROSS_FORK = (
    set() if _IS_WINDOWS
    else signal.valid_signals() - {getattr(signal, name) for name in _FAULT_SIGNALS if hasattr(signal, name)})


def _reset_inherited_signal_handlers(saved_mask) -> None:
    """``preexec_fn``: runs in the child after ptyprocess's fd sweep, before exec. Signals the parent
    ignores stay ignored, as across exec."""
    for sig in signal.valid_signals():
        with suppress(OSError, ValueError, RuntimeError):
            if callable(signal.getsignal(sig)):
                signal.signal(sig, signal.SIG_DFL)
    with suppress(OSError, ValueError, RuntimeError):
        signal.set_wakeup_fd(-1)
    signal.pthread_sigmask(signal.SIG_SETMASK, saved_mask)


def spawn_pty(argv, *, cwd, env, dimensions=(30, 120)):
    """``PtyProcess.spawn`` from pywinpty (Windows) or ptyprocess. Raises ImportError when the
    backend is not installed."""
    if _IS_WINDOWS:
        from winpty import PtyProcess
        return PtyProcess.spawn(argv, cwd=cwd, env=env, dimensions=dimensions)
    from ptyprocess import PtyProcess
    saved_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _BLOCKED_ACROSS_FORK)
    try:
        return PtyProcess.spawn(argv, cwd=cwd, env=env, dimensions=dimensions,
                                preexec_fn=lambda: _reset_inherited_signal_handlers(saved_mask))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, saved_mask)
