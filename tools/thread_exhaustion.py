"""Thread exhaustion detection and classification.

Helpers for detecting and classifying RuntimeError when OS thread creation
fails due to exhaustion of per-process or system-wide thread limits.

See also:
- agent/error_classifier.py for structured error classification
- agent/tool_executor.py for the submit-guard pattern
"""

import logging

logger = logging.getLogger(__name__)


def is_thread_exhaustion_error(exc: BaseException) -> bool:
    """Return True if exc is a RuntimeError from OS thread creation failure.
    
    Python's threading.Thread.start() raises RuntimeError with "can't start
    new thread" when the OS refuses to create a new thread (EAGAIN from
    pthread_create on Linux, similar on macOS/Windows). This is distinct
    from interpreter shutdown ("cannot schedule new futures after interpreter
    shutdown") and should not retry — the agent is out of threads.
    
    Args:
        exc: The exception to check.
        
    Returns:
        True if exc is a thread-creation RuntimeError, False otherwise.
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    # The canonical error text from CPython's threadmodule.c:
    # "can't start new thread" (Python 3.8+) or "can't create new thread"
    # (pre-3.8). Match both plus variations from different platforms.
    return (
        "can't start new thread" in msg
        or "can't create new thread" in msg
        or "cannot start new thread" in msg
        or "unable to create new thread" in msg
    )
