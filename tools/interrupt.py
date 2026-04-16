"""Shared interrupt signaling for all tools.

Provides a global threading.Event that any tool can check to determine
if the user has requested an interrupt. The agent's interrupt() method
sets this event, and tools poll it during long-running operations.

Usage in tools:
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"output": "[interrupted]", "returncode": 130}
"""

import threading

_interrupt_event = threading.Event()


def set_interrupt(active: bool) -> None:
    """Called by the agent to signal or clear the interrupt."""
    if active:
        _interrupt_event.set()
    else:
        _interrupt_event.clear()


def is_interrupted() -> bool:
    """Check if an interrupt has been requested. Safe to call from any thread."""
    return _interrupt_event.is_set()
