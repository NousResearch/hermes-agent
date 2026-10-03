"""Explicit selectors for inert registered-lease test providers only.

This is not a product adapter: fixtures provide already-created leases and a
separate read-only mapping getter when their target selection can change.
"""
from hermes_cli.session_execution import SessionExecutionError, SessionExecutionLease, TargetSelection
from tools.computer_use.session_context import read_access_epoch
from tools.terminal_targets import access_epoch


def lease_selection(lease, *, current=None):
    if not isinstance(lease, SessionExecutionLease):
        raise SessionExecutionError("fixture target unavailable")
    cua_epoch, terminal_epoch = read_access_epoch(lease), access_epoch(lease)
    def check():
        lease.check()
        if current is not None and current() is not lease:
            raise SessionExecutionError("fixture target mapping changed")
        if read_access_epoch(lease) != cua_epoch or access_epoch(lease) != terminal_epoch:
            raise SessionExecutionError("fixture control epoch changed")
    def realize(*, before_start=None):
        if before_start is not None:
            before_start()
        check()
        return lease
    return TargetSelection(realize, check)
