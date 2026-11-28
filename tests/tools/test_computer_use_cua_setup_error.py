"""cua-driver session-setup failures must name their root cause.

The MCP session is opened inside an anyio TaskGroup, so ``_setup_error`` arrives as a
``(Base)ExceptionGroup`` whose ``str()`` is ``"unhandled errors in a TaskGroup (1
sub-exception)"``. That string used to be handed straight to the model and the user
while the driver had already reported the real fault, e.g.::

    cua-driver-rs: invalid daemon response: incompatible daemon:
    contract version 0.8.0 does not match SDK 0.7.0

These tests pin the unwrapping so a future refactor cannot regress the diagnostic back
into an opaque group repr.
"""

from __future__ import annotations

import asyncio

import pytest

from tools.computer_use.cua_backend_session import _setup_error_detail

OPAQUE = "unhandled errors in a TaskGroup"


class _FakeMcpError(Exception):
    """Stands in for ``mcp.shared.exceptions.MCPError`` without importing the SDK."""


def test_contract_skew_surfaces_leaf_and_names_the_fix():
    leaf = _FakeMcpError(
        "cua-driver-rs: invalid daemon response: incompatible daemon: "
        "contract version 0.8.0 does not match SDK 0.7.0"
    )
    detail = _setup_error_detail(BaseExceptionGroup("unhandled errors in a TaskGroup", [leaf]))

    assert OPAQUE not in detail, "the opaque group repr must not reach the user"
    assert "contract version 0.8.0 does not match SDK 0.7.0" in detail
    assert "_FakeMcpError" in detail
    # Actionable next step, because this failure is fixed by updating the daemon.
    assert "cua-driver update" in detail
    assert "computer-use doctor" in detail


def test_plain_exception_is_reported_with_its_type():
    detail = _setup_error_detail(RuntimeError("driver exited before initialize"))

    assert detail == "RuntimeError: driver exited before initialize"
    assert OPAQUE not in detail


def test_nested_groups_are_unwrapped_to_the_leaf():
    leaf = FileNotFoundError("cua-driver binary not found")
    inner = BaseExceptionGroup("unhandled errors in a TaskGroup", [leaf])
    outer = BaseExceptionGroup("unhandled errors in a TaskGroup", [inner])

    detail = _setup_error_detail(outer)

    assert "cua-driver binary not found" in detail
    assert OPAQUE not in detail


def test_cancellation_noise_does_not_mask_the_real_error():
    """anyio sprays CancelledError onto sibling tasks; that must not win."""
    real = _FakeMcpError("Connection closed")
    group = BaseExceptionGroup(
        "unhandled errors in a TaskGroup",
        [asyncio.CancelledError(), real],
    )

    detail = _setup_error_detail(group)

    assert "Connection closed" in detail
    assert "CancelledError" not in detail


def test_keyboard_interrupt_still_propagates():
    """A KeyboardInterrupt leaf is re-raised by the shared unwrapper, never flattened."""
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [KeyboardInterrupt()])

    with pytest.raises(KeyboardInterrupt):
        _setup_error_detail(group)
