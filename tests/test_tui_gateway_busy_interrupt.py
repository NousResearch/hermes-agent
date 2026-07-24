"""Busy-guard messages must point at the real interrupt gesture (issue #42093).

Several TUI commands reject input while a turn is running. They used to tell the
user to run ``/interrupt`` first — but ``/interrupt`` is not a registered slash
command, so that was a dead end. The real gesture is Ctrl+C (see the composer's
``Ctrl+C to interrupt…`` hint), which the busy-guard messages must reference.
"""

import inspect

from hermes_cli.commands import resolve_command
from tui_gateway import server


def test_interrupt_is_not_a_registered_command():
    """The premise of the bug: busy errors pointed at a command that isn't real."""
    assert resolve_command("interrupt") is None


def test_busy_error_points_to_ctrl_c_not_slash_interrupt():
    msg = server._busy_interrupt_error("/undo")
    assert "Ctrl+C" in msg
    assert "/interrupt" not in msg


def test_busy_error_preserves_action_suffix():
    assert server._busy_interrupt_error("switching models").endswith(
        "before switching models"
    )
    assert "/retry" in server._busy_interrupt_error("/retry")


def test_no_busy_message_references_unregistered_interrupt_command():
    """Regression guard: no busy-guard string may tell users to run /interrupt.

    All seven busy messages route through ``_busy_interrupt_error``; this catches
    a future inline string that bypasses the helper and reintroduces the dead end.
    """
    assert "/interrupt the current turn" not in inspect.getsource(server)
