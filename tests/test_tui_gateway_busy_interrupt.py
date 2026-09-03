"""Regression guard for the TUI gateway's busy-state error messaging.

Hermes has no ``/interrupt`` slash command — the way to interrupt an in-flight
turn is the Ctrl+C gesture. A busy-guard error that tells the user to
"``/interrupt`` the current turn" therefore points at a command that does not
exist, leaving them stuck with no obvious way forward (issue #42093).

Most of the gateway's former "session busy" rejections have since been
replaced by a queue-until-idle mechanism, but the compute-host
mutate-while-running guard in ``_mirror_slash_side_effects`` still surfaces a
plain rejection string to the user. These tests pin its wording so the
misleading ``/interrupt`` phrasing cannot creep back in.
"""

import inspect

import tui_gateway.server as server


def test_busy_guard_does_not_reference_phantom_interrupt_command():
    src = inspect.getsource(server)
    assert "/interrupt the current turn" not in src, (
        "tui_gateway busy-guard references a non-existent /interrupt slash "
        "command; point users at the Ctrl+C gesture instead (issue #42093)."
    )


def test_busy_guard_points_at_ctrl_c():
    src = inspect.getsource(server)
    assert "press Ctrl+C to interrupt the current turn" in src, (
        "expected the mutate-while-running busy guard to direct users to "
        "Ctrl+C (issue #42093)."
    )
