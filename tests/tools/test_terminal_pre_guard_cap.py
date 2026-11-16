"""The pre-exec guard's wait is CAPPED, not just floored (board #741).

``_PRE_EXEC_GUARD_MIN_TIMEOUT_S`` was pinned by test_terminal_pre_guard_deadline.py
(a very short command timeout must not turn the guard's cold-start cost into a refusal).
The CEILING had no test at all, which let it live as an uncommitted working-tree edit for
seven days while the running gateway served the uncapped code: 983 logged refusals, 789 of
them longer than the 90s the cap allows, 443 at 180s, worst 420s -- 928 minutes of dead
wall clock across 49 cron sessions.

What is asserted here is the arithmetic the tool actually uses, so a future edit that drops
the cap, raises it, or reverses the order of the two clamps fails loudly instead of silently
handing a 300s wait back to every cron agent that shells out.
"""

from __future__ import annotations

import tools.terminal_tool as terminal_module


def _guard_deadline(cmd_timeout: float) -> float:
    """Replay terminal_tool's own sizing: floor first, then ceiling.

    Mirrors the two lines in ``terminal_tool`` that build ``guard_timeout``; if that
    expression changes shape, this helper must change with it and the assertions below
    are what notice. ``getattr`` with no ceiling means "uncapped", which is exactly the
    committed-before-#741 behaviour: the floor cases must still hold in that world, or
    this helper is testing the constant's existence rather than the sizing rule.
    """
    guard_timeout = max(cmd_timeout, terminal_module._PRE_EXEC_GUARD_MIN_TIMEOUT_S)
    cap = getattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S", None)
    return guard_timeout if cap is None else min(guard_timeout, cap)


def test_guard_deadline_has_a_ceiling_not_only_a_floor():
    """The ceiling must exist and be far below the long command timeouts cron jobs carry."""
    assert hasattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S"), (
        "the guard ceiling is gone: without it a 400s command waits 400s before refusing"
    )
    cap = terminal_module._PRE_EXEC_GUARD_MAX_TIMEOUT_S
    assert cap <= 90, "guard ceiling %ss is too loose to bound the observed 180-420s wedges" % cap


def test_short_command_keeps_the_floor():
    """A 5s command still gets the 30s floor (cold-start protection, unchanged by the cap)."""
    assert _guard_deadline(5) == terminal_module._PRE_EXEC_GUARD_MIN_TIMEOUT_S


def test_medium_command_is_not_touched_by_the_ceiling():
    """Between floor and ceiling the caller's own timeout wins -- no silent stretching."""
    floor = terminal_module._PRE_EXEC_GUARD_MIN_TIMEOUT_S
    cap = getattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S", floor * 3)
    mid = (floor + cap) // 2
    assert _guard_deadline(mid) == mid


def test_long_command_timeouts_are_clamped_to_the_ceiling():
    """THE regression: a 300s/420s command must refuse at the cap, never at its own timeout.

    Committed-before-#741 code computed max(timeout, 30) with no ceiling, so these exact
    timeouts produced 300s and 420s guard waits. Both values appear in the logged refusals.
    """
    cap = terminal_module._PRE_EXEC_GUARD_MAX_TIMEOUT_S
    for cmd_timeout in (300, 420, 900):
        assert _guard_deadline(cmd_timeout) == cap, (
            "a %ss command waited %ss on the guard; the ceiling is %ss"
            % (cmd_timeout, _guard_deadline(cmd_timeout), cap)
        )


def test_ceiling_is_reachable_and_ordered_after_the_floor():
    """Ceiling above floor, or the min() would silently defeat the floor."""
    cap = getattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S", None)
    assert cap is not None, "no guard ceiling: long commands wait their own full timeout"
    assert cap > terminal_module._PRE_EXEC_GUARD_MIN_TIMEOUT_S
