"""Regression for deeply nested command substitutions in approval scanning."""

from __future__ import annotations

import pytest

from tools.approval_detection import detect_dangerous_command, detect_hardline_command


@pytest.mark.parametrize("suffix", ["", "; true"])
@pytest.mark.parametrize(
    "detect, blocked",
    [
        (detect_hardline_command, lambda result: result[0] and "limit" in result[1]),
        (detect_dangerous_command, lambda result: result[0] and "limit" in result[1]),
    ],
)
def test_deeply_nested_substitutions_fail_closed(suffix, detect, blocked):
    command = "$(" * 1000 + "echo hi" + ")" * 1000 + suffix

    result = detect(command)

    assert blocked(result)


def test_in_limit_nesting_keeps_normal_detection():
    command = "$(" * 10 + "echo hi" + ")" * 10

    assert detect_hardline_command(command) == (False, None)
    assert detect_dangerous_command(command) == (False, None, None)
