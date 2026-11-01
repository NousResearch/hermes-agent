"""Regression coverage for large commands with sibling substitutions."""

import os
import subprocess
import sys

import pytest

from tools.approval_detection import detect_dangerous_command

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _assert_detection_is_bounded(command: str, detector: str) -> None:
    code = f"""
from tools.approval_detection import {detector}
result = {detector}({command!r})
assert result == (False, None), result
"""
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    subprocess.run(
        [sys.executable, "-c", code], check=True, timeout=5, cwd=REPO_ROOT, env=env
    )


def test_hardline_detection_is_bounded_for_sibling_backticks():
    command = "rm " + "`a` " * 500 + ";"
    _assert_detection_is_bounded(command, "detect_hardline_command")


def test_hardline_detection_is_bounded_for_sibling_dollar_substitutions():
    command = "rm " + "$(a) " * 2500 + ";"
    _assert_detection_is_bounded(command, "detect_hardline_command")


@pytest.mark.parametrize(
    "substitution", ["`bash -c 'echo ok'`", "$(bash -c 'echo ok')"]
)
def test_shell_execution_inside_substitution_is_detected(substitution: str):
    result = detect_dangerous_command(f"echo {substitution}")

    assert result[0] is True
    assert result[1] == "shell command via -c/-lc flag"
