"""A shell status cannot be attributed to a command that may never have run."""

import subprocess

import pytest

from tools.terminal_tool_result import _interpret_exit_code


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("command", [
    "false && grep pattern unused-file",
    "grep pattern < /nonexistent-forge-exit-fixture",
    "false && diff unused-a unused-b",
    "false && test -e unused-file",
])
def test_shell_failure_is_not_described_as_benign_tool_result(command):
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, timeout=5, check=False)
    assert result.returncode == 1
    assert _interpret_exit_code(command, result.returncode) is None


@pytest.mark.parametrize("command", [
    "grep 'left|right' file.txt",
    'grep "left && right" file.txt',
    "LC_ALL=C /usr/bin/grep pattern file.txt",
])
def test_simple_grep_keeps_no_match_explanation(command):
    assert "no matches" in _interpret_exit_code(command, 1).lower()


@pytest.mark.parametrize("command", [
    "producer | grep pattern",
    "false || grep pattern",
    "echo ready; grep pattern",
    "echo ready\ngrep pattern",
    'grep "$(producer)" file.txt',
])
def test_compound_status_does_not_guess_which_program_failed(command):
    assert _interpret_exit_code(command, 1) is None
