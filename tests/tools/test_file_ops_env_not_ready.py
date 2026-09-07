"""A terminal environment that cannot run commands (container still starting, removed out-of-band,
transport down) must surface as an environment error, never as "File not found": the model trusts
a false negative for the rest of the session (#44750)."""

import pytest

from tools.file_operations import ShellFileOperations


class _DeadEnv:
    cwd = "/workspace"

    def execute(self, command: str, cwd=None, **kwargs) -> dict:
        return {"output": "Error: container is not running", "returncode": 125}


class _HealthyEmptyEnv:
    """Runs commands fine; the filesystem has no files."""
    cwd = "/workspace"

    def execute(self, command: str, cwd=None, **kwargs) -> dict:
        if command.startswith("if [ -f"):
            return {"output": "__hermes_missing__\n", "returncode": 0}
        if command.startswith("test -e"):
            return {"output": "not_found\n", "returncode": 0}
        if command.startswith("echo "):
            return {"output": command[5:].strip() + "\n", "returncode": 0}
        return {"output": "", "returncode": 1}


@pytest.mark.parametrize("op", ["read_file", "read_file_raw", "read_file_bytes", "search"])
def test_dead_environment_is_reported_as_unavailable_not_missing(op):
    ops = ShellFileOperations(_DeadEnv())
    result = getattr(ops, op)("pattern", "state") if op == "search" else getattr(ops, op)("state/notes.md")
    assert result.error and "File not found" not in result.error and "environment unavailable" in result.error.lower()


@pytest.mark.parametrize("op", ["read_file", "read_file_raw", "read_file_bytes"])
def test_healthy_environment_still_reports_missing_files(op):
    result = getattr(ShellFileOperations(_HealthyEmptyEnv()), op)("state/notes.md")
    assert result.error and "File not found" in result.error
