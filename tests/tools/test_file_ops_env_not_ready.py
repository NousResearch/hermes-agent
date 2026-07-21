"""Tests for environment-failure disambiguation in tools/file_operations.py.

When the terminal environment cannot execute commands at all (docker container
still starting, container removed out-of-band, transport error), file
existence probes exit non-zero exactly like a missing file would. These tests
guard the disambiguation added for that case:

- ``read_file`` / ``read_file_raw`` must report an environment error, not
  "File not found", when the environment is down.
- ``search`` must report an environment error when its existence probe
  returns neither the ``exists`` nor the ``not_found`` marker.
- Genuinely missing files must still produce "File not found" when the
  environment is healthy.

Regression context: with ``terminal.backend: docker`` and non-persistent
containers, the first ``read_file`` of a session raced container startup and
returned a false "File not found", which the model then trusted for the rest
of the session.
"""

import pytest

from tools.file_operations import ShellFileOperations


class _DeadEnv:
    """Environment whose execute() always fails (e.g. container starting)."""

    cwd = "/workspace"

    def execute(self, command: str, cwd: str = None, **kwargs) -> dict:
        return {"output": "Error: container is not running", "returncode": 125}


class _HealthyEnvWithoutFile:
    """Environment that runs commands fine but has no files in it."""

    cwd = "/workspace"

    def execute(self, command: str, cwd: str = None, **kwargs) -> dict:
        if command.startswith("echo "):
            return {"output": command[len("echo "):].strip() + "\n", "returncode": 0}
        if command.startswith("test -e"):
            return {"output": "not_found\n", "returncode": 0}
        if command.startswith("test -d"):
            return {"output": "no\n", "returncode": 0}
        # wc -c, ls, head, ... — behave like an empty filesystem
        return {"output": "", "returncode": 1}


@pytest.fixture()
def dead_ops():
    return ShellFileOperations(_DeadEnv())


@pytest.fixture()
def healthy_ops():
    return ShellFileOperations(_HealthyEnvWithoutFile())


class TestReadFileEnvNotReady:
    def test_read_file_reports_env_failure_not_missing_file(self, dead_ops):
        result = dead_ops.read_file("state/notes.md")
        assert result.error
        assert "File not found" not in result.error
        assert "environment unavailable" in result.error.lower()

    def test_read_file_raw_reports_env_failure_not_missing_file(self, dead_ops):
        result = dead_ops.read_file_raw("state/notes.md")
        assert result.error
        assert "File not found" not in result.error
        assert "environment unavailable" in result.error.lower()

    def test_read_file_still_reports_missing_file_when_env_healthy(self, healthy_ops):
        result = healthy_ops.read_file("state/notes.md")
        assert result.error
        assert "File not found" in result.error

    def test_env_ready_true_on_healthy_env(self, healthy_ops):
        assert healthy_ops._env_ready() is True

    def test_env_ready_false_on_dead_env(self, dead_ops):
        assert dead_ops._env_ready() is False


class TestSearchEnvNotReady:
    def test_search_reports_env_failure_when_probe_returns_no_marker(self, dead_ops):
        result = dead_ops.search("pattern", path="/workspace")
        assert result.error
        assert "environment unavailable" in result.error.lower()
        assert "Path not found" not in (result.error or "")

    def test_search_still_reports_missing_path_when_env_healthy(self, healthy_ops):
        result = healthy_ops.search("pattern", path="/workspace/nope")
        assert result.error
        assert "Path not found" in result.error
