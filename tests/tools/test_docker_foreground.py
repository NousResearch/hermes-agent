"""Failures in ownership housekeeping cannot turn completed side effects into retries."""

from unittest.mock import Mock
import subprocess

import pytest

from tools.environments.docker_foreground import DockerForeground


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("command,expected,cleanup_fails", [(":", 0, False), ("printf result; exit 7", 7, False),
                                                         ("printf result; exit 7", 7, True)])
def test_wrapper_cleans_ownership_without_changing_command_outcome(tmp_path, command, expected, cleanup_fails):
    record = DockerForeground(Mock(get_temp_dir=lambda: str(tmp_path)))
    prefix = "rm() { return 1; }; " if cleanup_fails else ""
    result = subprocess.run(["bash", "-c", prefix + record.wrap(command)], capture_output=True, text=True, check=False)
    assert result.returncode == expected
    assert result.stdout == ("result" if expected else "")
    assert (tmp_path / record.directory).exists() is cleanup_fails


def test_failed_namespace_kill_never_claims_verified_termination(tmp_path, monkeypatch):
    record = DockerForeground(Mock(get_temp_dir=lambda: str(tmp_path)))
    monkeypatch.setattr(record, "control", Mock(return_value={"returncode": 1}))
    with pytest.raises(RuntimeError, match="termination could not be verified"):
        record.kill()
