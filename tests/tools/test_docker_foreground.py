"""Failures in ownership housekeeping cannot turn completed side effects into retries."""

from unittest.mock import Mock

import pytest

from tools.environments.docker_foreground import DockerForeground


def test_completed_command_cleanup_survives_container_loss(tmp_path, monkeypatch, caplog):
    record = DockerForeground(Mock(get_temp_dir=lambda: str(tmp_path)))
    monkeypatch.setattr(record, "control", Mock(side_effect=RuntimeError("Container not started")))
    record.cleanup()
    assert "Could not remove Docker foreground ownership record" in caplog.text


def test_failed_namespace_kill_never_claims_verified_termination(tmp_path, monkeypatch):
    record = DockerForeground(Mock(get_temp_dir=lambda: str(tmp_path)))
    monkeypatch.setattr(record, "control", Mock(return_value={"returncode": 1}))
    with pytest.raises(RuntimeError, match="termination could not be verified"):
        record.kill()
