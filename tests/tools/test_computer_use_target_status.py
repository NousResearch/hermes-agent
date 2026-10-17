"""Target resolution errors must remain diagnostic on the readiness surfaces (#115033)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli.config import save_config
from tools.computer_use import cua_backend_driver as driver, permissions


@pytest.mark.parametrize("target", ["mac", "unsupported-host"])
def test_target_errors_preserve_status_payload_and_recover(tmp_path, monkeypatch, capsys, target):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    # Exercise a native host without WSL interop, without pretending to be another OS.
    monkeypatch.setattr("hermes_constants.is_wsl", lambda: False)
    binary = str(tmp_path / "missing-driver")
    probe = Mock(side_effect=AssertionError("invalid or missing drivers must not be spawned"))
    monkeypatch.setattr(permissions, "_run", probe)
    monkeypatch.setattr(permissions.subprocess, "run", probe)
    save_config({"computer_use": {"target": "auto"}})
    missing = permissions.computer_use_status(binary)
    assert missing["installed"] is False and missing["ready"] is None
    assert missing["error"] is None
    if target == "unsupported-host":
        target = "windows" if sys.platform == "linux" else "linux"
    save_config({"computer_use": {"target": target}})
    with pytest.raises(ValueError) as invalid:
        driver.resolve_cua_driver_cmd(binary)
    diagnostic = str(invalid.value)
    status = permissions.computer_use_status(binary)
    assert list(status) == list(missing)
    assert status == {**missing, "error": diagnostic}
    if sys.platform == "darwin":
        assert permissions.request_permissions_grant(binary) == 2
        assert diagnostic in capsys.readouterr().err
    else:
        assert permissions.request_permissions_grant(binary) == 64
    save_config({"computer_use": {"target": "auto"}})
    assert permissions.computer_use_status(binary) == missing
    probe.assert_not_called()
