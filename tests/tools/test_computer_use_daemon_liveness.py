"""The machine-wide ``cua-driver serve`` daemon is a separate failure domain from the driver binary.

The unit that starts the daemon keeps a *concrete* binary path in ``ExecStart``. The upstream installer
repoints ``packages/current`` on upgrade and prunes the versioned release directories it replaced (last 5
kept), so a unit written against ``packages/releases/<version>/cua-driver`` becomes a 203/EXEC crash loop
that no version/platform/session/AX check can see — 46h and ~45k journal lines of silent unavailability,
while ``computer-use doctor`` still called the binary healthy and the one-shot auto-repair reinstalled the
driver (writing ``.release_installed/<ver>``, which made the install look repaired).

Pinned here: the diagnosis (dead daemon + pruned target path, with the fix), the "driver not installed"
vs "daemon not running" split, and the protection case — a host with no daemon configured gains no new
failure item.
"""

from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

import pytest

from tools.computer_use import cua_backend, cua_backend_driver, cua_daemon_health, doctor

_UNIT = "[Unit]\nDescription=cua-driver screenshot daemon\n\n[Service]\nExecStart={exec_start}\nRestart=always\n"
_PRUNED_VERSION = "0.20.0-x86_64-unknown-linux-gnu"


def _fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """HOME (POSIX) and USERPROFILE (native Windows) both, so ``expanduser('~')`` lands in tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _serve_unit(monkeypatch: pytest.MonkeyPatch, tmp_path, exec_start: str, name: str = "cua-driver-screenshot.service"):
    """Write one unit where a real user unit lives, and confine discovery to it (never the host's own dirs)."""
    unit_dir = tmp_path / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    path = unit_dir / name
    path.write_text(_UNIT.format(exec_start=exec_start), encoding="utf-8")
    monkeypatch.setattr(cua_daemon_health, "_reference_globs", lambda: [str(unit_dir / "*.service")])
    return path


def _versioned_exec_start() -> str:
    return (f"%h/.cua-driver/packages/releases/{_PRUNED_VERSION}/cua-driver serve "
            "--socket %h/.cache/cua-driver/cua-driver.sock")


def _healthy_report() -> dict:
    """A report with nothing wrong on the binary side — exactly what doctor reported during the outage."""
    return {
        "schema_version": "1",
        "platform": "linux",
        "driver_version": "0.28.2",
        "overall": "ok",
        "checks": [{"name": "binary_version", "status": "pass", "message": "cua-driver 0.28.2"}],
    }


def _pin_driver(monkeypatch, binary: str) -> None:
    """Both resolvers: ``doctor``'s own (module attribute) and the daemon module's binding."""
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda *a, **k: binary)
    monkeypatch.setattr(cua_daemon_health, "resolve_cua_driver_cmd", lambda *a, **k: binary)


def test_pruned_release_path_is_reported_as_a_dead_daemon(monkeypatch, tmp_path):
    """ExecStart pointing into a pruned ``packages/releases/<ver>/`` dir = dead daemon, named explicitly."""
    home = _fake_home(monkeypatch, tmp_path)
    pruned = home / ".cua-driver" / "packages" / "releases" / _PRUNED_VERSION / "cua-driver"
    _serve_unit(monkeypatch, tmp_path, _versioned_exec_start())
    _pin_driver(monkeypatch, "/usr/local/bin/cua-driver")
    probe = Mock(return_value=False)
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", probe)

    status = cua_daemon_health.cua_driver_daemon_status()

    assert status["state"] == "not_running"
    assert [os.path.normpath(p) for p in status["missing_targets"]] == [os.path.normpath(str(pruned))]
    assert "does not exist" in status["reason"]
    assert "packages/current/cua-driver" in status["hint"]  # the stable launcher, never the versioned path
    assert "systemctl --user daemon-reload" in status["hint"]
    assert "will NOT fix this" in status["hint"]  # a reinstall is not the fix, and must not read as one
    probe.assert_not_called()  # a pruned target is proof enough — no extra `cua-driver status` spawn


def test_dead_daemon_is_probed_on_the_socket_the_unit_configures(monkeypatch, tmp_path):
    """Target exists but nothing listens: probe the configured socket, and never suggest a reinstall."""
    home = _fake_home(monkeypatch, tmp_path)
    target = home / ".cua-driver" / "packages" / "current" / "cua-driver"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _serve_unit(monkeypatch, tmp_path,
                "~/.cua-driver/packages/current/cua-driver serve --socket %h/.cache/cua-driver/cua-driver.sock")
    _pin_driver(monkeypatch, str(target))
    probe = Mock(return_value=False)
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", probe)

    status = cua_daemon_health.cua_driver_daemon_status()

    assert status["state"] == "not_running"
    assert status["missing_targets"] == []
    assert os.path.normpath(status["socket"]) == os.path.normpath(str(home / ".cache" / "cua-driver" / "cua-driver.sock"))
    assert probe.call_args.args[1] == status["socket"]
    assert "no cua-driver daemon is listening" in status["reason"]
    assert "journalctl --user -u cua-driver-screenshot.service" in status["hint"]
    assert "install" not in status["hint"].lower()


def test_running_daemon_and_indeterminate_probe_are_never_a_failure(monkeypatch, tmp_path):
    home = _fake_home(monkeypatch, tmp_path)
    target = home / ".cua-driver" / "packages" / "current" / "cua-driver"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _serve_unit(monkeypatch, tmp_path, "~/.cua-driver/packages/current/cua-driver serve")
    _pin_driver(monkeypatch, str(target))

    monkeypatch.setattr(cua_daemon_health, "daemon_probe", Mock(return_value=True))
    assert cua_daemon_health.cua_driver_daemon_status()["state"] == "running"
    # None = the driver predates `status` / could not be asked — never reported as a dead daemon.
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", Mock(return_value=None))
    assert cua_daemon_health.cua_driver_daemon_status()["state"] == "unknown"


def test_host_without_a_serve_reference_is_not_probed(monkeypatch, tmp_path):
    """No unit starts a daemon here: nothing to probe, nothing to fail (the MCP runtime spawns its own)."""
    _fake_home(monkeypatch, tmp_path)
    monkeypatch.setattr(cua_daemon_health, "_reference_globs",
                        lambda: [str(tmp_path / ".config" / "systemd" / "user" / "*.service")])
    probe = Mock(return_value=False)
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", probe)

    status = cua_daemon_health.cua_driver_daemon_status()

    assert status["state"] == "not_configured" and status["configured"] is False
    probe.assert_not_called()


def test_uninstalled_driver_is_distinguished_from_a_dead_daemon(monkeypatch, tmp_path):
    """A unit referencing a driver that is not installed says "install"; that is not the daemon case."""
    _fake_home(monkeypatch, tmp_path)
    _serve_unit(monkeypatch, tmp_path, "cua-driver serve")
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda *a, **k: None)
    monkeypatch.setattr(cua_daemon_health, "resolve_cua_driver_cmd", lambda *a, **k: None)
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", Mock(return_value=None))

    status = cua_daemon_health.cua_driver_daemon_status()

    assert status["state"] == "not_installed"
    assert status["hint"] == "Run: hermes computer-use install"
    assert status["missing_targets"] == []


def test_doctor_reports_the_dead_daemon_as_degraded(monkeypatch, tmp_path, capsys):
    """RED before the fix: doctor said "ok" (exit 0) while the unit's ExecStart target was long gone."""
    _fake_home(monkeypatch, tmp_path)
    _serve_unit(monkeypatch, tmp_path, _versioned_exec_start())
    _pin_driver(monkeypatch, "/usr/local/bin/cua-driver")
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", Mock(return_value=False))

    with patch.object(doctor, "_drive_health_report", return_value=_healthy_report()):
        code = doctor.run_doctor(color=False)
    out = capsys.readouterr().out

    assert code == 1  # degraded — not "the binary is healthy", which is what hid this for 46h
    assert _PRUNED_VERSION in out and "does not exist" in out
    assert "daemon_service_path: " in out
    assert "packages/current/cua-driver" in out  # the fix names the stable path

    with patch.object(doctor, "_drive_health_report", return_value=_healthy_report()):
        assert doctor.run_doctor(json_output=True, color=False) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"] == "degraded"
    failed = {c["name"]: c for c in payload["checks"] if c["status"] == "fail"}
    assert set(failed) == {"daemon_service_path", "daemon_reachable"}
    assert "packages/current/cua-driver" in failed["daemon_service_path"]["hint"]


def test_doctor_healthy_host_gains_no_new_failure_item(monkeypatch, tmp_path, capsys):
    """Protection: no serve reference = nothing is probed and nothing is added, so overall and exit stay put."""
    _fake_home(monkeypatch, tmp_path)
    monkeypatch.setattr(cua_daemon_health, "_reference_globs",
                        lambda: [str(tmp_path / ".config" / "systemd" / "user" / "*.service")])
    probe = Mock(return_value=False)
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", probe)
    _pin_driver(monkeypatch, "/usr/local/bin/cua-driver")

    with patch.object(doctor, "_drive_health_report", return_value=_healthy_report()):
        code = doctor.run_doctor(color=False)
    out = capsys.readouterr().out

    assert code == 0
    assert "daemon_reachable" not in out and "daemon_service_path" not in out
    assert "❌" not in out
    probe.assert_not_called()

    with patch.object(doctor, "_drive_health_report", return_value=_healthy_report()):
        assert doctor.run_doctor(json_output=True, color=False) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"] == "ok"
    assert payload["checks"] == _healthy_report()["checks"]  # upstream payload untouched
    probe.assert_not_called()


def test_doctor_leaves_a_running_daemon_reported_ok(monkeypatch, tmp_path, capsys):
    """Protection: healthy daemon → pass rows, exit 0, no new failure."""
    home = _fake_home(monkeypatch, tmp_path)
    target = home / ".cua-driver" / "packages" / "current" / "cua-driver"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _serve_unit(monkeypatch, tmp_path, "~/.cua-driver/packages/current/cua-driver serve")
    _pin_driver(monkeypatch, str(target))
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", Mock(return_value=True))

    with patch.object(doctor, "_drive_health_report", return_value=_healthy_report()):
        code = doctor.run_doctor(color=False)
    out = capsys.readouterr().out

    assert code == 0
    assert "✅ daemon_service_path" in out and "✅ daemon_reachable" in out


def _incompatible_contract() -> dict:
    return {"ready": False, "binary": "/usr/local/bin/cua-driver", "version": "0.19.3",
            "reason": "Hermes computer use requires cua-driver 0.20.0 or newer"}


def test_dead_daemon_skips_the_reinstall_reflex(monkeypatch):
    """Distinction: the daemon case must not reinstall the driver (a no-op that fakes a repair)."""
    monkeypatch.setattr(cua_backend, "_contract_repair_attempted", False)
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    dead = {"state": "not_running", "reason": "unit starts a daemon from a path that does not exist: /gone/cua-driver",
            "hint": "Point it at ~/.cua-driver/packages/current/cua-driver and restart it."}
    with patch.object(cua_backend, "_daemon_problem", return_value=dead), \
         patch("hermes_cli.tools_config.install_cua_driver") as installer:
        contract = cua_backend._maybe_repair_runtime_contract(_incompatible_contract())

    installer.assert_not_called()
    assert "does not exist" in contract["daemon_problem"]
    hint = cua_backend._not_ready_hint(contract)
    assert "packages/current/cua-driver" in hint
    assert "hermes computer-use install" not in hint


def test_no_configured_daemon_keeps_the_install_reflex(monkeypatch):
    """Protection: with no daemon involved the existing repair behaviour is untouched."""
    monkeypatch.setattr(cua_backend, "_contract_repair_attempted", False)
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    with patch.object(cua_backend, "_daemon_problem", return_value=None), \
         patch("hermes_cli.tools_config.install_cua_driver", return_value=False) as installer:
        contract = cua_backend._maybe_repair_runtime_contract(_incompatible_contract())

    installer.assert_called_once_with(upgrade=False, show_installer_progress=False)
    assert contract["reason"].endswith("0.20.0 or newer")
    assert "hermes computer-use install" in cua_backend._not_ready_hint(contract)


def test_pruned_unit_host_is_diagnosed_as_a_daemon_problem(monkeypatch, tmp_path):
    """Wiring: ``_daemon_problem`` (the repair gate's input) sees the pruned-unit host as the daemon case."""
    _fake_home(monkeypatch, tmp_path)
    _serve_unit(monkeypatch, tmp_path, _versioned_exec_start())
    _pin_driver(monkeypatch, "/usr/local/bin/cua-driver")
    monkeypatch.setattr(cua_daemon_health, "daemon_probe", Mock(return_value=False))

    assert cua_backend._daemon_problem()["state"] == "not_running"
