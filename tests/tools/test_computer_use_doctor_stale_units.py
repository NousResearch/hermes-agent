"""`hermes computer-use doctor` names a daemon unit whose cua-driver Exec target was pruned (#114748).

Linux has no managed cua-driver autostart, so users hand-write systemd user units / XDG
autostart entries against a concrete ``packages/releases/<version>/`` directory. The installer
keeps only the last five release dirs, so after an upgrade the unit crash-loops with 203/EXEC
while every binary-level check stays green.  Doctor is the only surface that can name it.
"""

import json
import os
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use import doctor

pytestmark = pytest.mark.linux_only

_STALE = "%h/.cua-driver/packages/releases/0.20.0-x86_64-unknown-linux-gnu/cua-driver"


def _fake_health_report_proc() -> MagicMock:
    """Popen double for the MCP handshake + one ``health_report`` call returning an all-ok report."""
    report = {"schema_version": "1", "platform": "linux", "driver_version": "0.28.2", "overall": "ok",
              "checks": [{"name": "binary_version", "status": "pass", "message": "cua-driver 0.28.2"}]}
    lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n",
             json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": report}}) + "\n", ""]
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = MagicMock(side_effect=lines)
    proc.stderr = MagicMock()
    proc.stderr.read = MagicMock(return_value="")
    proc.wait = MagicMock(return_value=0)
    proc.kill = MagicMock()
    return proc


def _run_doctor_json(monkeypatch, home):
    monkeypatch.setenv("HOME", str(home))
    if "XDG_CONFIG_HOME" in os.environ and not os.environ["XDG_CONFIG_HOME"].startswith(str(home)):
        monkeypatch.delenv("XDG_CONFIG_HOME")  # the host's own config dir must not leak into the scan
    monkeypatch.setattr(doctor, "_read_cli_version", lambda binary, timeout=5.0: "cua-driver 0.28.2")
    out = StringIO()
    with patch("shutil.which", return_value="/fake/cua-driver"), \
         patch("subprocess.Popen", return_value=_fake_health_report_proc()), \
         patch("sys.stdout", out):
        code = doctor.run_doctor(json_output=True)
    return code, json.loads(out.getvalue())


def test_doctor_reports_pruned_unit_and_degrades(tmp_path, monkeypatch):
    unit_dir = tmp_path / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "cua-driver-screenshot.service").write_text(
        f"[Service]\nExecStart=-{_STALE} serve --socket %h/.cache/cua-driver/cua-driver.sock\n", encoding="utf-8")

    code, report = _run_doctor_json(monkeypatch, tmp_path)

    unit_checks = [c for c in report["checks"] if c["name"] == "daemon unit (cua-driver-screenshot.service)"]
    assert code == 1 and report["overall"] == "degraded"
    assert unit_checks[0]["status"] == "fail" and _STALE in unit_checks[0]["message"]
    assert "packages/current/cua-driver" in unit_checks[0]["hint"]


def test_doctor_scans_units_under_xdg_config_home(tmp_path, monkeypatch):
    """Hosts that relocate ``~/.config`` via XDG_CONFIG_HOME keep their systemd user units there;
    scanning only ``~/.config`` would give them the silent green doctor this fix exists to remove."""
    cfg = tmp_path / "data" / "cfg"
    unit_dir = cfg / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "cua-driver.service").write_text(f"[Service]\nExecStart={_STALE} serve\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))

    code, report = _run_doctor_json(monkeypatch, tmp_path)

    assert code == 1 and report["overall"] == "degraded"
    assert [c for c in report["checks"] if c["name"] == "daemon unit (cua-driver.service)"]


def test_doctor_is_silent_for_current_and_live_release_references(tmp_path, monkeypatch):
    live = tmp_path / ".cua-driver" / "packages" / "releases" / "0.28.2-x86_64-unknown-linux-gnu" / "cua-driver"
    live.parent.mkdir(parents=True)
    live.write_text("", encoding="utf-8")
    unit_dir = tmp_path / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "a.service").write_text("[Service]\nExecStart=%h/.cua-driver/packages/current/cua-driver serve\n",
                                        encoding="utf-8")
    (unit_dir / "b.service").write_text(f"[Service]\nExecStart={live} serve\n", encoding="utf-8")
    autostart = tmp_path / ".config" / "autostart"
    autostart.mkdir()
    (autostart / "cua.desktop").write_text("[Desktop Entry]\nExec=%h/.cua-driver/packages/current/cua-driver serve\n",
                                           encoding="utf-8")

    code, report = _run_doctor_json(monkeypatch, tmp_path)

    assert code == 0 and report["overall"] == "ok"
    assert not [c for c in report["checks"] if c["name"].startswith("daemon unit")]
