"""Linux daemon-unit references to pruned cua-driver release dirs (#114748).

Linux has no managed cua-driver autostart, so users hand-write systemd user
units / XDG autostart entries against a concrete release directory. The
installer prunes all but the last five, so a versioned Exec reference
crash-loops with 203/EXEC after every upgrade — for days, while every
binary-level check (and `computer-use install`'s binary repair) stays green.
The doctor guard surfaces the dead reference with the `packages/current`
recovery hint.
"""

import sys

import pytest

from tools.computer_use import doctor


def _write_unit(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_pruned_release_reference_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_unit(
        tmp_path / "systemd" / "user" / "cua-driver-screenshot.service",
        "[Service]\n"
        "ExecStart=%h/.cua-driver/packages/releases/0.20.0-x86_64-unknown-linux-gnu/cua-driver "
        "serve --socket %h/.cache/cua-driver/cua-driver.sock\n",
    )
    findings = doctor._stale_cua_exec_references(str(tmp_path))
    assert findings == [
        (
            "systemd user unit",
            "cua-driver-screenshot.service",
            "%h/.cua-driver/packages/releases/0.20.0-x86_64-unknown-linux-gnu/cua-driver",
        )
    ]


def test_current_reference_and_live_release_are_silent(tmp_path):
    # `packages/current` never carries a version, so it can never rot.
    _write_unit(
        tmp_path / "systemd" / "user" / "cua.service",
        "[Service]\nExecStart=~/.cua-driver/packages/current/cua-driver serve\n",
    )
    # A releases/ dir that still exists on disk is healthy.
    live = tmp_path / "live" / "packages" / "releases" / "0.28.2" / "cua-driver"
    _write_unit(live, "")
    _write_unit(tmp_path / "autostart" / "cua-driver.desktop", f"[Desktop Entry]\nExec={live}\n")
    assert doctor._stale_cua_exec_references(str(tmp_path)) == []


def test_desktop_entry_dead_release_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_unit(
        tmp_path / "autostart" / "cua-driver.desktop",
        "[Desktop Entry]\nExec=~/.cua-driver/packages/releases/0.19.0/cua-driver serve\n",
    )
    assert [f[1] for f in doctor._stale_cua_exec_references(str(tmp_path))] == ["cua-driver.desktop"]


def test_systemd_prefix_modifier_does_not_hide_dead_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # '-' means "restart failures are tolerated" — the target is still dead.
    _write_unit(
        tmp_path / "systemd" / "user" / "restart-anyway.service",
        "[Service]\nExecStart=-%h/.cua-driver/packages/releases/0.20.0/cua-driver serve\n",
    )
    assert len(doctor._stale_cua_exec_references(str(tmp_path))) == 1


def test_missing_config_dirs_are_silent(tmp_path):
    assert doctor._stale_cua_exec_references(str(tmp_path / "nonexistent")) == []


def test_guard_appends_fail_check_and_degrades_ok(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_stale_cua_exec_references",
        lambda: [
            (
                "systemd user unit",
                "cua-driver-screenshot.service",
                "%h/.cua-driver/packages/releases/0.20.0-x86_64-unknown-linux-gnu/cua-driver",
            )
        ],
    )
    monkeypatch.setattr(sys, "platform", "linux")
    out = doctor._apply_stale_unit_guard(
        {"overall": "ok", "checks": [{"name": "binary_version", "status": "pass"}]}
    )
    assert out["overall"] == "degraded"
    appended = out["checks"][-1]
    assert appended["status"] == "fail"
    assert "cua-driver-screenshot.service" in appended["name"]
    assert "pruned" in appended["message"]
    assert "packages/current" in appended["hint"]


def test_guard_never_softens_worse_overall(monkeypatch):
    monkeypatch.setattr(
        doctor, "_stale_cua_exec_references", lambda: [("XDG autostart entry", "cua.desktop", "/x/packages/releases/0.1.0/cua-driver")]
    )
    monkeypatch.setattr(sys, "platform", "linux")
    out = doctor._apply_stale_unit_guard({"overall": "failed", "checks": []})
    assert out["overall"] == "failed"
    assert out["checks"][-1]["status"] == "fail"


def test_guard_without_checks_list_appends_nothing(monkeypatch):
    monkeypatch.setattr(
        doctor, "_stale_cua_exec_references", lambda: [("systemd user unit", "cua.service", "/x/packages/releases/0.1.0/cua-driver")]
    )
    monkeypatch.setattr(sys, "platform", "linux")
    # No checks list to append to → no unexplained degraded either.
    out = doctor._apply_stale_unit_guard({"overall": "ok"})
    assert out == {"overall": "ok"}


def test_guard_off_linux_never_scans(monkeypatch):
    monkeypatch.setattr(doctor, "_stale_cua_exec_references", lambda: pytest.fail("must not scan off Linux"))
    monkeypatch.setattr(sys, "platform", "darwin")
    report = {"overall": "ok", "checks": []}
    assert doctor._apply_stale_unit_guard(report) is report
