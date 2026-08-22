"""Tests for the Electron cdp_attach branch of _enrich_escalation."""

from __future__ import annotations

import pytest

from tools.computer_use.backend import ActionResult
from tools.computer_use import tool as cu_tool


def _res(escalation=None, meta=None, **kw):
    return ActionResult(
        ok=kw.pop("ok", False),
        action="click",
        message="",
        meta=meta or {},
        escalation=escalation,
        **kw,
    )


FOREGROUND = {"recommended": "foreground", "reason": "occluded renderer"}


class TestElectronEscalationBranch:
    def test_electron_pid_gets_cdp_attach_alternative(self, monkeypatch):
        monkeypatch.setattr(cu_tool, "_is_electron_pid", lambda pid: pid == 4242)
        out = cu_tool._enrich_escalation(_res(dict(FOREGROUND), meta={"pid": 4242}))
        assert out["recommended"] == "foreground"  # driver verdict untouched
        assert out["alternative"] == "cdp_attach"
        assert "hermes browser attach" in out["alternative_hint"]

    def test_non_electron_pid_unchanged(self, monkeypatch):
        monkeypatch.setattr(cu_tool, "_is_electron_pid", lambda pid: False)
        esc = dict(FOREGROUND)
        out = cu_tool._enrich_escalation(_res(esc, meta={"pid": 77}))
        assert out == FOREGROUND
        assert "alternative" not in out

    def test_no_pid_in_meta_skips_silently(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            cu_tool, "_is_electron_pid", lambda pid: called.append(pid) or False
        )
        out = cu_tool._enrich_escalation(_res(dict(FOREGROUND), meta={}))
        assert "alternative" not in out
        assert called == [None]

    def test_non_foreground_recommendation_untouched(self, monkeypatch):
        monkeypatch.setattr(cu_tool, "_is_electron_pid", lambda pid: True)
        esc = {"recommended": "px", "reason": "x"}
        assert cu_tool._enrich_escalation(_res(dict(esc), meta={"pid": 1})) == esc

    def test_typed_page_branch_wins_for_browser_windows(self, monkeypatch):
        # A real browser window (chrome_widgetwin_1) with page input keeps the
        # existing typed-page alternative even if the pid also looks Electron
        # (Electron embeds Chromium and shares the window class on Windows).
        monkeypatch.setattr(cu_tool, "_is_electron_pid", lambda pid: True)
        out = cu_tool._enrich_escalation(
            _res(
                dict(FOREGROUND),
                meta={
                    "pid": 1,
                    "target_class": "Chrome_WidgetWin_1",
                    "event_kind": "text_input",
                },
            )
        )
        assert out["alternative"] == "page"

    def test_none_escalation_passthrough(self):
        assert cu_tool._enrich_escalation(_res(None)) is None


class TestIsElectronPid:
    def test_rejects_non_ints(self):
        assert cu_tool._is_electron_pid(None) is False
        assert cu_tool._is_electron_pid("42") is False
        assert cu_tool._is_electron_pid(True) is False
        assert cu_tool._is_electron_pid(-1) is False

    def test_dead_pid_is_false(self):
        # A pid that (almost certainly) doesn't exist → psutil raises → False.
        assert cu_tool._is_electron_pid(2**22 + 12345) is False

    def test_electron_layout_detected(self, tmp_path, monkeypatch):
        exe = tmp_path / "app" / "myapp"
        res_dir = tmp_path / "app" / "resources"
        res_dir.mkdir(parents=True)
        (res_dir / "app.asar").write_bytes(b"")
        exe.write_bytes(b"")

        class FakeProc:
            def __init__(self, pid):
                pass

            def exe(self):
                return str(exe)

        import psutil

        monkeypatch.setattr(psutil, "Process", FakeProc)
        assert cu_tool._is_electron_pid(1234) is True
