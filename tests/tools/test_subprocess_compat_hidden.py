"""Tests for the run_hidden / popen_hidden console-hiding helpers."""

from __future__ import annotations

import subprocess

from hermes_cli import _subprocess_compat as sc


def test_run_hidden_injects_hide_flags_and_delegates(monkeypatch):
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "RESULT"

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = sc.run_hidden(["git", "status"], capture_output=True, timeout=5)

    assert out == "RESULT"
    assert captured["args"] == (["git", "status"],)
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["timeout"] == 5
    assert captured["kwargs"]["creationflags"] == sc.windows_hide_flags()


def test_popen_hidden_injects_hide_flags_and_delegates(monkeypatch):
    captured: dict = {}

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "PROC"

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = sc.popen_hidden(["gh", "auth", "token"])

    assert out == "PROC"
    assert captured["args"] == (["gh", "auth", "token"],)
    assert captured["kwargs"]["creationflags"] == sc.windows_hide_flags()


def test_run_hidden_ors_caller_supplied_creationflags(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: captured.update(k))

    sentinel = 0x1  # CREATE_NEW_CONSOLE-ish placeholder
    sc.run_hidden(["git"], creationflags=sentinel)

    assert captured["creationflags"] == (sentinel | sc.windows_hide_flags())


def test_helpers_are_exported():
    assert "run_hidden" in sc.__all__
    assert "popen_hidden" in sc.__all__
