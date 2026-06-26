"""Spawn-hygiene invariants for the gateway git probe (tui_gateway.git_probe).

These guard the Windows-specific failure modes of ``run_git``: the dashboard
runs windowless (``pythonw``) and probes git once per project directory, so a
bare spawn would flash a console window per repo, and locale-codepage decoding
would crash the reader thread on non-ASCII repo paths. We assert the call wiring
(creationflags + utf-8) rather than spawning real ``git`` so the test is
deterministic on every platform.
"""

from __future__ import annotations

import subprocess
from unittest import mock

from hermes_cli._subprocess_compat import windows_hide_flags
from tui_gateway import git_probe


def _run_with_captured_kwargs(returncode=0, stdout="out\n"):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run):
        result = git_probe.run_git("/some/repo", "rev-parse", "--show-toplevel")
    return result, captured


def test_run_git_passes_no_window_and_utf8():
    result, captured = _run_with_captured_kwargs()
    assert result == "out"
    kwargs = captured["kwargs"]
    # No console window flash on Windows; no-op (0) elsewhere.
    assert kwargs.get("creationflags") == windows_hide_flags()
    # Explicit codec so non-ASCII repo paths don't crash the reader thread on
    # locale-ANSI (e.g. cp950) Windows.
    assert kwargs.get("encoding") == "utf-8"
    assert kwargs.get("errors") == "replace"


def test_run_git_returns_empty_on_nonzero_exit():
    result, _ = _run_with_captured_kwargs(returncode=128, stdout="")
    assert result == ""


def test_run_git_empty_cwd_does_not_spawn():
    with mock.patch("subprocess.run") as run:
        assert git_probe.run_git("", "status") == ""
    run.assert_not_called()
