"""Regression tests for the ``_bash_starts`` timeout path (PR #83413).

On Windows, the external-program probe could wedge a process forever:
the probe inherited the parent's stdin (a JSON-RPC pipe under ACP/gateway
embedding), and on ``TimeoutExpired`` CPython's ``run()`` killed only the
direct child.  With ``HERMES_GIT_BASH_PATH`` pointing at Git-for-Windows
``bin\\bash.exe`` (a shim that spawns ``usr\\bin\\bash.exe``) the kill
orphaned the real bash, which held the pipe write ends open — and the
follow-up ``communicate()`` blocked indefinitely.

Per AGENTS.md ("Don't fake the host OS"), the per-OS kill arms are tested
through ``_reap_timed_out_probe(proc, is_windows=...)`` — a parameter, not
a patched host constant — and the end-to-end ``_bash_starts`` assertions
are either host-agnostic or gated behind the native platform markers.
"""
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments import local as local_mod

FAKE_BASH = r"C:\fake\git\bin\bash.exe"


@pytest.fixture(autouse=True)
def _clean_probe_caches():
    saved_starts = dict(local_mod._bash_starts_cache)
    saved_details = dict(local_mod._bash_probe_details_cache)
    local_mod._bash_starts_cache.clear()
    local_mod._bash_probe_details_cache.clear()
    yield
    local_mod._bash_starts_cache.clear()
    local_mod._bash_starts_cache.update(saved_starts)
    local_mod._bash_probe_details_cache.clear()
    local_mod._bash_probe_details_cache.update(saved_details)


def _hung_proc(drain=("", "")):
    """A fake Popen for a probe that already hit its 15s timeout."""
    proc = MagicMock()
    proc.pid = 4242
    proc.communicate.side_effect = [drain]
    return proc


# ── the reaper, both arms as input→output ─────────────────────────────────


class TestReapTimedOutProbe:

    def test_windows_arm_tree_kills_by_pid(self, monkeypatch):
        run_calls = []
        monkeypatch.setattr(
            local_mod.subprocess,
            "run",
            lambda args, **kw: run_calls.append(list(args)) or MagicMock(returncode=0),
        )
        proc = _hung_proc()

        local_mod._reap_timed_out_probe(proc, is_windows=True)

        assert any(
            call[:3] == ["taskkill", "/T", "/F"] and str(proc.pid) in call
            for call in run_calls
        ), f"expected taskkill /T /F on pid {proc.pid}, got {run_calls}"
        proc.kill.assert_not_called()

    def test_posix_arm_kills_process_directly(self, monkeypatch):
        run_calls = []
        monkeypatch.setattr(
            local_mod.subprocess,
            "run",
            lambda args, **kw: run_calls.append(list(args)) or MagicMock(returncode=0),
        )
        proc = _hung_proc()

        local_mod._reap_timed_out_probe(proc, is_windows=False)

        proc.kill.assert_called_once()
        assert not run_calls, "taskkill must not be used off Windows"

    def test_taskkill_failure_still_reaches_the_bounded_drain(self, monkeypatch):
        def exploding_run(args, **kw):
            raise subprocess.TimeoutExpired(cmd="taskkill", timeout=10)

        monkeypatch.setattr(local_mod.subprocess, "run", exploding_run)
        proc = _hung_proc()

        # Must not raise: cleanup failures stay inside the reaper so the
        # caller's cache write always happens.
        local_mod._reap_timed_out_probe(proc, is_windows=True)

        proc.communicate.assert_called_once()

    def test_orphaned_fork_child_cannot_wedge_the_drain(self, monkeypatch):
        monkeypatch.setattr(
            local_mod.subprocess, "run", lambda args, **kw: MagicMock(returncode=0)
        )
        proc = _hung_proc(drain=subprocess.TimeoutExpired(cmd="bash", timeout=5))
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="bash", timeout=5)
        ]

        local_mod._reap_timed_out_probe(proc, is_windows=True)  # must return


# ── _bash_starts behavior, host-agnostic ──────────────────────────────────


class TestBashStartsTimeoutPath:

    def _timeout_popen(self, monkeypatch):
        proc = MagicMock()
        proc.pid = 4242
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="bash", timeout=15),
            ("", ""),
        ]
        monkeypatch.setattr(
            local_mod.subprocess, "Popen", MagicMock(return_value=proc)
        )
        monkeypatch.setattr(
            local_mod.subprocess, "run", lambda args, **kw: MagicMock(returncode=0)
        )
        return proc

    def test_timeout_returns_false_and_caches_verdict(self, monkeypatch):
        self._timeout_popen(monkeypatch)

        ok = local_mod._bash_starts(FAKE_BASH)

        assert ok is False
        assert local_mod._bash_starts_cache[FAKE_BASH] is False
        assert "timed out" in local_mod._bash_probe_details_cache[FAKE_BASH]

    def test_healthy_probe_uses_devnull_stdin_and_caches_true(self, monkeypatch):
        proc = MagicMock()
        proc.pid = 4242
        proc.returncode = 0
        proc.communicate.return_value = ("", "")
        popen = MagicMock(return_value=proc)
        monkeypatch.setattr(local_mod.subprocess, "Popen", popen)

        ok = local_mod._bash_starts(FAKE_BASH)

        assert ok is True
        assert local_mod._bash_starts_cache[FAKE_BASH] is True
        _args, kwargs = popen.call_args
        assert kwargs.get("stdin") == subprocess.DEVNULL


# ── native-arm smoke, on the real host only ───────────────────────────────


class TestBashStartsNativeArms:

    @pytest.mark.windows_only
    def test_native_windows_timeout_invokes_taskkill(self, monkeypatch):
        proc = MagicMock()
        proc.pid = 4242
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="bash", timeout=15),
            ("", ""),
        ]
        run_calls = []
        monkeypatch.setattr(
            local_mod.subprocess, "Popen", MagicMock(return_value=proc)
        )
        monkeypatch.setattr(
            local_mod.subprocess,
            "run",
            lambda args, **kw: run_calls.append(list(args)) or MagicMock(returncode=0),
        )

        assert local_mod._bash_starts(FAKE_BASH) is False
        assert any(call[:3] == ["taskkill", "/T", "/F"] for call in run_calls)

    @pytest.mark.linux_only
    def test_native_linux_timeout_kills_directly(self, monkeypatch):
        proc = MagicMock()
        proc.pid = 4242
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="bash", timeout=15),
            ("", ""),
        ]
        run_calls = []
        monkeypatch.setattr(
            local_mod.subprocess, "Popen", MagicMock(return_value=proc)
        )
        monkeypatch.setattr(
            local_mod.subprocess,
            "run",
            lambda args, **kw: run_calls.append(list(args)) or MagicMock(returncode=0),
        )

        assert local_mod._bash_starts("/usr/bin/bash") is False
        proc.kill.assert_called_once()
        assert not run_calls
