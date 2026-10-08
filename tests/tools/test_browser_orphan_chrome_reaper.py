"""Tests for the orphaned-Chromium reaper (tools/browser_tool_lifecycle.py).

Covers the half of the browser leak the daemon reaper cannot reach: Chromium whose
agent-browser daemon already exited, so no socket dir / pid file / ``_active_sessions``
entry points at it any more (upstream #100855, #32047, PR #100998).
"""

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import browser_tool_lifecycle as lifecycle
from tools import browser_tool as bt


@pytest.fixture()
def owner_state(tmp_path, monkeypatch):
    """Isolate the persisted ownership record and the temp-profile roots."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setattr(lifecycle, "get_hermes_home", lambda: home)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(scratch))
    monkeypatch.delenv("TMPDIR", raising=False)
    return home / lifecycle._BROWSER_OWNER_STATE_FILE, scratch


@pytest.fixture()
def profile(owner_state):
    state_path, scratch = owner_state
    prof = scratch / "chrome-test-profile"
    prof.mkdir()
    return prof


def _run(monkeypatch, procs, *, cmdlines=None, port=None, client=False, fingerprint=1234,
         terminations=None):
    """Drive ``_reap_orphaned_chrome_processes`` with a fake process table."""
    monkeypatch.setattr(lifecycle, "_chromium_main_processes", lambda: procs)
    monkeypatch.setattr(lifecycle, "_all_cmdlines", lambda: dict(cmdlines or {}))
    monkeypatch.setattr(lifecycle, "_devtools_port", lambda _p: port)
    monkeypatch.setattr(lifecycle, "_port_has_client", lambda _p: client)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda _pid: fingerprint)
    calls = terminations if terminations is not None else []

    def fake_terminate(pid, expected_start=None):
        calls.append((pid, expected_start))

    monkeypatch.setattr("tools.process_registry.ProcessRegistry._terminate_host_pid",
                        staticmethod(fake_terminate))
    return lifecycle._reap_orphaned_chrome_processes(), calls


def _seed(state_path, profile, first_seen):
    state_path.write_text(json.dumps({
        str(profile): {"first_seen": first_seen, "last_seen": first_seen,
                       "main_pid": 4242, "start_time": 99, "exe": "/usr/bin/chromium-browser"},
    }), encoding="utf-8")


class TestOwnershipRecord:
    def test_first_sighting_is_recorded_not_reaped(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)])

        assert reaped == 0 and calls == []          # never reap on the pass that discovers it
        record = json.loads(state_path.read_text(encoding="utf-8"))
        assert str(profile) in record
        assert record[str(profile)]["main_pid"] == 4242
        assert record[str(profile)]["first_seen"] > 0

    def test_vanished_profiles_are_forgotten(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        reaped, _calls = _run(monkeypatch, [])      # nothing running any more

        assert reaped == 0
        assert json.loads(state_path.read_text(encoding="utf-8")) == {}

    def test_scan_failure_leaves_the_record_untouched(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        monkeypatch.setattr(lifecycle, "_chromium_main_processes", lambda: None)

        assert lifecycle._reap_orphaned_chrome_processes() == 0
        assert str(profile) in json.loads(state_path.read_text(encoding="utf-8"))


class TestReaping:
    def test_watched_orphan_is_tree_killed_with_identity_check(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)               # watched since the epoch → past the grace
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)],
                             fingerprint=777)

        assert reaped == 1
        assert calls == [(4242, 777)]               # fingerprint re-validated at kill time
        assert not profile.exists()                 # temp profile goes with the tree
        assert str(profile) not in json.loads(state_path.read_text(encoding="utf-8"))

    def test_grace_period_protects_a_fresh_orphan(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        import time as _time
        _seed(state_path, profile, _time.time() - 60)
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)])

        assert reaped == 0 and calls == []
        assert profile.exists()

    def test_live_launcher_is_spared(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 777)])

        assert reaped == 0 and calls == []          # ppid != 1 → daemon/launcher alive

    def test_live_owner_reference_is_spared(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        cmdlines = {999: f"node /usr/lib/agent-browser/daemon.js --user-data-dir={profile}"}
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)],
                             cmdlines=cmdlines)

        assert reaped == 0 and calls == []
        assert profile.exists()

    def test_active_cdp_client_is_spared(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)],
                             port="9243", client=True)

        assert reaped == 0 and calls == []
        assert profile.exists()

    def test_missing_fingerprint_refuses_the_kill(self, owner_state, profile, monkeypatch):
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)],
                             fingerprint=None)

        assert reaped == 0 and calls == []
        assert profile.exists()                     # a leaked orphan beats killing a stranger

    def test_the_tree_processes_are_not_mistaken_for_owners(self, owner_state, profile, monkeypatch):
        """Chromium's own children carry the same --user-data-dir; they must not shield it."""
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        cmdlines = {
            4242: f"/usr/bin/chromium-browser --user-data-dir={profile}",
            4243: f"/usr/lib64/chromium-browser/chromium-browser --type=renderer --user-data-dir={profile}",
        }
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)],
                             cmdlines=cmdlines)

        assert reaped == 1 and calls == [(4242, 1234)]

    def test_a_non_chromium_process_referencing_the_profile_still_shields_it(
            self, owner_state, profile, monkeypatch):
        """A live daemon (node/python) that names the profile is a real owner."""
        state_path, _scratch = owner_state
        _seed(state_path, profile, 0)
        cmdlines = {5000: f"node /usr/local/lib/agent-browser/cli.js --user-data-dir={profile}"}
        reaped, calls = _run(monkeypatch, [(4242, str(profile), "/usr/bin/chromium-browser", 99, 1)],
                             cmdlines=cmdlines)

        assert reaped == 0 and calls == []


class TestCandidateDiscovery:
    def test_only_temp_profiles_are_candidates(self, owner_state, profile, monkeypatch):
        assert lifecycle._profile_is_temporary(str(profile)) is True
        assert lifecycle._profile_is_temporary("/tmp/chrome-x") is True
        home = Path.home() / ".config" / "google-chrome"
        assert lifecycle._profile_is_temporary(str(home)) is False

    def test_filters_by_executable_type_flag_and_profile_root(self, owner_state, profile, monkeypatch):
        state_path, scratch = owner_state
        temp_profile = str(profile)
        user_profile = str(Path.home() / ".config" / "google-chrome")

        class _Proc:
            def __init__(self, info):
                self.info = info

        fake = types.SimpleNamespace(
            process_iter=lambda _attrs: [
                _Proc({"pid": 1, "name": "chromium-browser", "cmdline": [f"--user-data-dir={temp_profile}"],
                       "exe": "/usr/bin/chromium-browser", "ppid": 1, "create_time": 10.0}),
                _Proc({"pid": 2, "name": "chromium-browser", "cmdline": ["--type=renderer",
                       f"--user-data-dir={temp_profile}"], "exe": "/usr/bin/chromium-browser",
                       "ppid": 1, "create_time": 10.0}),
                _Proc({"pid": 3, "name": "chromium-browser", "cmdline": [f"--user-data-dir={user_profile}"],
                       "exe": "/usr/bin/chromium-browser", "ppid": 1, "create_time": 10.0}),
                _Proc({"pid": 4, "name": "python3", "cmdline": [f"--user-data-dir={temp_profile}"],
                       "exe": "/usr/bin/python3", "ppid": 1, "create_time": 10.0}),
            ])
        monkeypatch.setitem(sys.modules, "psutil", fake)

        found = lifecycle._chromium_main_processes()

        assert found == [(1, temp_profile, "/usr/bin/chromium-browser", 10, 1)]


class TestAllCmdlines:
    def test_maps_every_live_process_cmdline(self, monkeypatch):
        """psutil-backed ``_all_cmdlines`` must flatten argv like the old /proc reader."""
        import sys
        import types

        class _Proc:
            def __init__(self, pid, cmdline):
                self.info = {"pid": pid, "cmdline": cmdline}

        fake = types.SimpleNamespace(process_iter=lambda _attrs: [
            _Proc(100, ["python", "--user-data-dir=/tmp/x"]),
            _Proc(101, None),                       # vanished between scan and read
            _Proc(102, []),                         # kernel thread: empty cmdline
        ])
        monkeypatch.setitem(sys.modules, "psutil", fake)

        assert lifecycle._all_cmdlines() == {
            100: "python --user-data-dir=/tmp/x",
            101: "",
            102: "",
        }

    def test_psutil_unavailable_yields_empty_map(self, monkeypatch):
        """No psutil -> no cmdline reference data; the reap path gates on psutil anyway."""
        import sys
        monkeypatch.setitem(sys.modules, "psutil", None)

        assert lifecycle._all_cmdlines() == {}


class TestWiring:
    def test_daemon_sweep_also_sweeps_chromium(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(lifecycle, "_reap_orphaned_chrome_processes", lambda: calls.append("chrome"))
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(lifecycle, "_best_effort",
                            lambda label, fn: fn() if "Chromium" in label else None)

        lifecycle._reap_orphaned_browser_sessions()

        assert calls == ["chrome"]

    def test_atexit_registers_the_sweep_before_the_emergency_teardown(self):
        """LIFO: registering first makes it run last, after the emergency teardown.

        Behavioral check, not source-reading: in a fresh interpreter, interpose on
        ``atexit.register`` while importing ``browser_tool`` and assert the sweep is
        wrapped in ``_best_effort`` and registered *before* the emergency teardown.
        """
        import subprocess
        import sys
        import textwrap
        script = textwrap.dedent("""\
            import atexit

            seen = []
            real_register = atexit.register

            def wrapped_register(fn, *args, **kwargs):
                seen.append((getattr(fn, "__name__", repr(fn)), [
                    getattr(a, "__name__", None) for a in args
                ], kwargs))
                return real_register(fn, *args, **kwargs)

            atexit.register = wrapped_register
            try:
                import tools.browser_tool  # noqa: F401 -- import performs the registrations
            finally:
                atexit.register = real_register

            sweep_pos = emergency_pos = None
            for pos, (name, arg_names, _kwargs) in enumerate(seen):
                if name == "_best_effort" and "_reap_orphaned_chrome_processes" in arg_names:
                    sweep_pos = pos
                if name == "_emergency_cleanup_all_sessions":
                    emergency_pos = pos

            if sweep_pos is None or emergency_pos is None or not (sweep_pos < emergency_pos):
                print(f"BAD sweep={sweep_pos} emergency={emergency_pos} seen={seen}")
                raise SystemExit(1)
            print("OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert hasattr(lifecycle, "_reap_orphaned_chrome_processes")
