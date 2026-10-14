"""Probe and degrade diagnostics for the ``systemd-run --user --scope`` path.

Split from ``tests/tools/test_process_registry.py`` so both files stay under
their FILE_LINES cap; the diagnostics tests pin the #137222 behaviour (a missing
binary must not be reported as a user-bus problem)."""

import subprocess
import threading

import pytest

import tools.process_registry as pr


@pytest.fixture(autouse=True)
def _reset_systemd_scope_cache():
    """Reset the cached ``systemd-run --user --scope`` availability flag and its
    unavailable reason before each test so a probe run on a real systemd host
    (where ``INVOCATION_ID`` is set) doesn't leak into tests that mock
    ``subprocess``. Tests that exercise the probe directly reset the cache
    themselves."""
    original = pr._SYSTEMD_SCOPE_AVAILABLE
    original_reason = pr._SYSTEMD_SCOPE_UNAVAILABLE_REASON
    pr._SYSTEMD_SCOPE_AVAILABLE = False
    pr._SYSTEMD_SCOPE_UNAVAILABLE_REASON = None
    yield
    pr._SYSTEMD_SCOPE_AVAILABLE = original
    pr._SYSTEMD_SCOPE_UNAVAILABLE_REASON = original_reason


class TestSystemdScopeProbe:
    @pytest.mark.platforms("linux")
    def test_systemd_run_user_scope_available_caches_after_probe(self, monkeypatch):
        """The availability check probes once and caches — a second call must
        not re-probe (and must return the same value)."""
        # Reset the cache.
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        first = pr._systemd_run_user_scope_available()
        second = pr._systemd_run_user_scope_available()
        assert first is True
        assert second is True
        assert len(probe_calls) == 1, "probe must run only once (cached)"
        # The probe must not carry OOMPolicy= either: that is the argv systemd
        # rejected on scope units and cached as "unavailable" (#102486).
        probe_argv = probe_calls[0][0]
        assert not any(
            value.startswith("OOMPolicy=") for value in probe_argv if isinstance(value, str)
        ), probe_argv

    @pytest.mark.platforms("linux")
    @pytest.mark.parametrize("rejects_no_expand", [False, True], ids=["systemd>=254", "systemd<254"])
    def test_scoped_command_is_not_expanded_by_systemd_run(self, monkeypatch, rejects_no_expand):
        """systemd >= 254 rewrites ``$$``/``${X}`` in a --scope command unless given
        ``--expand-environment=no``; older systemd-run rejects that option, and losing
        it there must not cost the scope (#132385)."""
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_RUN_NO_EXPAND", True)
        probes = []

        def fake_run(argv, **kwargs):
            probes.append(argv)
            if rejects_no_expand and "--expand-environment=no" in argv:
                return subprocess.CompletedProcess(
                    argv, 1, stderr=b"systemd-run: unrecognized option '--expand-environment=no'")
            return subprocess.CompletedProcess(argv, 0, stderr=b"")

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        assert pr._systemd_run_user_scope_available() is True
        spawn = pr._build_systemd_scope_argv(["bash", "-c", "echo $$"], unit_suffix="t")
        assert ("--expand-environment=no" in spawn) is not rejects_no_expand
        assert spawn[spawn.index("--") + 1:] == ["bash", "-c", "echo $$"]
        assert len(probes) == (2 if rejects_no_expand else 1)

    @pytest.mark.platforms("linux")
    def test_successful_systemd_probe_revalidates_after_cache_ttl(self, monkeypatch):
        """A vanished user bus invalidates a formerly successful scope verdict."""
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_PROBED_AT", 0.0)
        clock = [100.0]
        probe_results = [0, 1]
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(
                args=args[0], returncode=probe_results.pop(0)
            )

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("tools.process_registry.time.monotonic", lambda: clock[0])
        monkeypatch.setattr("subprocess.run", fake_run)

        assert pr._systemd_run_user_scope_available() is True
        clock[0] += 61
        assert pr._systemd_run_user_scope_available() is False
        assert len(probe_calls) == 2

    @pytest.mark.platforms("linux")
    def test_systemd_scope_first_probe_is_serialized(self, monkeypatch):
        """Concurrent first-use callers must wait for one definitive probe.

        A temporary cached ``False`` would let a racing worker spawn inside the
        gateway cgroup, defeating the OOM isolation guarantee.
        """
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        probe_started = threading.Event()
        release_probe = threading.Event()
        probe_calls = []
        results = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            probe_started.set()
            assert release_probe.wait(timeout=2)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        first = threading.Thread(
            target=lambda: results.append(pr._systemd_run_user_scope_available())
        )
        second = threading.Thread(
            target=lambda: results.append(pr._systemd_run_user_scope_available())
        )
        first.start()
        assert probe_started.wait(timeout=2)
        second.start()

        # The racing caller must be blocked behind the probe, not observe a
        # temporary False cache value.
        second.join(timeout=0.05)
        assert second.is_alive()

        release_probe.set()
        first.join(timeout=2)
        second.join(timeout=2)

        assert not first.is_alive()
        assert not second.is_alive()
        assert results == [True, True]
        assert len(probe_calls) == 1


class TestSystemdScopeDegradeDiagnostics:
    def test_scope_unavailable_reason_distinguishes_missing_binary(self, monkeypatch):
        """A missing systemd-run binary must degrade as a service-PATH problem
        with the PATH named, not as a user-bus problem: the linger advice is the
        wrong remedy there (#137222)."""
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_UNAVAILABLE_REASON", None)
        monkeypatch.setattr(pr, "_IS_LINUX", True)
        monkeypatch.setattr("shutil.which", lambda name: None)

        assert pr._systemd_run_user_scope_available() is False
        detail = pr._systemd_scope_unavailable_detail()
        assert "not found on PATH" in detail
        assert "PATH=" in detail
        assert "extraPackages" in detail
        assert "enable-linger" not in detail

    def test_scope_unavailable_reason_names_probe_failure(self, monkeypatch):
        """A failed probe keeps the user-bus/linger advice and now also carries
        the rc and stderr, so the real cause is diagnosable from the message
        (#137222, #133546)."""
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_UNAVAILABLE_REASON", None)
        monkeypatch.setattr(pr, "_IS_LINUX", True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, 1, stderr=b"Failed to connect to user bus: No medium found\n")

        monkeypatch.setattr("subprocess.run", fake_run)

        assert pr._systemd_run_user_scope_available() is False
        detail = pr._systemd_scope_unavailable_detail()
        assert "rc=1" in detail
        assert "Failed to connect to user bus" in detail
        assert "enable-linger" in detail

    def test_kanban_refusal_names_missing_binary(self, monkeypatch):
        """``require_restart_safe_scope=True`` (kanban's hard-fail path) raises
        RestartSafeScopeUnavailable carrying the real cause when systemd-run is
        missing from the unit PATH (#137222)."""
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_UNAVAILABLE_REASON", None)
        monkeypatch.setattr(pr, "_IS_LINUX", True)
        monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
        monkeypatch.setenv("INVOCATION_ID", "1")
        monkeypatch.setattr("shutil.which", lambda name: None)

        with pytest.raises(pr.RestartSafeScopeUnavailable) as excinfo:
            pr.restart_safe_gateway_child_argv(
                ["hermes", "worker"], unit_suffix="kanban",
                require_restart_safe_scope=True, outlives_parent=True,
            )
        assert "not found on PATH" in str(excinfo.value)
        assert "enable-linger" not in str(excinfo.value)
