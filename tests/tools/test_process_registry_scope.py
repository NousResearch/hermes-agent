"""Shared systemd scope launch, manager targeting and quiescence contracts."""

from tools import process_registry_scope as _process_scope

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


@pytest.fixture(autouse=True)
def _reset_systemd_scope_cache():
    """Reset the cached ``systemd-run --user --scope`` availability flag
    before each test so a probe run on a real systemd host (where
    ``INVOCATION_ID`` is set) doesn't leak into tests that mock
    ``subprocess.Popen``. Tests that exercise the probe directly reset the
    cache themselves."""
    import tools.process_registry as _pr

    original = _process_scope._SYSTEMD_SCOPE_AVAILABLE
    _process_scope._SYSTEMD_SCOPE_AVAILABLE = False
    yield
    _process_scope._SYSTEMD_SCOPE_AVAILABLE = original


def _make_session(
    sid="proc_test123",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    started_at=None,
) -> ProcessSession:
    """Helper to create a ProcessSession for testing."""
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=started_at or time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
    )
    return s


def _spawn_python_sleep(seconds: float) -> subprocess.Popen:
    """Spawn a portable short-lived Python sleep process."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: systemd scopes")
class TestSystemdCgroupIsolation:
    """Verify spawn_local wraps the worker in ``systemd-run --user --scope``
    when running under a supervisor and systemd-run is available, and falls
    back to the legacy ``start_new_session`` path otherwise.

    Issue #70716: local background terminal executors inherit the gateway's
    cgroup, so an OOM in a memory-heavy worker lets systemd-oomd kill the
    ENTIRE gateway cgroup, taking down the messaging control plane.
    """

    @pytest.fixture()
    def _gateway_identity(self, monkeypatch):
        """Opt-in: mark this test as running AS the live gateway process."""
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid(),
        )

    def _fake_popen_capture(self):
        """Return (fake_popen, captured) where captured["argv"] gets the
        argv passed to subprocess.Popen."""
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["start_new_session"] = kwargs.get("start_new_session")
            proc = MagicMock()
            proc.pid = 4321
            proc.stdout = iter([])
            proc.stdin = MagicMock()
            proc.poll.return_value = None
            return proc

        return fake_popen, captured

    @pytest.mark.linux_only
    def test_wraps_in_systemd_scope_when_supervisor_and_available(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Under a supervisor with systemd-run available, the spawn argv is
        wrapped in ``systemd-run --user --scope --unit=hermes-worker-<id>``."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        # _build_systemd_scope_argv calls shutil.which — point it at a stub.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            session = registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv[0] == "/usr/bin/systemd-run", argv
        assert "--user" in argv
        assert "--scope" in argv
        assert "--quiet" in argv, (
            "systemd-run argv must include --quiet (#70716 gap #3)"
        )
        assert "--unit" in argv
        unit_idx = argv.index("--unit")
        assert argv[unit_idx + 1].startswith("hermes-worker-"), argv
        assert argv[unit_idx + 1] == f"hermes-worker-{session.id}", (
            argv
        )  # _build_systemd_scope_argv uses bare name
        properties = [
            argv[index + 1]
            for index, value in enumerate(argv[:-1])
            if value == "--property"
        ]
        assert "MemoryAccounting=yes" in properties
        # systemd rejects OOMPolicy= on transient --scope units across the versions
        # users run (239/245/249, #102486); emitting it fails the probe and every
        # cron worker dispatch. MemoryMax + MemoryAccounting carry the isolation.
        assert not any(p.startswith("OOMPolicy=") for p in properties), properties
        memory_max = next(
            value for value in properties if value.startswith("MemoryMax=")
        )
        assert int(memory_max.split("=", 1)[1]) > 0
        # The original shell command must still be present at the tail,
        # after the ``--`` separator that prevents systemd-run from
        # interpreting command flags as its own.
        assert "--" in argv, "systemd-run argv must use -- to separate command"
        sep_idx = argv.index("--")
        assert "/bin/bash" in argv[sep_idx:]
        assert "set +m; echo hello" in argv[sep_idx:]
        # systemd-run --scope gives the worker a new cgroup but NOT a new
        # session (#70716 regression: start_new_session was False, so the
        # worker kept the parent's session + controlling terminal → SIGTTIN/
        # SIGTTOU stopped the TUI).  start_new_session=True gives systemd-run
        # (and the scoped worker below it) a private session.
        assert captured["start_new_session"] is True
        # The session must record the unit name so kill_process can stop it.
        assert session.systemd_unit == f"hermes-worker-{session.id}.scope"

    def test_falls_back_when_systemd_run_unavailable(self, registry, monkeypatch, _gateway_identity):
        """Under a supervisor but without systemd-run, fall back to the
        legacy ``start_new_session=True`` path (worker shares the gateway
        cgroup)."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: False,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        # No systemd-run wrapping — direct shell invocation.
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True

    def test_falls_back_when_not_under_supervisor(self, registry, monkeypatch):
        """CLI mode (no supervisor) must NOT wrap in a systemd scope even if
        systemd-run is available — isolation is a gateway concern."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: False,
        )

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True

    @pytest.mark.parametrize("use_pty", [False, True])
    def test_inherited_systemd_marker_does_not_scope_interactive_cli(
        self, registry, monkeypatch, use_pty
    ):
        """A CLI inside a supervised terminal must keep workers off its tty.

        INVOCATION_ID is inherited by every descendant, so its presence
        alone must not activate the gateway-only systemd scope path.
        """
        monkeypatch.setenv("INVOCATION_ID", "herdr-service-inherited-marker")
        monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        if use_pty:
            from ptyprocess import PtyProcess

            fake_pty = MagicMock(pid=4321)
            with (
                patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn,
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)
            assert pty_spawn.call_args.args[0] == [
                "/bin/bash", "-lic", "set +m; codex",
            ]
        else:
            fake_popen, captured = self._fake_popen_capture()
            with (
                patch("subprocess.Popen", side_effect=fake_popen),
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("echo hello", cwd="/tmp")
            assert captured["argv"] == [
                "/bin/bash", "-lic", "set +m; echo hello",
            ]
            assert captured["start_new_session"] is True

        assert session.systemd_unit == ""

    @pytest.mark.parametrize("use_pty", [False, True])
    def test_inherited_gateway_tree_markers_do_not_scope_child_cli(
        self, registry, monkeypatch, use_pty
    ):
        """Gateway descendants are not the gateway process that owns the PID file.

        _HERMES_GATEWAY is inherited (and set by importing gateway.run), so
        both it and INVOCATION_ID may be present in a child process. The
        PID-ownership gate must still keep the scope path off.
        """
        monkeypatch.setenv("INVOCATION_ID", "inherited-systemd-marker")
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid() + 1,
        )
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        if use_pty:
            from ptyprocess import PtyProcess

            fake_pty = MagicMock(pid=4321)
            with (
                patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn,
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)
            assert pty_spawn.call_args.args[0] == [
                "/bin/bash", "-lic", "set +m; codex",
            ]
        else:
            fake_popen, captured = self._fake_popen_capture()
            with (
                patch("subprocess.Popen", side_effect=fake_popen),
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("echo hello", cwd="/tmp")
            assert captured["argv"] == [
                "/bin/bash", "-lic", "set +m; echo hello",
            ]
            assert captured["start_new_session"] is True

        assert session.systemd_unit == ""

    @pytest.mark.linux_only
    def test_systemd_post_spawn_failure_never_kills_gateway_process_group(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Cleanup must not killpg: scope teardown is the authoritative path."""
        fake_popen, _captured = self._fake_popen_capture()
        fake_proc = fake_popen(["placeholder"])

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        broken_reader = MagicMock()
        broken_reader.start.side_effect = RuntimeError("reader failed")

        with patch("subprocess.Popen", return_value=fake_proc), \
            patch("threading.Thread", return_value=broken_reader), \
            patch('tools.process_registry_scope._stop_systemd_unit', return_value=True) as stop_unit, \
            patch("os.killpg") as killpg, \
            patch.object(registry, "_write_checkpoint"):
            with pytest.raises(RuntimeError, match="reader failed"):
                registry.spawn_local("echo hello", cwd="/tmp")

        stop_unit.assert_called_once()
        assert stop_unit.call_args.args[0].startswith("hermes-worker-proc_")
        assert stop_unit.call_args.args[0].endswith(".scope")
        killpg.assert_not_called()

    @pytest.mark.linux_only
    def test_pty_spawn_is_wrapped_in_systemd_scope(self, registry, monkeypatch, _gateway_identity):
        """Interactive executors receive the same sibling-cgroup isolation."""
        from ptyprocess import PtyProcess

        fake_pty = MagicMock()
        fake_pty.pid = 4321

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn, \
            patch("threading.Thread", return_value=MagicMock()), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        argv = pty_spawn.call_args.args[0]
        assert argv[0] == "/usr/bin/systemd-run"
        assert "--scope" in argv
        assert "--unit" in argv
        assert "--" in argv
        assert argv[-3:] == ["/bin/bash", "-lic", "set +m; codex"]
        assert session.systemd_unit == f"hermes-worker-{session.id}.scope"

    @pytest.mark.linux_only
    def test_pty_spawn_failure_reaps_scope_before_distinct_pipe_fallback(
        self, registry, monkeypatch, _gateway_identity
    ):
        """A failed PTY scope must not collide with the pipe fallback scope."""
        from ptyprocess import PtyProcess

        events = []
        fake_proc = MagicMock()
        fake_proc.pid = 4321
        fake_proc.stdout = iter([])
        fake_proc.stdin = MagicMock()
        fake_proc.poll.return_value = None

        def fake_popen(argv, **_kwargs):
            events.append(("pipe", list(argv)))
            return fake_proc

        def fake_stop(unit_name):
            events.append(("stop", unit_name))
            return True

        def fail_pty(*_args, **_kwargs):
            events.append(("pty", None))
            raise RuntimeError("PTY wrapper failed after scope creation")

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(PtyProcess, "spawn", side_effect=fail_pty), \
            patch("subprocess.Popen", side_effect=fake_popen), \
            patch('tools.process_registry_scope._stop_systemd_unit', side_effect=fake_stop), \
            patch("threading.Thread", return_value=MagicMock()), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        assert [event[0] for event in events] == ["pty", "stop", "pipe"]
        stopped_unit = events[1][1]
        fallback_argv = events[2][1]
        assert stopped_unit == f"hermes-worker-{session.id}.scope"
        unit_idx = fallback_argv.index("--unit")
        assert fallback_argv[unit_idx + 1] == (
            f"hermes-worker-{session.id}-pipe-fallback"
        )
        assert session.systemd_unit == (
            f"hermes-worker-{session.id}-pipe-fallback.scope"
        )

    @pytest.mark.linux_only
    def test_pty_spawn_failure_does_not_fallback_when_scope_reap_fails(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Do not launch a duplicate command while the failed PTY scope may live."""
        from ptyprocess import PtyProcess

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            'tools.process_registry_scope._systemd_run_user_scope_available',
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(
            PtyProcess,
            "spawn",
            side_effect=RuntimeError("PTY wrapper failed after scope creation"),
        ), patch("subprocess.Popen") as pipe_spawn, patch(
            'tools.process_registry_scope._stop_systemd_unit', return_value=False
        ) as stop_unit:
            with pytest.raises(RuntimeError, match="could not be reaped"):
                registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        stop_unit.assert_called_once()
        pipe_spawn.assert_not_called()

    def test_worker_memory_limit_honors_local_guard_mb_override(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "123")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch("tools.process_registry.logger.warning") as warning:
            argv = _process_scope._build_systemd_scope_argv(
                ["/bin/bash", "-lc", "true"],
                unit_suffix="test",
                # Memory bounds are caller-resolved since the builder
                # stopped auto-filling them.
                memory_max_bytes=_process_scope._worker_memory_max_bytes(),
            )

        warning.assert_not_called()
        assert f"MemoryMax={123 * 1024 * 1024}" in argv

    def test_worker_memory_limit_caps_oversized_local_guard_override(
        self, monkeypatch
    ):
        import tools.process_registry as pr

        monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "999999")
        monkeypatch.setattr(
            pr.Path,
            "read_text",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no cgroup")),
        )
        monkeypatch.setattr(
            pr.os,
            "sysconf",
            lambda *_args: (_ for _ in ()).throw(OSError("no sysconf")),
        )

        assert _process_scope._worker_memory_max_bytes() == _process_scope._DEFAULT_WORKER_MEMORY_MAX_BYTES

    def test_kill_recovered_detached_already_exited_stops_persisted_scope(
        self, registry, monkeypatch
    ):
        """Recovered detached sessions whose wrapper PID is gone/recycled must
        still stop their persisted systemd scope before the already_exited
        return, while retaining the PID-reuse guard (no PID tree kill)."""
        session = _make_session(sid="proc_recovered_scope", command="daemonize")
        session.detached = True
        session.pid_scope = "host"
        session.pid = 12345
        session.host_start_time = 67890
        session.systemd_unit = "hermes-worker-proc_recovered_scope.scope"
        registry._running[session.id] = session

        stopped = []
        terminated = []
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda pid, start: False)
        monkeypatch.setattr(registry, "_terminate_host_pid", lambda pid, start: terminated.append((pid, start)))
        monkeypatch.setattr('tools.process_registry_scope._stop_systemd_unit', lambda unit: stopped.append(unit) or True)

        with patch.object(registry, "_write_checkpoint"):
            result = registry.kill_process(session.id)

        assert result["status"] == "already_exited"
        assert stopped == ["hermes-worker-proc_recovered_scope.scope"]
        assert terminated == []
        assert session.exited is True
        assert session.id in registry._finished
        assert session.id not in registry._running

    @pytest.mark.linux_only
    def test_systemd_run_user_scope_available_caches_after_probe(
        self, registry, monkeypatch
    ):
        """The availability check probes once and caches — a second call must
        not re-probe (and must return the same value)."""
        import tools.process_registry as pr

        # Reset the cache.
        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        first = _process_scope._systemd_run_user_scope_available()
        second = _process_scope._systemd_run_user_scope_available()
        assert first is True
        assert second is True
        assert len(probe_calls) == 1, "probe must run only once (cached)"
        # The probe must not carry OOMPolicy= either: that is the argv systemd
        # rejected on scope units and cached as "unavailable" (#102486).
        probe_argv = probe_calls[0][0]
        assert not any(
            value.startswith("OOMPolicy=") for value in probe_argv if isinstance(value, str)
        ), probe_argv

    @pytest.mark.linux_only
    def test_systemd_probe_derives_owned_user_bus_env_for_system_gateway(
        self, registry, monkeypatch, request
    ):
        """A system service running as an unprivileged user has no login env,
        but may still have a valid lingering user manager and D-Bus socket."""
        import socket
        import tempfile

        import tools.process_registry as pr

        # Short path: AF_UNIX socket paths are capped at ~104 bytes, longer than most tmp_path values.
        runtime_dir = pr.Path(tempfile.mkdtemp(prefix="hbus-", dir="/tmp"))
        runtime_dir.chmod(0o700)
        bus_path = runtime_dir / "bus"
        bus_socket = socket.socket(socket.AF_UNIX)
        bus_socket.bind(str(bus_path))

        def _cleanup():
            bus_socket.close()
            bus_path.unlink(missing_ok=True)
            runtime_dir.rmdir()

        request.addfinalizer(_cleanup)

        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(_process_scope, "_default_user_runtime_dir", lambda: runtime_dir)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        derived = _launch_registry.systemd_user_bus_env(
            {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/tmp/untrusted-bus"}
        )
        assert derived["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_path}"
        probe_kwargs = []

        def fake_run(*args, **kwargs):
            probe_kwargs.append(kwargs)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        assert _process_scope._systemd_run_user_scope_available() is True
        env = probe_kwargs[0]["env"]
        assert env["XDG_RUNTIME_DIR"] == str(runtime_dir)
        assert env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_path}"
        assert "XDG_RUNTIME_DIR" not in os.environ
        assert "DBUS_SESSION_BUS_ADDRESS" not in os.environ

    @pytest.mark.linux_only
    def test_probe_succeeds_without_bin_true(self, monkeypatch):
        """An absent ``/bin/true`` must not make a usable scope fail its probe."""
        import tools.process_registry as pr

        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_PROBED_AT", 0.0)
        real_run = subprocess.run
        executed = []

        def systemd_run_on_nixos_shaped_root(argv, **kwargs):
            # Simulate NixOS's missing executable, but run the selected replacement.
            payload = argv[argv.index("--") + 1 :]
            if payload[0] == "/bin/true":
                return subprocess.CompletedProcess(payload, 127, stderr=b"No such file or directory")
            executed.append(payload)
            return real_run(payload, **kwargs)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", systemd_run_on_nixos_shaped_root)

        assert _process_scope._systemd_run_user_scope_available() is True
        assert len(executed) == 1, "payload must really run (exit 0) on the host, not just be spelled right"

    @pytest.mark.linux_only
    def test_systemd_scope_first_probe_is_serialized(self, monkeypatch):
        """Concurrent first-use callers must wait for one definitive probe.

        A temporary cached ``False`` would let a racing worker spawn inside the
        gateway cgroup, defeating the OOM isolation guarantee.
        """
        import tools.process_registry as pr

        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
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
            target=lambda: results.append(_process_scope._systemd_run_user_scope_available())
        )
        second = threading.Thread(
            target=lambda: results.append(_process_scope._systemd_run_user_scope_available())
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

    @pytest.mark.linux_only
    def test_failed_systemd_probe_retries_after_cache_ttl(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_PROBED_AT", 0.0, raising=False)
        clock = [100.0]
        probe_results = [1, 0]
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(
                args=args[0], returncode=probe_results.pop(0)
            )

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("tools.process_registry.time.monotonic", lambda: clock[0])
        monkeypatch.setattr("subprocess.run", fake_run)

        assert _process_scope._systemd_run_user_scope_available() is False
        assert _process_scope._systemd_run_user_scope_available() is False
        assert len(probe_calls) == 1

        clock[0] += 61
        assert _process_scope._systemd_run_user_scope_available() is True
        assert len(probe_calls) == 2

    def test_stop_systemd_unit_treats_absent_unit_as_clean(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
        # Pass 9 (AH): the stop client runs through the cancellable
        # Popen+poll helper, not subprocess.run — mock that seam.
        monkeypatch.setattr(
            _process_scope, "_run_systemctl_cancellable",
            lambda *args, **kwargs: (
                5, b"", b"Unit hermes-worker-gone.scope not loaded.\n"
            ),
        )

        assert _process_scope._stop_systemd_unit("hermes-worker-gone.scope") is True

    @pytest.mark.macos_only
    def test_darwin_never_takes_scope_path_even_with_systemd_run_on_path(
        self, registry, monkeypatch, _gateway_identity
    ):
        """macOS no-op guarantee (#70716 cross-platform audit).

        On a native macOS host, the spawn path must be
        byte-identical to the legacy path even when a ``systemd-run``
        binary is somehow on PATH and the gateway identity checks pass:
        no probe, no wrapping, no unit recorded.
        """
        import tools.process_registry as pr

        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process", lambda: True
        )
        # If any branch consults the probe or builds a scope argv on darwin,
        # fail loudly.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/systemd-run")
        scope_builds = []
        real_build = _process_scope._build_systemd_scope_argv
        monkeypatch.setattr(
            _process_scope,
            "_build_systemd_scope_argv",
            lambda *a, **k: scope_builds.append(a) or real_build(*a, **k),
        )
        probe_runs = []

        def fake_probe_run(argv, **kwargs):
            probe_runs.append(argv)
            return subprocess.CompletedProcess(args=argv, returncode=0)

        monkeypatch.setattr("subprocess.run", fake_probe_run)

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            session = registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True
        assert session.systemd_unit == ""
        assert scope_builds == [], "darwin must never build a systemd scope argv"
        assert probe_runs == [], "darwin must never run the systemd-run probe"

    @pytest.mark.macos_only
    def test_probe_returns_false_off_linux(self, monkeypatch):
        """``_systemd_run_user_scope_available`` is False on non-Linux even
        when a ``systemd-run`` binary exists on PATH."""
        import tools.process_registry as pr

        monkeypatch.setattr(_process_scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/systemd-run")
        probe_runs = []
        monkeypatch.setattr(
            "subprocess.run",
            lambda argv, **kwargs: probe_runs.append(argv)
            or subprocess.CompletedProcess(args=argv, returncode=0),
        )

        assert _process_scope._systemd_run_user_scope_available() is False
        assert probe_runs == [], "non-Linux must not exec the probe"


_TERM_IGNORING_HELPER = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "with open(os.environ['HELPER_PID_FILE'], 'w') as f:\n"
    "    f.write(str(os.getpid()))\n"
    "time.sleep(60)\n"
)


def test_scope_stop_helper_cancel_reaps_term_ignoring_helper(
    tmp_path, monkeypatch,
):
    """Pass 10 (AM): the cancel path must kill AND unconditionally reap
    the helper. A helper that ignores SIGTERM proves the kill is a real
    SIGKILL, and a pathological caller poll interval (30 s, clamped to
    <=0.5 s) proves cancellation is observed promptly — a reaped child's
    pid is gone entirely, while a zombie would still answer ``kill(0)``.
    """
    from tools import process_registry as pr

    helper = tmp_path / "helper.py"
    helper.write_text(_TERM_IGNORING_HELPER)
    pid_file = tmp_path / "helper.pid"
    monkeypatch.setenv("HELPER_PID_FILE", str(pid_file))

    cancel = threading.Event()

    def cancel_once_running():
        for _ in range(200):  # <=10 s, no blind sleep
            if pid_file.exists():
                cancel.set()
                return
            time.sleep(0.05)

    watcher = threading.Thread(target=cancel_once_running, daemon=True)
    watcher.start()

    started = time.monotonic()
    result = _process_scope._run_systemctl_cancellable(
        [sys.executable, str(helper)],
        timeout=30,
        cancel_event=cancel,
        poll_interval=30.0,  # clamped to <=0.5 s internally
    )
    elapsed = time.monotonic() - started

    assert result is None, "cancelled helper reports the None verdict"
    assert elapsed < 5.0, (
        "cancellation must be observed within the clamped poll, not the "
        "caller's 30 s interval"
    )
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the killed helper was left alive or unreaped (zombie)")


def test_scope_stop_helper_reap_survives_kill_race(monkeypatch, tmp_path):
    """Pass 10 (AM): a helper exiting during the kill race can make the
    post-kill ``communicate`` raise; the swallowed exception used to
    skip reaping entirely. The unconditional ``wait`` must reap the
    child anyway, and the pipes must be closed."""
    from tools import process_registry as pr

    real_popen = pr.subprocess.Popen

    class _RaceyHelper:
        """Real child; communicate() breaks once killed (the race)."""

        def __init__(self, real):
            self._real = real
            self._killed = False
            self.waited = False

        def kill(self):
            self._killed = True
            return self._real.kill()

        def communicate(self, timeout=None):
            if self._killed:
                raise OSError("helper exited during the kill race")
            return self._real.communicate(timeout=timeout)

        def wait(self, timeout=None):
            self.waited = True
            return self._real.wait(timeout=timeout)

        @property
        def pid(self):
            return self._real.pid

        @property
        def returncode(self):
            return self._real.returncode

        @property
        def stdout(self):
            return self._real.stdout

        @property
        def stderr(self):
            return self._real.stderr

    wrapped: dict = {}

    def popen_racey(argv, *args, **kwargs):
        wrapped["proc"] = _RaceyHelper(real_popen(argv, *args, **kwargs))
        cancel.set()  # cancellation races a helper that has really launched
        return wrapped["proc"]

    monkeypatch.setattr(pr.subprocess, "Popen", popen_racey)

    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n")
    cancel = threading.Event()

    result = _process_scope._run_systemctl_cancellable(
        [sys.executable, str(script)], timeout=30, cancel_event=cancel,
    )

    assert result is None
    racey = wrapped["proc"]
    assert racey.waited, "the unconditional wait() ran despite the race"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(racey.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the race-killed helper was left unreaped (zombie)")
    assert racey.stdout.closed and racey.stderr.closed




@pytest.mark.linux_only
@pytest.mark.parametrize(
    ('events', 'child_pids', 'expected'),
    [('populated 1\n', '123\n', 'alive'),
     ('populated 0\n', '', 'dead'),
     ('populated broken\n', '', 'unknown'),
     ('frozen 0\n', '', 'unknown'),
     (None, '123\n', 'alive'),
     (None, '', 'dead'),
     (None, None, 'unknown')],
)
def test_scope_quiescence_uses_recursive_kernel_evidence(tmp_path, monkeypatch, events, child_pids, expected):
    """A loaded scope's empty root membership does not prove its children stopped."""
    scope = tmp_path / 'worker.scope'
    child = scope / 'nested'
    child.mkdir(parents=True)
    (scope / 'cgroup.procs').write_text('')
    if child_pids is not None:
        (child / 'cgroup.procs').write_text(child_pids)
    if events is not None:
        (scope / 'cgroup.events').write_text(events)
    helper = tmp_path / 'systemctl'
    helper.write_text(
        f'#!{sys.executable}\n'
        f'print("LoadState=loaded\\nActiveState=active\\nControlGroup={scope}")\n'
    )
    helper.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ.get('PATH', ''))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))

    assert _process_scope._scope_unit_liveness('worker.scope') == expected


@pytest.mark.linux_only
@pytest.mark.parametrize(
    ('load_state', 'evidence', 'expected'),
    [('not-found', 'populated', 'alive'),
     ('not-found', 'nested', 'alive'),
     ('not-found', 'absent', 'unknown'),
     ('not-found', 'unreadable', 'unknown'),
     ('not-found', 'empty', 'dead'),
     ('loaded', 'empty', 'dead')],
)
def test_missing_manager_unit_requires_recursive_kernel_evidence(
    tmp_path, monkeypatch, load_state, evidence, expected,
):
    """Manager absence cannot erase descendants left in the deterministic path."""
    mount = tmp_path / 'cgroup'
    uid = os.getuid()
    scope = (mount / 'user.slice' / f'user-{uid}.slice' /
             f'user@{uid}.service' / 'app.slice' / 'worker.scope')
    if evidence != 'absent':
        scope.mkdir(parents=True)
        (scope / 'cgroup.procs').write_text('')
        if evidence == 'unreadable':
            (scope / 'cgroup.events').mkdir()
        elif evidence == 'nested':
            child = scope / 'descendant'
            child.mkdir()
            (child / 'cgroup.procs').write_text('123\n')
        else:
            (scope / 'cgroup.events').write_text(
                'populated 1\n' if evidence == 'populated' else 'populated 0\n'
            )
    helper = tmp_path / 'systemctl'
    helper.write_text(
        f'#!{sys.executable}\n'
        f'print("LoadState={load_state}\\nActiveState=inactive\\nControlGroup=")\n'
    )
    helper.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ.get('PATH', ''))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(_process_scope, '_cgroup_mount_point', lambda: (2, str(mount)))

    assert _process_scope._scope_unit_liveness('worker.scope') == expected
    # Launch refusal has its own fresh-launch probe and is not inferred from
    # the reclamation result for a previously persisted unit.
    assert _process_scope._scope_unit_was_created('worker.scope') is (load_state == 'loaded')

from tools import process_registry as _launch_registry
