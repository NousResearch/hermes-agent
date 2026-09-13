"""Process ownership, checkpoint recovery, teardown and notification contracts."""

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


class TestCheckpoint:
    def test_recover_dead_pid(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_dead",
            "command": "sleep 999",
            "pid": 999999999,  # almost certainly not running
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0

    def test_recover_dead_wrapper_retries_unreaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            'tools.process_registry_scope._stop_systemd_unit', return_value=False
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        assert json.loads(checkpoint.read_text()) == [entry]

    def test_recover_dead_wrapper_drops_reaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            'tools.process_registry_scope._stop_systemd_unit', return_value=True
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        assert json.loads(checkpoint.read_text()) == []

    def test_cgroup2_mount_point_prefers_canonical_mount(self, tmp_path):
        """With several cgroup2 mounts (host + container bind), the
        canonical ``/sys/fs/cgroup`` mount wins — it is where systemd's
        ControlGroup paths resolve."""
        from tools.process_registry_scope import _cgroup2_mount_point

        mi = tmp_path / "mountinfo"
        mi.write_text(
            "34 33 253:0 / / rw,relatime - ext4 /dev/mapper/root rw\n"
            "36 35 0:30 / /custom/cgroup rw,nosuid,nodev,noexec - cgroup2 none rw\n"
            "37 35 0:31 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 none rw\n"
        )
        assert _cgroup2_mount_point(str(mi)) == "/sys/fs/cgroup"

    def test_cgroup2_mount_point_uses_sole_custom_mount(self, tmp_path):
        """A container exposes exactly one cgroup2 mount, mounted
        wherever its namespace places it — that one is used."""
        from tools.process_registry_scope import _cgroup2_mount_point

        mi = tmp_path / "mountinfo"
        mi.write_text(
            "34 33 253:0 / / rw,relatime - ext4 /dev/mapper/root rw\n"
            "36 35 0:30 / /host/cgroupv2 rw,nosuid,nodev,noexec - cgroup2 none rw\n"
        )
        assert _cgroup2_mount_point(str(mi)) == "/host/cgroupv2"

    def test_cgroup2_mount_point_falls_back_when_unreadable_or_absent(
        self, tmp_path
    ):
        """Unreadable mountinfo (non-Linux) or no cgroup2 entry falls
        back to the canonical mount instead of erroring."""
        from tools.process_registry_scope import _CGROUP_V2_MOUNT_FALLBACK
        from tools.process_registry_scope import _cgroup2_mount_point

        missing = tmp_path / "does-not-exist"
        assert _cgroup2_mount_point(str(missing)) == _CGROUP_V2_MOUNT_FALLBACK
        no_cgroup2 = tmp_path / "mountinfo"
        no_cgroup2.write_text(
            "34 33 253:0 / / rw,relatime - ext4 /dev/mapper/root rw\n"
        )
        assert _cgroup2_mount_point(str(no_cgroup2)) == _CGROUP_V2_MOUNT_FALLBACK

    def test_scope_cgroup_procs_path_joins_derived_mount_point(
        self, tmp_path, monkeypatch
    ):
        """cgroupfs-relative ControlGroup values join onto the REAL
        cgroup v2 mount point, not a hardcoded /sys/fs/cgroup prefix —
        joining onto the wrong prefix yields an unopenable path, which
        liveness would otherwise misread as verified death.  Absolute
        paths outside the known slices (a test shim's state dir) are
        honoured as-is, and a unit systemd did not report falls back
        to the canonical user-scope location under the real mount."""
        import tools.process_registry as pr

        monkeypatch.setattr(
            _process_scope, "_cgroup_mount_point", lambda *a, **k: (2, "/custom/cgroup")
        )

        assert _process_scope._scope_cgroup_procs_path(
            "w.scope", "/user.slice/user-1000.slice/user@1000.service/app.slice/w.scope"
        ) == (
            "/custom/cgroup/user.slice/user-1000.slice/"
            "user@1000.service/app.slice/w.scope/cgroup.procs"
        )
        # Absolute path outside the standard slices: honoured as-is.
        assert _process_scope._scope_cgroup_procs_path(
            "w.scope", f"{tmp_path}/units/w.scope"
        ) == f"{tmp_path}/units/w.scope/cgroup.procs"
        # No ControlGroup reported: canonical user-scope layout derived
        # from the real uid under the real mount.
        monkeypatch.setattr(pr.platform, "system", lambda: "Linux")
        expected = (
            f"/custom/cgroup/user.slice/user-{os.getuid()}.slice/"
            f"user@{os.getuid()}.service/app.slice/w.scope/cgroup.procs"
        )
        assert _process_scope._scope_cgroup_procs_path("w.scope", "") == expected

    def test_cgroup_mount_point_v1_prefers_systemd_hierarchy(
        self, tmp_path, monkeypatch
    ):
        """Gate B pass 4 (S): a v1 host (no cgroup2 mount) resolves its
        procs paths against the systemd controller hierarchy's mount —
        the hierarchy ControlGroup paths are relative to — not the
        hardcoded /sys/fs/cgroup root, which on such hosts yields a
        nonexistent path and liveness stuck at "unknown" forever."""
        import tools.process_registry as pr

        mi = tmp_path / "mountinfo"
        mi.write_text(
            "30 23 0:24 / /sys ro,nosuid - sysfs sysfs rw\n"
            "36 30 0:26 / /sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,xattr,name=systemd\n"
            "37 30 0:27 / /sys/fs/cgroup/pids rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,pids\n"
            "38 30 0:28 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,memory\n",
            encoding="utf-8",
        )
        assert _process_scope._cgroup_mount_point(str(mi)) == (1, "/sys/fs/cgroup/systemd")
        assert _process_scope._cgroup_v1_controller_mount(
            str(mi), preferred=("memory",)
        ) == "/sys/fs/cgroup/memory"
        assert _process_scope._cgroup_v1_controller_mount(
            str(mi), preferred=("pids",)
        ) == "/sys/fs/cgroup/pids"
        # cgroupfs-relative ControlGroup joins the v1 systemd mount
        # (resolved against the same fake mountinfo).
        monkeypatch.setattr(
            _process_scope, "_cgroup_mount_point",
            lambda *a, **k: (1, "/sys/fs/cgroup/systemd"),
        )
        assert _process_scope._scope_cgroup_procs_path(
            "w.scope", "/user.slice/user-1000.slice/app.slice/w.scope"
        ) == (
            "/sys/fs/cgroup/systemd/user.slice/user-1000.slice/"
            "app.slice/w.scope/cgroup.procs"
        )

    def test_cgroup_mount_point_prefers_v2_when_both_mounted(self, tmp_path):
        """A hybrid host with both hierarchies mounted: the unified
        hierarchy wins (its mount, canonical preferred) — matches the
        pre-existing v2 behaviour exactly."""
        import tools.process_registry as pr

        mi = tmp_path / "mountinfo"
        mi.write_text(
            "36 30 0:26 / /host/cgroupv2 rw,relatime - cgroup2 cgroup2 rw\n"
            "37 30 0:27 / /sys/fs/cgroup/systemd rw,relatime - cgroup cgroup rw,name=systemd\n",
            encoding="utf-8",
        )
        assert _process_scope._cgroup_mount_point(str(mi)) == (2, "/host/cgroupv2")

    def test_cgroup_mount_point_no_hierarchy_is_definite_unsupported(
        self, tmp_path, monkeypatch,
    ):
        """A readable mountinfo with NO cgroup mount of either version
        is a DEFINITE (0, None): cgroup verification is impossible by
        construction, surfaced as "unsupported" so callers fall back to
        PID semantics instead of retrying "unknown" forever. An
        UNREADABLE mountinfo keeps the legacy canonical-v2 assumption
        ("unknown", not a new definite verdict)."""
        import tools.process_registry as pr

        mi = tmp_path / "mountinfo"
        mi.write_text(
            "30 23 0:24 / / rw - apfs /dev/disk1 rw\n",
            encoding="utf-8",
        )
        assert _process_scope._cgroup_mount_point(str(mi)) == (0, None)
        # Unreadable mountinfo (non-Linux): legacy fallback, not a
        # definite verdict.
        assert _process_scope._cgroup_mount_point(str(tmp_path / "missing")) == (
            2, "/sys/fs/cgroup",
        )
        # Canonical-prefix ControlGroup cannot be joined to anything.
        monkeypatch.setattr(
            _process_scope, "_cgroup_mount_point", lambda *a, **k: (0, None)
        )
        assert _process_scope._scope_cgroup_procs_path(
            "w.scope", "/user.slice/user-1000.slice/app.slice/w.scope"
        ) is None

    def test_scope_unit_liveness_unsupported_when_no_hierarchy(
        self, monkeypatch,
    ):
        """Liveness on a cgroupfs-less host with a loaded unit returns
        the definite "unsupported" (and active_state maps it through) —
        not "unknown", which callers would retry forever."""
        import subprocess as _sp
        import tools.process_registry as pr

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
        monkeypatch.setattr(
            _process_scope, "_cgroup_mount_point", lambda *a, **k: (0, None)
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: _sp.CompletedProcess(
                args=(a[0] if a else k.get("args", [])),
                returncode=0,
                stdout=(
                    b"LoadState=loaded\nActiveState=active\n"
                    b"ControlGroup=/user.slice/user-1000.slice/app.slice/w.scope\n"
                ),
            ),
        )
        assert _process_scope._scope_unit_liveness("w.scope") == "unsupported"
        assert _process_scope._scope_unit_active_state("w.scope") == "unsupported"

    def test_scope_unit_bus_inactive_confirms_only_terminal_bus_states(
        self, monkeypatch,
    ):
        """The bus-truth fallback for cgroupfs-less hosts: not-found /
        inactive / failed confirm, deactivating and loaded-active do
        not."""
        import subprocess as _sp
        import tools.process_registry as pr

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")

        def fake_show(props: bytes):
            return lambda *a, **k: _sp.CompletedProcess(
                args=(a[0] if a else k.get("args", [])),
                returncode=0,
                stdout=props,
            )

        for props, expected in [
            (b"LoadState=not-found\nActiveState=inactive\n", True),
            (b"LoadState=loaded\nActiveState=inactive\n", True),
            (b"LoadState=loaded\nActiveState=failed\n", True),
            (b"LoadState=loaded\nActiveState=deactivating\n", False),
            (b"LoadState=loaded\nActiveState=active\n", False),
        ]:
            monkeypatch.setattr("subprocess.run", fake_show(props))
            assert _process_scope._scope_unit_bus_inactive("w.scope") is expected

    def test_stop_systemd_unit_verified_confirms_via_bus_on_unsupported(
        self, monkeypatch,
    ):
        """On a cgroupfs-less host the verified stop still terminates:
        after the stop lands, the bus reporting the unit inactive is
        the confirmation — previously every stop burned the full
        escalation budget and re-queued forever."""
        import subprocess as _sp
        import tools.process_registry as pr

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
        monkeypatch.setattr(
            _process_scope, "_cgroup_mount_point", lambda *a, **k: (0, None)
        )
        show_props: dict[str, bytes] = {"now": b"LoadState=loaded\nActiveState=active\n"}
        # **kwargs: the verified stop threads its cancel plumbing into
        # the stop call (pass 9, AH).
        monkeypatch.setattr(
            _process_scope, "_stop_systemd_unit",
            lambda unit, **kwargs: show_props.__setitem__(
                "now", b"LoadState=loaded\nActiveState=inactive\n"
            ),
        )

        def fake_run(*a, **k):
            return _sp.CompletedProcess(
                args=(a[0] if a else k.get("args", [])),
                returncode=0,
                stdout=show_props["now"],
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        # The SIGKILL escalation client is the cancellable helper now.
        monkeypatch.setattr(
            _process_scope, "_run_systemctl_cancellable", lambda *a, **k: (0, b"", b"")
        )
        monkeypatch.setattr(_process_scope, "_collect_dead_systemd_unit", lambda unit: None)

        assert _process_scope._stop_systemd_unit_verified("w.scope") is True

    def test_worker_memory_limit_reads_v1_memory_controller(self, tmp_path):
        """v1 host: the limit comes from the memory controller
        hierarchy's memory.limit_in_bytes (mounted per controller), not
        the hardcoded /sys/fs/cgroup v2 layout."""
        import tools.process_registry as pr

        mi = tmp_path / "mountinfo"
        mem_mount = tmp_path / "memv1"
        mi.write_text(
            f"36 30 0:26 / {mem_mount} rw,relatime - cgroup cgroup rw,memory\n",
            encoding="utf-8",
        )
        rel = "user.slice/user-1000.slice/user@1000.service"
        limit_dir = tmp_path / "memv1" / rel
        limit_dir.mkdir(parents=True)
        (limit_dir / "memory.limit_in_bytes").write_text("536870912\n")
        self_cgroup = tmp_path / "self_cgroup"
        self_cgroup.write_text(
            f"10:memory:/{rel}\n1:name=systemd:/{rel}\n", encoding="utf-8"
        )

        bound = _process_scope._worker_memory_max_bytes(
            str(self_cgroup), str(mi)
        )
        assert bound == 536870912  # 512 MiB — tighter than half of RAM

    def test_recovery_skips_explicit_sandbox_backed_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        original = [{
            "session_id": "proc_remote",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "pid_scope": "sandbox",
        }]
        checkpoint.write_text(json.dumps(original))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0
            assert registry.get("proc_remote") is None

            data = json.loads(checkpoint.read_text())
            assert data == []

    def test_checkpoint_redacts_command_with_inline_secret(self, registry, tmp_path):
        """Issue #77484: the checkpoint file persists raw commands; inline
        credentials (e.g. ``curl -H 'Authorization: Bearer sk-...'``) must be
        redacted before write. Recovery only uses command for display/logging
        (the process is already running), so masking is lossless."""
        checkpoint = tmp_path / "procs.json"
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            secret = "sk-secret1234567890"
            command = f"curl -H 'Authorization: Bearer {secret}' http://x"
            s = _make_session(sid="proc_secret", command=command)
            s.pid = 12345
            s.host_start_time = int(time.time())
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads(checkpoint.read_text())
            assert data[0]["session_id"] == "proc_secret"
            assert secret not in data[0]["command"]
            assert data[0]["command"] != command


class TestKillProcess:
    def test_kill_already_exited(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        result = registry.kill_process(s.id)
        assert result["status"] == "already_exited"


    def test_kill_detached_session_uses_host_pid(self, registry):
        s = _make_session(sid="proc_detached", command="sleep 999")
        s.pid = 424242
        s.detached = True
        registry._running[s.id] = s

        terminate_calls = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
            def children(self, recursive=False):
                return []
            def terminate(self):
                terminate_calls.append(("terminate", self.pid))

        import psutil as _psutil

        try:
            # Post-#21561: liveness probe routes through
            # ``ProcessRegistry._is_host_pid_alive`` (→
            # ``gateway.status._pid_exists``), and the actual kill on POSIX
            # routes through ``psutil.Process(pid).terminate()``. Neither
            # touches ``os.kill`` directly. Mock both seams.  Disable the
            # SIGKILL-escalation step (grace=0) so it doesn't call
            # ``psutil.wait_procs`` on the FakeProcess.
            with patch("gateway.status._pid_exists", return_value=True), \
                 patch.object(ProcessRegistry, "_daemon_term_grace_seconds",
                              staticmethod(lambda: 0.0)), \
                 patch.object(_psutil, "Process", side_effect=lambda pid: FakeProcess(pid)):
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert ("terminate", 424242) in terminate_calls
        finally:
            registry._running.pop(s.id, None)


class TestProcessToolHandler:
    def test_unknown_action(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "unknown_action"}))
        assert "error" in result


from tools.process_registry_notifications import format_process_notification


def test_drain_notifications_completion_callback_exception_fails_closed(registry):
    event = {
        "type": "completion",
        "session_id": "proc_callback_error",
        "session_key": "session-a",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "done",
    }
    registry.completion_queue.put(event)

    def broken(_event):
        raise RuntimeError("ownership check exploded")

    results = registry.drain_notifications(
        session_key="session-a",
        owns_event=broken,
    )

    assert results == []
    assert registry.completion_queue.get_nowait() == event
    assert registry.completion_queue.empty()


def test_drain_notifications_filters_async_delegation_by_session_key():
    """Async-delegation events should only be consumed by the matching session's drain.

    Regression test for issue #58684: background delegation results delivered
    to the wrong session when the user switches sessions while a subagent runs.
    """
    from tools.process_registry import process_registry

    # Clear the queue first
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        # Put events for different sessions
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_a",
            "session_key": "telegram:dm:111:user_a",
            "goal": "task A",
            "status": "completed",
            "summary": "done A",
            "api_calls": 1,
            "duration_seconds": 0.5,
        })
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_b",
            "session_key": "telegram:dm:222:user_b",
            "goal": "task B",
            "status": "completed",
            "summary": "done B",
            "api_calls": 1,
            "duration_seconds": 0.3,
        })

        # Drain for session A — should only get deleg_session_a
        results_a = process_registry.drain_notifications(session_key="telegram:dm:111:user_a")
        assert len(results_a) == 1, (
            f"Expected 1 event for session A, got {len(results_a)}"
        )
        assert results_a[0][0]["delegation_id"] == "deleg_session_a"
        assert "done A" in results_a[0][1]

        # Session B's event should have been re-queued — drain for session B
        results_b = process_registry.drain_notifications(session_key="telegram:dm:222:user_b")
        assert len(results_b) == 1, (
            f"Expected 1 event for session B, got {len(results_b)}"
        )
        assert results_b[0][0]["delegation_id"] == "deleg_session_b"
        assert "done B" in results_b[0][1]

        # No more events should remain
        assert process_registry.completion_queue.empty()
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_drain_notifications_owns_event_callback_beats_key_equality():
    """The positive-proof ownership callback consumes ONLY approved events —
    including across a compression rotation where bare key equality would
    wrongly re-queue the session's own pre-compression dispatch (#55578)."""
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        # Pre-compression dispatch: event carries the OLD key.
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_precompress",
            "session_key": "old_parent_key",
            "goal": "task", "status": "completed", "summary": "mine",
            "api_calls": 1, "duration_seconds": 0.1,
        })
        # Foreign event that plain key equality would also reject.
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_foreign",
            "session_key": "someone_else",
            "goal": "task", "status": "completed", "summary": "not mine",
            "api_calls": 1, "duration_seconds": 0.1,
        })

        # Chain-aware ownership: this session's lineage includes old_parent_key.
        lineage = {"old_parent_key", "new_child_key"}
        results = process_registry.drain_notifications(
            session_key="new_child_key",
            owns_event=lambda e: e.get("session_key") in lineage,
        )
        assert [r[0]["delegation_id"] for r in results] == ["deleg_precompress"]

        # The foreign event was re-queued, not consumed.
        leftover = process_registry.completion_queue.get_nowait()
        assert leftover["delegation_id"] == "deleg_foreign"
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


class TestTerminateHostPidWindows:
    """Windows branch uses ``taskkill /T /F`` — the documented MS tree-kill
    primitive. We can't use psutil's ``children(recursive=True)`` /
    ``.terminate()`` path on Windows because (1) Windows doesn't maintain
    a Unix-style process tree so the walk is unreliable, and (2)
    ``Process.terminate()`` on Windows is ``TerminateProcess()`` for the
    target handle only, not the tree.
    """

    @pytest.mark.windows_only
    def test_windows_invokes_taskkill_with_tree_and_force_flags(self, monkeypatch):
        """The Windows branch must shell out to ``taskkill /PID N /T /F``.

        Windows-only: ``taskkill.exe`` is the thing under test and only exists
        here — with a faked ``_IS_WINDOWS`` the argv was asserted against a
        binary that could never have run.
        """
        from tools import process_registry as pr

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(pr.subprocess, "run", fake_run)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert captured["args"][0] == "taskkill"
        assert "/PID" in captured["args"]
        assert "12345" in captured["args"]
        assert "/T" in captured["args"], "Tree flag required to reach descendants"
        assert "/F" in captured["args"], "Force flag required for headless Chromium"


class TestTerminateHostPidPosix:
    """POSIX branch walks the tree via psutil and SIGTERMs children first."""

    def test_posix_walks_tree_and_terminates_children_then_parent(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        terminate_order = []

        class _FakeChild:
            def __init__(self, pid):
                self.pid = pid

            def terminate(self):
                terminate_order.append(self.pid)

        class _FakeParent:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=False):
                assert recursive is True
                return [_FakeChild(101), _FakeChild(102), _FakeChild(103)]

            def terminate(self):
                terminate_order.append(self.pid)

        monkeypatch.setattr(psutil, "Process", _FakeParent)
        # This test covers only the SIGTERM tree-walk ordering; disable the
        # SIGKILL-escalation step (which would call psutil.wait_procs on the
        # fakes) by setting the grace to 0.
        monkeypatch.setattr(pr.ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.0))

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert terminate_order == [101, 102, 103, 12345], (
            "Children must be terminated before the parent"
        )

    def test_posix_oserror_falls_back_to_os_kill(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        def boom(pid):
            raise PermissionError("can't read /proc")

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        monkeypatch.setattr(psutil, "Process", boom)
        monkeypatch.setattr(pr.os, "kill", fake_kill)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert kill_calls == [(12345, signal.SIGTERM)]


class TestPidReuseGuard:
    def test_terminate_refuses_when_start_time_mismatches(self, registry):
        """A live PID whose start time changed (recycled) is NOT killed."""
        proc = _spawn_python_sleep(30)
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert real_start is not None, "no /proc start time on this platform?"
            # Simulate recycling: the recorded baseline no longer matches.
            registry._terminate_host_pid(proc.pid, expected_start=real_start + 1)
            # The process must still be alive — the guard refused to signal it.
            assert not _wait_until(lambda: proc.poll() is not None, timeout=0.3)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()


    def test_refresh_detached_marks_recycled_pid_exited(self, registry):
        """A detached session whose PID got recycled is moved to finished."""
        wrong_start = (ProcessRegistry._safe_host_start_time(os.getpid()) or 0) + 999
        s = _make_session(sid="proc_detached")
        s.pid = os.getpid()          # alive, but...
        s.pid_scope = "host"
        s.detached = True
        s.host_start_time = wrong_start  # ...identity no longer matches
        registry._running[s.id] = s
        refreshed = registry._refresh_detached_session(s)
        assert refreshed.exited is True
        assert s.id in registry._finished


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX SIGTERM→SIGKILL escalation; Windows uses taskkill /F")
class TestSigkillEscalation:
    """Bounded SIGTERM→SIGKILL escalation in _terminate_host_pid.

    A daemon that ignores/stalls on SIGTERM must be force-killed after the
    configured grace window so it can't leak indefinitely — while well-behaved
    processes still exit cleanly on SIGTERM and the recycled-PID guard is never
    bypassed.
    """

    # A process that traps SIGTERM (ignores it): only SIGKILL stops it.
    # It prints "ready" AFTER installing the handler so the parent never
    # signals it during the startup window (before SIG_IGN is in place).
    _TRAP = (
        "import signal, sys, time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "sys.stdout.write('ready\\n'); sys.stdout.flush();"
        "[time.sleep(0.2) for _ in iter(int, 1)]"
    )

    def _spawn_trap(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", self._TRAP],
            stdout=subprocess.PIPE, text=True,
        )
        # Wait until the handler is installed before returning.
        line = proc.stdout.readline()
        assert line.strip() == "ready", "trap process failed to start"
        return proc

    def test_sigterm_ignoring_daemon_is_sigkilled(self, monkeypatch):
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.3))
        proc = self._spawn_trap()
        try:
            ProcessRegistry._terminate_host_pid(proc.pid)
            assert _wait_until(lambda: proc.poll() is not None, timeout=4.0), \
                "SIGTERM-ignoring daemon should be SIGKILLed after grace"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_escalation_does_not_bypass_recycled_pid_guard(self, monkeypatch):
        """A start-time mismatch must still spare the PID — no SIGTERM, no SIGKILL."""
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.3))
        proc = self._spawn_trap()
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            ProcessRegistry._terminate_host_pid(
                proc.pid, expected_start=(real_start or 0) + 1)
            assert not _wait_until(lambda: proc.poll() is not None, timeout=0.3)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_grace_reader_floors_at_zero(self, monkeypatch):
        """A negative configured grace is clamped to 0 (no escalation)."""
        import hermes_cli.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "read_raw_config",
                            lambda: {"terminal": {"daemon_term_grace_seconds": -5}})
        assert ProcessRegistry._daemon_term_grace_seconds() == 0.0

    @pytest.mark.live_system_guard_bypass
    def test_entire_tree_is_sigkilled_not_just_parent(self, monkeypatch):
        """A SIGTERM-ignoring parent + children are ALL force-killed.

        Regression: an earlier implementation trusted psutil.wait_procs's
        gone/alive partition, which mis-partitioned across a parent/child tree
        and left survivors un-killed (flaky — sometimes the parent lived,
        sometimes a child). The escalation now re-probes every target directly.
        """
        import psutil
        # 2.0s grace (not 1.0): with three interpreters mid-startup on a
        # loaded runner, a 1s SIGTERM->partition window races child spawn and
        # is how a child PID escaped the live-system guard in CI.
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 2.0))
        # Parent spawns 2 children; all trap SIGTERM. Parent prints child pids
        # after the handler is installed.
        parent_src = (
            "import signal, subprocess, sys, time;"
            "child='import signal,time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
            "[time.sleep(0.2) for _ in iter(int,1)]';"
            "kids=[subprocess.Popen([sys.executable,'-c',child]) for _ in range(2)];"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "sys.stdout.write(' '.join(str(k.pid) for k in kids)+'\\n'); sys.stdout.flush();"
            "[time.sleep(0.2) for _ in iter(int,1)]"
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_src],
                                  stdout=subprocess.PIPE, text=True)
        # Bound the readline: if the parent wedges before printing, fail THIS
        # test with a clear message instead of letting the per-file timeout
        # SIGKILL the whole pytest process (opaque rc=124 in CI).
        import select as _select
        ready, _, _ = _select.select([parent.stdout], [], [], 20.0)
        assert ready, "parent process failed to print child pids within 20s"
        child_pids = [int(x) for x in parent.stdout.readline().split()]
        all_pids = [parent.pid] + child_pids
        try:
            ProcessRegistry._terminate_host_pid(parent.pid)

            def _pid_dead(p: int) -> bool:
                # A pid is "dead" for our purposes if it no longer exists OR
                # exists only as an unreaped zombie (already terminated, just
                # not reaped by its reparented parent yet). psutil can also
                # raise mid-probe if the pid vanishes between the existence
                # check and the status read — treat any such race as dead.
                try:
                    if not psutil.pid_exists(p):
                        return True
                    return not ProcessRegistry._proc_alive(psutil.Process(p))
                except Exception:
                    return True

            def _all_dead():
                return all(_pid_dead(p) for p in all_pids)

            # _terminate_host_pid SIGKILLs synchronously before returning, so
            # the kill signals are already delivered here. The only remaining
            # wait is the kernel tearing down 3 processes and the reparented
            # children transitioning to zombie — which can lag on a loaded CI
            # runner. Give a generous budget (matches the wait() test's 10s)
            # so this asserts the escalation BEHAVIOR, not the runner's
            # scheduling latency. The assertion itself never weakens: every
            # tree member must end up dead/zombie.
            assert _wait_until(_all_dead, timeout=15.0, interval=0.02), (
                "entire SIGTERM-ignoring tree (parent + children) must be SIGKILLed"
            )
        finally:
            for p in all_pids:
                try:
                    os.kill(p, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            parent.wait()


class TestHandleProcessRedaction:
    """`_handle_process` redacts background-process output before it reaches the
    model / session.db / CLI display — issue #43025.

    Mirrors the foreground `terminal` redaction so the two surfaces can't
    diverge. Env-dump commands (`printenv`/`env`) get the ENV-assignment pass
    so opaque tokens are masked; other commands stay on the code_file path.
    """

    def _setup(self, monkeypatch, command, output):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact1", command=command)
        sess.output_buffer = output
        sess.exited = True
        sess.exit_code = 0
        reg._running.clear()
        reg._finished[sess.id] = sess
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        return pr, sess

    def test_log_redacts_env_dump_opaque_token(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "printenv",
            "MY_SERVICE_TOKEN=abc123randomopaquetokenvalue999\nHOME=/home/u",
        )
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "abc123randomopaquetokenvalue999" not in out["output"]
        assert "HOME=/home/u" in out["output"]

    def test_poll_redacts_prefix_key(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "python app.py",
            "leaked OPENAI_API_KEY sk-proj-abc123def456ghi789jkl012 here",
        )
        out = json.loads(pr._handle_process({"action": "poll", "session_id": sess.id}))
        assert "abc123def456" not in out["output_preview"]

    def test_list_redacts_command_and_output(self, monkeypatch):
        """`process(action=list)` redacts command + output_preview — issue #77484.

        The list branch previously returned raw ``command[:200]`` and
        ``output_preview[-200:]`` with no redaction wrap, leaking inline
        secrets (unlike poll/log/wait/kill).
        """
        pr, sess = self._setup(
            monkeypatch, "curl -H 'Authorization: Bearer sk-abc123def456ghi789jkl012345'",
            "opaque token sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGG output",
        )
        out = json.loads(pr._handle_process({"action": "list"}))
        assert len(out["processes"]) >= 1
        entry = out["processes"][0]
        assert "sk-abc123def456ghi789jkl012345" not in entry["command"]
        assert "sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGG" not in entry["output_preview"]
        assert "curl" in entry["command"]

    def test_disabled_passes_through(self, monkeypatch):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", False)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact2", command="printenv")
        sess.output_buffer = "CUSTOM_TOKEN=zzzopaque1234567890abcdef"
        sess.exited = True
        sess.exit_code = 0
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "zzzopaque1234567890abcdef" in out["output"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: select() on pipes")
class TestReaderLoopOrphanedPipe:
    """Regression tests for issue #68915.

    When an agent command backgrounds a long-lived process (``node server.js
    &``), the grandchild inherits the write end of the reader's stdout pipe.
    The direct bash child exits, but the pipe never EOFs — the old blocking
    ``read1()`` parked the reader thread forever, ``session.exited`` never
    flipped on its own, and ``notify_on_complete`` never fired. The reader
    must instead terminate shortly after the direct child exits, even while
    a descendant still holds the pipe open.
    """

    def test_reader_exits_when_orphan_holds_pipe(self, registry):
        """Reader loop must return promptly after the direct child exits,
        even though a backgrounded descendant keeps the pipe open."""
        proc = subprocess.Popen(
            ["sh", "-c", "echo started; sleep 30 & exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=os.setsid,
        )
        s = _make_session(sid="proc_orphan_reader")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        done = threading.Event()

        def _run():
            registry._reader_loop(s)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            # The direct child exits immediately; the reader must notice and
            # return well before the 30s descendant releases the pipe.
            assert done.wait(timeout=10.0), (
                "_reader_loop is still blocked on the orphan-held pipe "
                "(issue #68915) — session.exited would never flip and "
                "notify_on_complete would never fire"
            )
            assert s.exited is True
            assert s.exit_code == 0
            assert s.completion_reason == "exited"
            assert "started" in s.output_buffer
            assert s.id in registry._finished
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def test_reader_exit_fires_notify_on_complete(self, registry):
        """The autonomous completion notification must not depend on a
        poll()/wait() call when an orphan holds the pipe."""
        proc = subprocess.Popen(
            ["sh", "-c", "sleep 30 & echo bg-started"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=os.setsid,
        )
        s = _make_session(sid="proc_orphan_notify")
        s.process = proc
        s.pid = proc.pid
        s.notify_on_complete = True
        registry._running[s.id] = s

        done = threading.Event()

        def _run():
            registry._reader_loop(s)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            assert done.wait(timeout=10.0), (
                "_reader_loop blocked — completion notification lost (#68915)"
            )
            # Exactly one completion event must have been queued.
            item = registry.completion_queue.get_nowait()
            assert item["type"] == "completion"
            assert item["session_id"] == s.id
            assert item["exit_code"] == 0
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class TestNotificationRedaction:
    """Background-process notification delivery (completion_queue) applies the
    same redaction as the explicit process tool — issue #43025 gap.

    The _move_to_finished() and _check_watch_patterns() paths enqueue raw
    output into the completion_queue.  After the fix, _redact_process_result()
    is called before enqueueing so secrets are masked in the [IMPORTANT: ...]
    messages delivered to the LLM.
    """

    def test_completion_notification_redacts_secret(self, monkeypatch):
        """_move_to_finished completion notification redacts API keys."""
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr

        reg = ProcessRegistry()
        sess = _make_session(sid="proc_notif1", command="env")
        sess.output_buffer = "OPENAI_API_KEY=sk-proj-secret123\nHOME=/home/u"
        sess.notify_on_complete = True
        sess.exited = True
        sess.exit_code = 0
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)

        reg._move_to_finished(sess)

        # Drain and check the notification
        results = reg.drain_notifications()
        assert len(results) == 1
        _evt, text = results[0]
        assert "sk-proj-secret123" not in text
        assert "REDACTED" in text or "sk-proj" not in text

    def test_watch_match_notification_redacts_secret(self, monkeypatch):
        """_check_watch_patterns watch_match notification redacts secrets."""
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr

        reg = ProcessRegistry()
        sess = _make_session(sid="proc_notif2", command="python server.py")
        sess.output_buffer = "Server started\nAPI_TOKEN=ghp_abc123def456\nListening on :8080"
        sess.watch_patterns = ["API_TOKEN"]
        sess._watch_disabled = False
        sess._watch_hits = 0
        sess._watch_suppressed = 0
        sess.watcher_platform = None
        sess.watcher_chat_id = None
        sess.watcher_user_id = None
        sess.watcher_user_name = None
        sess.watcher_thread_id = None
        sess.watcher_message_id = None
        sess.exited = False
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)

        reg._check_watch_patterns(sess, "API_TOKEN=ghp_abc123def456\n")

        results = reg.drain_notifications()
        assert len(results) == 1
        _evt, text = results[0]
        assert "ghp_abc123def456" not in text
        assert "ghp_" not in text or "REDACTED" in text


class TestGetByPrefix:
    """ProcessRegistry.get() resolves unique ID prefixes like git short hashes."""

    def test_full_id_still_exact(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4dae56ca81f6") is s

    def test_unique_prefix_resolves(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4dae5") is s

    def test_bare_suffix_resolves(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("4dae56") is s

    def test_finished_sessions_also_resolve(self, registry):
        s = _make_session(sid="proc_9bee77aa0011", exited=True, exit_code=0)
        registry._finished[s.id] = s
        assert registry.get("proc_9bee") is s

    def test_ambiguous_prefix_returns_none(self, registry):
        a = _make_session(sid="proc_4dae56ca81f6")
        b = _make_session(sid="proc_4dae99999999")
        registry._running[a.id] = a
        registry._running[b.id] = b
        assert registry.get("proc_4dae") is None

    def test_too_short_prefix_returns_none(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4da") is None
        assert registry.get("4da") is None
        assert registry.get("proc_") is None
        assert registry.get("") is None

    def test_exact_id_wins_over_prefix_scan(self, registry):
        # A session whose FULL id happens to be a prefix of another's must
        # resolve to itself, never trigger the ambiguity path.
        short = _make_session(sid="proc_4dae")
        long = _make_session(sid="proc_4dae56ca81f6")
        registry._running[short.id] = short
        registry._running[long.id] = long
        assert registry.get("proc_4dae") is short

    def test_no_match_returns_none(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_ffff") is None

    def test_poll_accepts_prefix(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6", output="hello world")
        registry._running[s.id] = s
        result = registry.poll("4dae56ca")
        assert result["session_id"] == "proc_4dae56ca81f6"
        assert result["status"] == "running"


def _make_delegation_batch_evt(results):
    """A batch async-delegation event carrying a per-task ``results`` list."""
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_97654",
        "is_batch": True,
        "results": results,
        "goals": [r.get("goal") or "" for r in results],
        "session_key": "agent:main:cli:dm:local",
        "status": "completed",
        "model": "upstage/solar-pro-4",
    }


def _patch_delegation_config(
    monkeypatch, model="upstage/solar-pro-4", provider="openrouter", **over
):
    import tools.process_registry_notifications as _prn

    cfg = {"model": model, "provider": provider}
    cfg.update(over)
    monkeypatch.setattr(_prn, "_delegation_config", lambda: cfg)
    return cfg


def _format_async(evt) -> str:
    from tools.process_registry_notifications import format_process_notification

    text = format_process_notification(evt)
    assert text is not None, "format_process_notification returned None"
    return text


def test_model_not_found_notice_single_failure_once(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "exit_reason": "error",
            "goal": "Create bridge module",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
            "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        }
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text is not None
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text
    assert "openrouter" in text
    assert "No fallback chain is configured" in text


def test_model_not_found_notice_mixed_batch_named_model(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "exit_reason": "error",
            "goal": "A",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
            "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        },
        {
            "task_index": 1,
            "status": "completed",
            "goal": "B",
            "summary": "ok",
            "api_calls": 3,
        },
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text


def test_model_not_found_notice_absent_for_non_model_errors(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 429: rate limit exceeded",
        },
        {
            "task_index": 1,
            "status": "failed",
            "goal": "B",
            "error": "Connection timed out",
        },
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert "SUBAGENT MODEL REJECTED" not in text


def test_model_not_found_notice_absent_when_configured_model_not_named(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 400: gpt-99 is not a valid model ID",
        }
    ])
    # Configured model is upstage/solar-pro-4; the rejection names gpt-99.
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert "SUBAGENT MODEL REJECTED" not in text


def test_model_not_found_notice_single_dispatch(monkeypatch):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_single",
        "session_key": "agent:main:cli:dm:local",
        "goal": "task A",
        "model": "upstage/solar-pro-4",
        "status": "failed",
        "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
    }
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text


def test_model_not_found_notice_absent_when_fallback_chain_configured(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        }
    ])
    _patch_delegation_config(
        monkeypatch,
        fallback_providers=[{"provider": "openrouter", "model": "upstage/solar-pro4"}],
    )
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "No fallback chain is configured" not in text
