from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from hermes_cli.active_sessions import (
    active_session_liveness_guard,
    active_session_registry_snapshot,
    try_acquire_active_session,
)
from tui_gateway import server


_LEASE_HOLDER_SCRIPT = """
import os
import time
from pathlib import Path

from hermes_cli import active_sessions
from hermes_cli.active_sessions import try_acquire_active_session

boundary_file = os.environ.get("BOUNDARY_FILE")
if boundary_file:
    original_enter = active_sessions._FileLock.__enter__
    def instrumented_enter(self):
        Path(boundary_file).write_text("boundary", encoding="utf-8")
        return original_enter(self)
    active_sessions._FileLock.__enter__ = instrumented_enter

go_file = os.environ.get("GO_FILE")
if go_file:
    Path(os.environ["WAITING_FILE"]).write_text("waiting", encoding="utf-8")
    deadline = time.monotonic() + 120
    while not Path(go_file).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for acquisition signal")
        time.sleep(0.02)
lease, message = try_acquire_active_session(
    session_id=os.environ["SESSION_ID"],
    surface="desktop",
    config={},
    track_liveness=True,
)
assert lease is not None and message is None, message
Path(os.environ["READY_FILE"]).write_text("ready", encoding="utf-8")
try:
    deadline = time.monotonic() + 120
    release_file = Path(os.environ["RELEASE_FILE"])
    while not release_file.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for release signal")
        time.sleep(0.02)
finally:
    lease.release()
"""


def _spawn_lease_holder(
    *,
    home: Path,
    session_id: str,
    ready_file: Path,
    release_file: Path,
    boundary_file: Path | None = None,
    go_file: Path | None = None,
    waiting_file: Path | None = None,
) -> subprocess.Popen[str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            env.pop(key)
    env.update({
        "HERMES_HOME": str(home),
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(repo_root), env.get("PYTHONPATH", "")) if part
        ),
        "READY_FILE": str(ready_file),
        "RELEASE_FILE": str(release_file),
        "SESSION_ID": session_id,
    })
    if boundary_file is not None:
        env["BOUNDARY_FILE"] = str(boundary_file)
    if go_file is not None and waiting_file is not None:
        env["GO_FILE"] = str(go_file)
        env["WAITING_FILE"] = str(waiting_file)
    return subprocess.Popen(
        [sys.executable, "-c", _LEASE_HOLDER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_child_file(
    child: subprocess.Popen[str],
    path: Path,
    *,
    label: str,
    timeout: float = 60.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if child.poll() is not None:
            stdout, stderr = child.communicate()
            pytest.fail(
                f"{label} process exited before signalling readiness\n"
                f"stdout: {stdout}\nstderr: {stderr}"
            )
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out waiting for {label} process")
        time.sleep(0.02)


def _stop_child(child: subprocess.Popen[str], release_file: Path) -> None:
    release_file.touch()
    if child.poll() is None:
        child.kill()
    child.communicate()


def test_unlimited_session_lease_remains_noop_without_liveness_tracking(
    tmp_path: Path,
) -> None:
    home = tmp_path / "untracked-home"

    lease, message = try_acquire_active_session(
        session_id="untracked-session",
        surface="tui",
        config={},
        registry_home=home,
    )

    assert lease is not None and message is None
    assert lease.enabled is False
    assert active_session_registry_snapshot(registry_home=home) == []


def test_orphan_guard_fails_closed_when_registry_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import active_sessions

    def _unavailable(*_args, **_kwargs):
        raise OSError("registry unavailable")

    monkeypatch.setattr(
        active_sessions,
        "active_session_liveness_guard",
        _unavailable,
    )

    with server._other_runtime_lease_guard(
        "preserved-session",
        {"profile_home": None},
    ) as sibling_active:
        assert sibling_active is True


def test_desktop_claim_fails_closed_when_registry_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: (_ for _ in ()).throw(OSError("config unavailable")),
    )

    desktop_lease, desktop_message = server._claim_active_session_slot(
        "desktop-session",
        live_session_id="desktop-runtime",
        surface="desktop",
    )
    tui_lease, tui_message = server._claim_active_session_slot(
        "tui-session",
        live_session_id="tui-runtime",
        surface="tui",
    )

    assert desktop_lease is None
    assert desktop_message == server._SESSION_OWNERSHIP_UNAVAILABLE
    assert (tui_lease, tui_message) == (None, None)


def test_server_release_retries_liveness_lease_before_dropping_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Lease:
        enabled = True
        released = False
        track_liveness = True
        calls = 0

        def release(self):
            self.calls += 1
            if self.calls == 1:
                raise OSError("replace failed once")
            self.released = True

    lease = _Lease()
    session = {"active_session_lease": lease}
    monkeypatch.setattr(server.time, "sleep", lambda *_args: None)

    assert server._release_active_session_slot(session) is True
    assert lease.calls == 2
    assert "active_session_lease" not in session


def test_automatic_cleanup_preserves_corrupt_registry_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_home = tmp_path / "profile-home"
    state_path = profile_home / "runtime" / "active_sessions.json"
    state_path.parent.mkdir(parents=True)
    corrupt = "{not-json"
    state_path.write_text(corrupt, encoding="utf-8")
    ended: list[tuple[str, str]] = []

    class _FakeDB:
        def get_session(self, target: str) -> dict[str, str]:
            return {"id": target, "source": "desktop"}

        def end_session(self, target: str, reason: str) -> None:
            ended.append((target, reason))

    @contextlib.contextmanager
    def _profile_db(_session: dict):
        yield _FakeDB()

    monkeypatch.setattr(server, "_session_db", _profile_db)
    monkeypatch.setattr(
        server, "_notify_session_boundary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session", lambda *args, **kwargs: None
    )
    session = {
        "agent": None,
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": str(profile_home),
        "session_key": "preserved-session",
        "slash_worker": None,
        "source": "desktop",
    }

    server._finalize_session(session, end_reason="idle_timeout")

    assert ended == []
    assert state_path.read_text(encoding="utf-8") == corrupt


def test_compression_ownership_failure_quiesces_until_next_rpc_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    outcomes = iter((False, True))
    monkeypatch.setattr(
        server,
        "_transfer_active_session_slot",
        lambda *_args, **_kwargs: next(outcomes),
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(approval, "unregister_gateway_notify", lambda *_args: None)
    monkeypatch.setattr(approval, "register_gateway_notify", lambda *_args: None)
    monkeypatch.setattr(approval, "is_session_yolo_enabled", lambda *_args: False)
    session = {
        "agent": types.SimpleNamespace(session_id="session-new"),
        "session_key": "session-old",
    }
    server._sessions["runtime-1"] = session

    try:
        with pytest.raises(RuntimeError, match="ownership transfer failed"):
            server._sync_session_key_after_compress("runtime-1", session)
        assert session["session_key"] == "session-old"
        assert session["_ownership_sync_pending"] == "session-new"

        resumed, err = server._sess_nowait({"session_id": "runtime-1"}, "retry-request")
        assert err is None
        assert resumed is session
        assert session["session_key"] == "session-new"
        assert "_ownership_sync_pending" not in session
    finally:
        server._sessions.pop("runtime-1", None)


def test_liveness_guard_serializes_cross_process_acquire(tmp_path: Path) -> None:
    home = tmp_path / "guard-home"
    waiting_file = tmp_path / "child-waiting"
    boundary_file = tmp_path / "child-lock-boundary"
    go_file = tmp_path / "child-go"
    acquired_file = tmp_path / "child-acquired"
    release_file = tmp_path / "child-release"
    session_id = "guarded-session"
    child: subprocess.Popen[str] | None = None

    try:
        child = _spawn_lease_holder(
            home=home,
            session_id=session_id,
            ready_file=acquired_file,
            release_file=release_file,
            boundary_file=boundary_file,
            go_file=go_file,
            waiting_file=waiting_file,
        )
        _wait_for_child_file(child, waiting_file, label="lease contender bootstrap")

        with active_session_liveness_guard(
            session_id,
            registry_home=home,
        ) as active:
            assert active is False
            go_file.write_text("go", encoding="utf-8")
            _wait_for_child_file(child, boundary_file, label="lease lock boundary")
            time.sleep(0.25)
            assert not acquired_file.exists()
            assert child.poll() is None

        assert child is not None
        _wait_for_child_file(child, acquired_file, label="lease contender")
        release_file.write_text("release", encoding="utf-8")
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, f"stdout: {stdout}\nstderr: {stderr}"
        assert active_session_registry_snapshot(registry_home=home) == []
    finally:
        if child is not None:
            _stop_child(child, release_file)


def test_automatic_desktop_cleanup_preserves_sibling_and_ends_sole_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every automatic cleanup reason must preserve another Desktop backend."""
    profile_home = tmp_path / "profile-home"
    ready_file = tmp_path / "child-ready"
    release_file = tmp_path / "child-release"
    session_id = "shared-profile-session"
    child = _spawn_lease_holder(
        home=profile_home,
        session_id=session_id,
        ready_file=ready_file,
        release_file=release_file,
    )
    ended: list[tuple[str, str]] = []

    class _FakeDB:
        def get_session(self, target: str) -> dict[str, str] | None:
            return {"id": target, "source": "desktop"}

        def end_session(self, target: str, reason: str) -> None:
            ended.append((target, reason))

    @contextlib.contextmanager
    def _profile_db(_session: dict):
        yield _FakeDB()

    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_session_db", _profile_db)
    monkeypatch.setattr(
        server, "_notify_session_boundary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session", lambda *args, **kwargs: None
    )

    def _session(lease) -> dict:
        return {
            "active_session_lease": lease,
            "agent": None,
            "history": [],
            "history_lock": threading.Lock(),
            "profile_home": str(profile_home),
            "session_key": session_id,
            "slash_worker": None,
            "source": "desktop",
        }

    try:
        _wait_for_child_file(child, ready_file, label="lease holder")

        reasons = (
            "ws_orphan_reap",
            "ws_disconnect",
            "idle_timeout",
            "lru_evict",
            "tui_shutdown",
        )
        assert set(reasons) == server._AUTOMATIC_SESSION_END_REASONS

        for index, reason in enumerate(reasons):
            local_lease, message = server._claim_active_session_slot(
                session_id,
                live_session_id=f"local-runtime-{index}",
                surface="desktop",
                profile_home=profile_home,
            )
            assert local_lease is not None and message is None
            assert (
                len(active_session_registry_snapshot(registry_home=profile_home)) == 2
            )

            server._finalize_session(_session(local_lease), end_reason=reason)

            assert ended == []
            remaining = active_session_registry_snapshot(registry_home=profile_home)
            assert len(remaining) == 1
            assert remaining[0]["session_id"] == session_id

        # Explicit user close retains force/end semantics even with a sibling.
        explicit_lease, message = server._claim_active_session_slot(
            session_id,
            live_session_id="explicit-runtime",
            surface="desktop",
            profile_home=profile_home,
        )
        assert explicit_lease is not None and message is None
        server._finalize_session(_session(explicit_lease), end_reason="tui_close")
        assert ended == [(session_id, "tui_close")]
        ended.clear()

        release_file.write_text("release", encoding="utf-8")
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, f"stdout: {stdout}\nstderr: {stderr}"
        assert active_session_registry_snapshot(registry_home=profile_home) == []

        for index, reason in enumerate(reasons):
            sole_lease, message = server._claim_active_session_slot(
                session_id,
                live_session_id=f"sole-runtime-{index}",
                surface="desktop",
                profile_home=profile_home,
            )
            assert sole_lease is not None and message is None

            server._finalize_session(_session(sole_lease), end_reason=reason)
            assert active_session_registry_snapshot(registry_home=profile_home) == []

        assert ended == [(session_id, reason) for reason in reasons]
    finally:
        _stop_child(child, release_file)
