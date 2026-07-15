from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
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

from hermes_cli.active_sessions import try_acquire_active_session

attempted_file = os.environ.get("ATTEMPTED_FILE")
if attempted_file:
    Path(attempted_file).write_text("attempted", encoding="utf-8")
lease, message = try_acquire_active_session(
    session_id=os.environ["SESSION_ID"],
    surface="desktop",
    config={},
    track_liveness=True,
)
assert lease is not None and message is None, message
Path(os.environ["READY_FILE"]).write_text("ready", encoding="utf-8")
try:
    deadline = time.monotonic() + 30
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
    attempted_file: Path | None = None,
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
    if attempted_file is not None:
        env["ATTEMPTED_FILE"] = str(attempted_file)
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
    timeout: float = 10.0,
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


def test_liveness_guard_serializes_cross_process_acquire(tmp_path: Path) -> None:
    home = tmp_path / "guard-home"
    attempted_file = tmp_path / "child-attempted"
    acquired_file = tmp_path / "child-acquired"
    release_file = tmp_path / "child-release"
    session_id = "guarded-session"
    child: subprocess.Popen[str] | None = None

    try:
        with active_session_liveness_guard(
            session_id,
            registry_home=home,
        ) as active:
            assert active is False
            child = _spawn_lease_holder(
                home=home,
                session_id=session_id,
                ready_file=acquired_file,
                release_file=release_file,
                attempted_file=attempted_file,
            )
            _wait_for_child_file(child, attempted_file, label="lease contender")
            assert not acquired_file.exists()

        assert child is not None
        _wait_for_child_file(child, acquired_file, label="lease contender")
        release_file.write_text("release", encoding="utf-8")
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, f"stdout: {stdout}\nstderr: {stderr}"
        assert active_session_registry_snapshot(registry_home=home) == []
    finally:
        if child is not None:
            _stop_child(child, release_file)


def test_ws_orphan_reap_preserves_session_owned_by_other_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile sidecar must not end another backend's live durable session."""
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

        local_lease, message = server._claim_active_session_slot(
            session_id,
            live_session_id="local-runtime",
            surface="desktop",
            profile_home=profile_home,
        )
        assert local_lease is not None and message is None
        assert len(active_session_registry_snapshot(registry_home=profile_home)) == 2

        server._finalize_session(_session(local_lease), end_reason="ws_orphan_reap")

        assert ended == []
        remaining = active_session_registry_snapshot(registry_home=profile_home)
        assert len(remaining) == 1
        assert remaining[0]["session_id"] == session_id

        release_file.write_text("release", encoding="utf-8")
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, f"stdout: {stdout}\nstderr: {stderr}"
        assert active_session_registry_snapshot(registry_home=profile_home) == []

        sole_lease, message = server._claim_active_session_slot(
            session_id,
            live_session_id="sole-runtime",
            surface="desktop",
            profile_home=profile_home,
        )
        assert sole_lease is not None and message is None

        server._finalize_session(_session(sole_lease), end_reason="ws_orphan_reap")

        assert ended == [(session_id, "ws_orphan_reap")]
        assert active_session_registry_snapshot(registry_home=profile_home) == []
    finally:
        _stop_child(child, release_file)
