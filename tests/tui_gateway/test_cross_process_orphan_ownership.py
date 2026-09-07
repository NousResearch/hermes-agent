from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import active_sessions
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


def test_unlimited_session_lease_is_real_even_without_a_cap(
    tmp_path: Path,
) -> None:
    """No cap configured must still fence the session (#94595).

    The old contract returned a disabled no-op lease here, which meant two
    processes could run one stored session concurrently by default.
    """
    home = tmp_path / "untracked-home"

    lease, message = try_acquire_active_session(
        session_id="untracked-session",
        surface="tui",
        config={},
        registry_home=home,
    )

    assert lease is not None and message is None
    assert lease.enabled is True
    entries = active_session_registry_snapshot(registry_home=home)
    assert [entry["session_id"] for entry in entries] == ["untracked-session"]
    lease.release()
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
    # Every surface fails closed now (#94595): a claim that errored has not
    # proven the session is unowned, and proceeding leaseless reopens the
    # double-writer hole.
    assert tui_lease is None
    assert tui_message == server._SESSION_OWNERSHIP_UNAVAILABLE


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


def test_own_live_lease_ids_reports_live_owners_and_skips_the_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Lease:
        def __init__(self, lease_id: str) -> None:
            self.lease_id = lease_id

    first = _Lease("first")
    second = _Lease("second")
    monkeypatch.setattr(
        server,
        "_sessions",
        {
            "one": {"active_session_lease": first},
            "two": {"active_session_lease": second},
            "three": {"active_session_lease": None},
        },
    )

    assert server._own_live_lease_ids() == {"first", "second"}
    assert server._own_live_lease_ids(exclude=first) == {"second"}


def test_automatic_cleanup_reclaims_own_orphan_lease_not_treated_as_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_home = tmp_path / "profile-home"
    session_id = "own-orphan-session"
    owner_lease, message = server._claim_active_session_slot(
        session_id,
        live_session_id="vanished-runtime",
        surface="desktop",
        profile_home=profile_home,
    )
    assert owner_lease is not None and message is None
    # The owner vanished minutes ago; a lease written seconds ago is still
    # inside the self-orphan grace window and must be left alone.
    monkeypatch.setattr(
        "hermes_cli.active_sessions._SELF_ORPHAN_GRACE_SECONDS", 0.0
    )
    ended: list[tuple[str, str]] = []

    class _FakeDB:
        def get_session(self, target: str) -> dict[str, str]:
            return {"id": target, "source": "desktop"}

        def end_session(self, target: str, reason: str) -> None:
            ended.append((target, reason))

    @contextlib.contextmanager
    def _profile_db(_session: dict):
        yield _FakeDB()

    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_session_db", _profile_db)
    monkeypatch.setattr(
        server, "_notify_session_boundary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session", lambda *args, **kwargs: None
    )
    session = {
        "active_session_lease": None,
        "agent": None,
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": str(profile_home),
        "session_key": session_id,
        "slash_worker": None,
        "source": "desktop",
    }

    server._finalize_session(session, end_reason="ws_orphan_reap")

    assert ended == [(session_id, "ws_orphan_reap")]
    assert active_session_registry_snapshot(registry_home=profile_home) == []


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

        # With the per-session fence (#94595) this backend can no longer claim
        # a second lease on a session another backend owns — the exact
        # double-writer state the fence exists to prevent.
        refused_lease, refusal = server._claim_active_session_slot(
            session_id,
            live_session_id="local-runtime",
            surface="desktop",
            profile_home=profile_home,
        )
        assert refused_lease is None
        assert getattr(refusal, "reason", None) == "SESSION_NOT_OWNED"
        assert (
            len(active_session_registry_snapshot(registry_home=profile_home)) == 1
        )

        # A LEASELESS local record of that session (a viewer / never-ran-a-turn
        # tab) must still preserve the sibling's session on every automatic
        # cleanup reason: the lifecycle guard consults the registry directly.
        for reason in reasons:
            server._finalize_session(_session(None), end_reason=reason)

            assert ended == []
            remaining = active_session_registry_snapshot(registry_home=profile_home)
            assert len(remaining) == 1
            assert remaining[0]["session_id"] == session_id

        # Explicit user close retains force/end semantics even with a sibling.
        server._finalize_session(_session(None), end_reason="tui_close")
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


# ── #104691: the reclaim sweep must not vouch for dead-lane records ──

def _acquire_root_lease(session_id: str, live_session_id: str):
    lease, message = try_acquire_active_session(
        session_id=session_id,
        surface="desktop",
        config={},
        metadata={"live_session_id": live_session_id},
        track_liveness=True,
    )
    assert lease is not None and message is None
    return lease


def _dead_lane_record(lease, *, last_active: float, created_at: float, transport=None, running: bool = False, agent_ready=None) -> dict:
    return {
        "active_session_lease": lease,
        "transport": server._detached_ws_transport if transport is None else transport,
        "running": running,
        "last_active": last_active,
        "created_at": created_at,
        "agent_ready": agent_ready,
        "source": "desktop",
        "session_key": "zombie-session",
    }


def test_reclaim_preserves_record_building_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record whose agent is still building keeps its lease (agent_ready unset, non-lazy)."""
    _pin_reclaim_env(monkeypatch)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    monkeypatch.setattr(
        server,
        "_sessions",
        {"ui": _dead_lane_record(lease, last_active=old, created_at=old,
                                agent_ready=threading.Event())},
    )

    try:
        server._reclaim_orphaned_leases()
        assert [e["session_id"] for e in active_session_registry_snapshot()] == [
            "zombie-session"
        ]
    finally:
        lease.release()


def _pin_reclaim_env(monkeypatch: pytest.MonkeyPatch, *, floor: float = 0.0, grace: float = 0.0, delegations: bool = False) -> None:
    monkeypatch.setattr(server, "_LEASE_RECLAIM_IDLE_S", floor, raising=False)
    monkeypatch.setattr("hermes_cli.active_sessions._SELF_ORPHAN_GRACE_SECONDS", grace)
    monkeypatch.setattr(
        server, "_session_has_active_delegations", lambda sid, session=None: delegations
    )


def test_reclaim_drops_lease_backed_by_dead_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core #104691 repro: a lease vouched only by a dead-lane record is reclaimed."""
    _pin_reclaim_env(monkeypatch)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    monkeypatch.setattr(
        server, "_sessions", {"ui": _dead_lane_record(lease, last_active=old, created_at=old)}
    )

    assert server._own_live_lease_ids() == set()
    server._reclaim_orphaned_leases()
    assert active_session_registry_snapshot() == []
    # The dead lease object is detached too: the next submit must claim a fresh
    # fenced lease instead of short-circuiting on the stale object.
    assert "active_session_lease" not in server._sessions["ui"]
    assert lease.released is True


def test_reclaim_preserves_lease_backed_by_running_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-turn record (running) still owns its lease, even on a dead transport."""
    _pin_reclaim_env(monkeypatch)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    monkeypatch.setattr(
        server,
        "_sessions",
        {"ui": _dead_lane_record(lease, last_active=old, created_at=old, running=True)},
    )

    try:
        server._reclaim_orphaned_leases()
        assert [e["session_id"] for e in active_session_registry_snapshot()] == [
            "zombie-session"
        ]
    finally:
        lease.release()


def test_reclaim_preserves_lease_with_live_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record on a live transport is not a dead lane, however idle it is."""
    _pin_reclaim_env(monkeypatch)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    monkeypatch.setattr(
        server,
        "_sessions",
        {"ui": _dead_lane_record(lease, last_active=old, created_at=old, transport=object())},
    )

    try:
        server._reclaim_orphaned_leases()
        assert [e["session_id"] for e in active_session_registry_snapshot()] == [
            "zombie-session"
        ]
    finally:
        lease.release()


def test_reclaim_preserves_lease_with_active_delegations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live background work keeps the lane (and its lease) alive."""
    _pin_reclaim_env(monkeypatch, delegations=True)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    monkeypatch.setattr(
        server, "_sessions", {"ui": _dead_lane_record(lease, last_active=old, created_at=old)}
    )

    try:
        server._reclaim_orphaned_leases()
        assert [e["session_id"] for e in active_session_registry_snapshot()] == [
            "zombie-session"
        ]
    finally:
        lease.release()


def test_reclaim_preserves_young_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recently active lane is not idle past the reclaim floor."""
    _pin_reclaim_env(monkeypatch, floor=300.0)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    now = time.time()
    monkeypatch.setattr(
        server, "_sessions", {"ui": _dead_lane_record(lease, last_active=now, created_at=now)}
    )

    try:
        assert server._own_live_lease_ids() == {lease.lease_id}
        server._reclaim_orphaned_leases()
        assert [e["session_id"] for e in active_session_registry_snapshot()] == [
            "zombie-session"
        ]
    finally:
        lease.release()


def test_reclaim_preserves_record_without_activity_clocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an idle floor set, a record without activity clocks keeps its lease.

    Idleness is unprovable without ``created_at``/``last_active``: fail closed and keep
    vouching (an operator-set floor of 0 opts into transport-death alone). See #104691.
    """
    _pin_reclaim_env(monkeypatch, floor=300.0)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    monkeypatch.setattr(
        server, "_sessions", {"ui": _dead_lane_record(lease, last_active=0.0, created_at=0.0)}
    )

    try:
        server._reclaim_orphaned_leases()
        assert [e["session_id"] for e in active_session_registry_snapshot()] == [
            "zombie-session"
        ]
    finally:
        lease.release()


@pytest.mark.live_system_guard_bypass
def test_reclaim_never_drops_foreign_pid_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reclaim only ever drops this process's leases; a live sibling keeps its own."""
    hermes_home = Path(os.environ["HERMES_HOME"])
    worker_home = hermes_home / "profiles" / "worker"
    ready_file = tmp_path / "foreign-ready"
    release_file = tmp_path / "foreign-release"
    child = _spawn_lease_holder(
        home=worker_home,
        session_id="foreign-session",
        ready_file=ready_file,
        release_file=release_file,
    )
    try:
        _wait_for_child_file(child, ready_file, label="foreign lease holder")
        _pin_reclaim_env(monkeypatch)
        lease = _acquire_root_lease("zombie-session", "runtime-a")
        old = time.time() - 7200.0
        monkeypatch.setattr(
            server,
            "_sessions",
            {"ui": _dead_lane_record(lease, last_active=old, created_at=old)},
        )

        server._reclaim_orphaned_leases()

        assert active_session_registry_snapshot() == []
        remaining = active_session_registry_snapshot(registry_home=worker_home)
        assert [e["session_id"] for e in remaining] == ["foreign-session"]
    finally:
        release_file.write_text("release", encoding="utf-8")
        if child.poll() is None:
            child.terminate()
        try:
            child.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()


def test_reclaim_race_with_concurrent_submit_leaves_consistent_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sweep and submit serialize on the registry lock; no lost updates, no crash."""
    _pin_reclaim_env(monkeypatch)
    lease = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    monkeypatch.setattr(
        server, "_sessions", {"ui": _dead_lane_record(lease, last_active=old, created_at=old)}
    )
    errors: list[Exception] = []

    def _sweep() -> None:
        try:
            for _ in range(10):
                server._reclaim_orphaned_leases()
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)

    def _churn() -> None:
        try:
            for _ in range(10):
                churn_lease, _message = try_acquire_active_session(
                    session_id="churn-session",
                    surface="desktop",
                    config={},
                    metadata={"live_session_id": "churn-runtime"},
                    track_liveness=True,
                )
                if churn_lease is not None:
                    churn_lease.release()
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)

    workers = [threading.Thread(target=_sweep), threading.Thread(target=_churn)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)

    assert errors == []
    assert active_session_registry_snapshot() == []


def test_submit_after_reclaim_claims_fresh_fenced_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submit that loses the detach race must not run lease-less.

    The sweep can drop the registry row before the lease object is detached from its
    record. ``_ensure_active_session_slot`` treats a released-but-still-attached lease
    as absent and claims a fresh one, so the per-session fence never lapses.
    """
    _pin_reclaim_env(monkeypatch)
    stale = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    stale.released = True  # the sweep already dropped the registry row
    record = _dead_lane_record(stale, last_active=old, created_at=old)
    monkeypatch.setattr(server, "_sessions", {"ui": record})

    claims: list[str] = []

    def _fake_claim(session_key, *, live_session_id, surface="tui", profile_home=None):
        claims.append(live_session_id)
        return object(), None

    monkeypatch.setattr(server, "_claim_active_session_slot", _fake_claim)

    assert server._ensure_active_session_slot("ui", record) is None
    assert claims == ["ui"]  # claimed a fresh lease instead of short-circuiting
    assert record["active_session_lease"] is not stale
    assert stale.released is True


def test_reclaim_same_sid_reconnect_stays_fenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-sid resurrection racing the sweep must converge to a fenced session.

    Fault injection for #104691 blocker 1: the vouch snapshot judges the lane dead,
    then the same ``sid`` reconnects (live transport, fresh activity) before the
    registry mutation. The stale snapshot still deletes the durable row, but the
    receipt detaches the lease object and marks it released — so the live session's
    next submit claims a fresh fenced lease instead of running on a rowless object,
    and a foreign backend stays refused.
    """
    _pin_reclaim_env(monkeypatch)
    stale = _acquire_root_lease("zombie-session", "runtime-a")
    old = time.time() - 7200.0
    record = _dead_lane_record(stale, last_active=old, created_at=old)
    monkeypatch.setattr(server, "_sessions", {"ui": record})
    real_receipt = active_sessions.release_orphaned_leases_receipt

    def _reconnect_mid_sweep(live_ids: set[str]):
        record["transport"] = object()  # the lane is live again
        record["last_active"] = time.time()
        return real_receipt(live_ids)

    monkeypatch.setattr(
        "hermes_cli.active_sessions.release_orphaned_leases_receipt", _reconnect_mid_sweep
    )

    server._reclaim_orphaned_leases()

    # The stale row is gone, but so is the object: detached + released.
    assert active_session_registry_snapshot() == []
    assert record.get("active_session_lease") is None
    assert stale.released is True

    # The live session re-claims instead of short-circuiting lease-less.
    assert server._ensure_active_session_slot("ui", record) is None
    fresh = record.get("active_session_lease")
    assert fresh is not None and fresh is not stale
    assert [e["session_id"] for e in active_session_registry_snapshot()] == [
        "zombie-session"
    ]

    # A foreign backend remains fenced while our fresh row exists.
    other, refusal = try_acquire_active_session(
        session_id="zombie-session",
        surface="desktop",
        config={},
        metadata={"live_session_id": "foreign-runtime"},
        track_liveness=True,
    )
    assert other is None
    assert getattr(refusal, "reason", None) == "SESSION_NOT_OWNED"
    assert fresh is not None
    fresh.release()


def test_reclaim_partial_home_failure_stays_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A home whose sweep fails keeps its records attached and vouched (#104691).

    The receipt only settles homes the sweep provably mutated. The failed home's
    durable row survives, so detaching its lease object would strand the next submit
    on a row it no longer matches — it must stay attached until a later tick settles it.
    """
    hermes_home = Path(os.environ["HERMES_HOME"])
    worker_home = hermes_home / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    _pin_reclaim_env(monkeypatch)
    root_lease = _acquire_root_lease("zombie-session", "runtime-a")
    worker_lease, message = try_acquire_active_session(
        session_id="worker-session",
        surface="desktop",
        config={},
        metadata={"live_session_id": "runtime-w"},
        track_liveness=True,
        registry_home=worker_home,
    )
    assert worker_lease is not None and message is None
    old = time.time() - 7200.0
    monkeypatch.setattr(
        server,
        "_sessions",
        {
            "ui": _dead_lane_record(root_lease, last_active=old, created_at=old),
            "ui-w": {
                **_dead_lane_record(worker_lease, last_active=old, created_at=old),
                "session_key": "worker-session",
            },
        },
    )
    real_in_home = active_sessions._release_orphaned_leases_in_home

    def _flaky_home(home: Path, live_ids: set[str]):
        if str(home) == str(worker_home):
            raise OSError("injected home failure")
        return real_in_home(home, live_ids)

    monkeypatch.setattr(
        "hermes_cli.active_sessions._release_orphaned_leases_in_home", _flaky_home
    )

    server._reclaim_orphaned_leases()

    # Settled home: row gone, object detached + released.
    assert active_session_registry_snapshot() == []
    assert server._sessions["ui"].get("active_session_lease") is None
    assert root_lease.released is True
    # Failed home: row kept, object attached + unreleased.
    assert server._sessions["ui-w"].get("active_session_lease") is worker_lease
    assert worker_lease.released is False
    remaining = active_session_registry_snapshot(registry_home=worker_home)
    assert [e["session_id"] for e in remaining] == ["worker-session"]
    worker_lease.release()
