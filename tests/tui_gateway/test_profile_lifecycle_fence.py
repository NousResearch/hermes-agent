"""Session admission decides from disk state, never from the profile lease.

Regression for the PR #93508 review: ``ProfileLifecycleFence.rejected()`` took
the cross-process profile lease for a locally retired pathname while its callers
held ``_sessions_lock``. Lease holders take that lock in turn (publication admits
the generation, deletion tears sessions down), so both sides stalled for the full
lease timeout. A tokenless retirement also rejected its pathname forever, even
after another process published a successor generation.
"""

import threading
import time
from pathlib import Path

import pytest

from hermes_cli import profile_lifecycle, profiles
from hermes_cli.profile_incarnation import read_profile_incarnation
from tui_gateway.profile_lifecycle import ProfileLifecycleFence


@pytest.fixture
def server(tmp_path, monkeypatch):
    import tui_gateway.server as server

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_bot_desktop", "_notify_multiplexer"):
        monkeypatch.setattr(profiles, name, lambda *a, **k: None)
    monkeypatch.setattr(profiles, "_profile_bound_backend_pids", lambda *a, **k: [])
    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", lambda *_: [])
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_profile_lifecycle", ProfileLifecycleFence())
    return server


def test_admission_under_sessions_lock_never_waits_for_a_publishing_lease(server, monkeypatch):
    home = profiles.create_profile("worker", no_alias=True, no_skills=True)
    old = server._capture_profile_incarnation(home)
    assert profiles.delete_profile("worker", yes=True) == home  # retires `old` here

    # A regression waits this long for the lease instead of the production 120 s.
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS", 6.0)
    publishing = threading.Event()
    admit = profile_lifecycle.allow_in_process_profile_resources

    def signal_then_admit(*args, **kwargs):
        publishing.set()  # tombstone cleared, lease held, _sessions_lock needed next
        admit(*args, **kwargs)

    monkeypatch.setattr(profile_lifecycle, "allow_in_process_profile_resources", signal_then_admit)
    recreate = threading.Thread(
        target=profiles.create_profile, args=("worker",),
        kwargs={"no_alias": True, "no_skills": True}, daemon=True,
    )
    with server._sessions_lock:
        recreate.start()
        assert publishing.wait(30)
        new = read_profile_incarnation(home)
        started = time.monotonic()
        verdicts = (
            server._profile_home_rejected(home, old, require_incarnation=True),
            server._profile_home_rejected(home, new, require_incarnation=True),
        )
        waited = time.monotonic() - started
    recreate.join(30)

    assert waited < 2, f"admission waited {waited:.1f}s on the profile lease"
    assert verdicts == (True, False)
    assert not recreate.is_alive()
    assert server._capture_profile_incarnation(home) == new


def test_tokenless_retirement_admits_a_successor_published_elsewhere(server, monkeypatch, tmp_path):
    home = tmp_path / ".hermes" / "profiles" / "worker"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    # A delete begun before incarnation markers left a tokenless fence; its
    # retry retires the generation here without any token.
    profile_lifecycle.mark_profile_deleting(home)
    assert profiles.delete_profile("worker", yes=True) == home

    # Another process publishes the successor: its admission never reaches this process.
    monkeypatch.setattr(profile_lifecycle, "allow_in_process_profile_resources", lambda *a, **k: None)
    profiles.create_profile("worker", no_alias=True, no_skills=True)
    new = read_profile_incarnation(home)

    assert server._capture_profile_incarnation(home) == new
    assert not server._profile_home_rejected(home, new, require_incarnation=True)
    assert server._profile_home_rejected(home, None, require_incarnation=True)
