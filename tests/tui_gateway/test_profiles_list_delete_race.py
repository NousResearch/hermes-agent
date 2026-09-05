"""Profile listing must not recreate a profile deleted mid-refresh."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import hermes_state
import hermes_state_repair
import tui_gateway.server as srv
from hermes_cli import config, profile_lifecycle, profiles
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: fake_home / ".local" / "bin")
    with srv._sessions_lock:
        srv._profile_lifecycle.retired_homes.clear()
        srv._profile_lifecycle.retired_incarnations.clear()
    return hermes_home


def test_missing_named_profile_never_falls_back_to_launch_home(home: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._profile_home("ghost")

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._methods["session.create"]("rid-ghost", {"profile": "ghost"})


def test_provider_auth_write_cannot_resurrect_deleted_profile(home: Path) -> None:
    from hermes_cli import auth

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=profile_dir / "state.db").close()
    profiles.delete_profile("worker", yes=True)
    token = set_hermes_home_override(profile_dir)
    try:
        with pytest.raises(FileNotFoundError, match="missing or being deleted"):
            auth._update_config_for_provider("openrouter", "https://example.invalid/api")
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True


def test_direct_auth_store_save_cannot_resurrect_deleted_profile(home: Path) -> None:
    from hermes_cli import auth

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    profiles.delete_profile("worker", yes=True)
    token = set_hermes_home_override(profile_dir)
    try:
        with pytest.raises(FileNotFoundError, match="missing or being deleted"):
            auth._save_auth_store({"providers": {}})
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True


def test_credential_mirror_rewrite_cannot_recreate_profile_deleted_before_commit(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import credential_lifecycle
    import utils

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text(
        "model:\n  api_key: old-secret\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(profile_dir)
    real_atomic_write = utils.atomic_yaml_write

    def delete_before_commit(path, data, **kwargs):
        profile_lifecycle.mark_profile_deleting(profile_dir)
        shutil.rmtree(profile_dir)
        return real_atomic_write(path, data, **kwargs)

    monkeypatch.setattr(utils, "atomic_yaml_write", delete_before_commit)
    try:
        with pytest.raises(FileNotFoundError):
            credential_lifecycle._scrub_config_yaml_mirrors(
                "old-secret",
                "new-secret",
            )
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True


def test_zeroed_db_quarantine_cannot_recreate_profile_deleted_after_precheck(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    db_path.write_bytes(b"\0" * 64)
    real_zeroed_check = hermes_state.is_zeroed_state_db
    deleted = False

    def delete_during_zeroed_check(path, *args, **kwargs):
        nonlocal deleted
        if Path(path) == db_path and not deleted:
            assert real_zeroed_check(path, *args, **kwargs) is True
            profile_lifecycle.mark_profile_deleting(profile_dir)
            shutil.rmtree(profile_dir)
            deleted = True
            return True
        return real_zeroed_check(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state, "is_zeroed_state_db", delete_during_zeroed_check)

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        SessionDB(db_path=db_path)

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True


def test_schema_repair_lock_cannot_recreate_profile_deleted_after_precheck(
    home: Path,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    db_path.write_bytes(b"malformed")
    profile_lifecycle.mark_profile_deleting(profile_dir)
    shutil.rmtree(profile_dir)

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        with hermes_state_repair._cross_process_repair_lock(db_path):
            pytest.fail("repair lock must not publish for a deleted profile")

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True


def test_stale_profiles_list_cannot_resurrect_deleted_profile(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "state.db"
    SessionDB(db_path=db_path).close()

    stale_rows = [row for row in profiles.list_profiles() if row.name == "worker"]
    assert len(stale_rows) == 1

    removed = profiles.delete_profile("worker", yes=True)
    assert removed == profile_dir
    assert not profile_dir.exists()

    real_exists = Path.exists

    def stale_state_db_exists(path: Path) -> bool:
        if path == db_path:
            return True
        return real_exists(path)

    # Model a profiles.list worker that observed state.db immediately before the
    # DELETE removed the directory, then resumed with its stale ProfileInfo row.
    monkeypatch.setattr(Path, "exists", stale_state_db_exists)
    monkeypatch.setattr(profiles, "list_profiles", lambda: stale_rows)

    result = srv._methods["profiles.list"]("list", {"include_sessions": True})

    assert "result" in result
    assert not profile_dir.is_dir()


def test_delete_retires_live_session_and_blocks_stale_writable_db_open(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=profile_dir / "state.db").close()

    class Agent:
        closed = False

        def close(self) -> None:
            self.closed = True

    agent = Agent()
    with srv._sessions_lock:
        srv._sessions["worker-live"] = {
            "agent": agent,
            "history": [],
            "profile_home": str(profile_dir),
            "session_key": "",
        }

    profiles.delete_profile("worker", yes=True)

    with srv._sessions_lock:
        assert "worker-live" not in srv._sessions
    assert agent.closed is True
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True
    with srv._session_db({"profile_home": str(profile_dir)}) as db:
        assert db is None
    late_agent = Agent()
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._init_session(
            "late-session",
            "late-session",
            late_agent,
            [],
            profile_home=str(profile_dir),
        )
    assert late_agent.closed is True
    with srv._sessions_lock:
        assert "late-session" not in srv._sessions
    deferred = {"profile_home": str(profile_dir), "session_key": "late-deferred"}
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._claim_or_reuse_live("late-deferred", "late-deferred", deferred, None)
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._methods["session.create"]("rid-late", {"profile": "worker"})
    with pytest.raises(FileNotFoundError, match="being deleted"):
        profiles.write_profile_meta(profile_dir, display_name="stale")
    stale_session = {"profile_home": str(profile_dir)}
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._queue_attached_image(stale_session, b"stale", ".png", prefix="stale")
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._stage_session_file_attachment(
            stale_session, raw_path="", data_url="data:text/plain;base64,c3RhbGU=", name="stale.txt")
    from hermes_cli import clipboard

    monkeypatch.setattr(srv, "_sess_building", lambda _params, _rid: (stale_session, None))
    monkeypatch.setattr(clipboard, "has_clipboard_image", lambda: True)
    monkeypatch.setattr(
        clipboard,
        "save_clipboard_image",
        lambda target: Path(target).write_bytes(b"stale") or True,
    )
    response = srv._methods["clipboard.paste"]("rid-stale", {"session_id": "stale"})
    assert response["error"]["code"] == 4041
    assert not profile_dir.exists()


def test_explicit_recreate_clears_profile_deletion_tombstone(home: Path) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=profile_dir / "state.db").close()
    profiles.delete_profile("worker", yes=True)
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True

    created = profiles.create_profile(
        "worker",
        no_alias=True,
        no_skills=True,
        description="restored profile",
    )

    assert created == profile_dir
    assert profiles.profile_exists("worker") is True
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is False
    assert srv._profile_home_rejected(profile_dir) is False
    assert profiles.read_profile_meta(profile_dir)["description"] == "restored profile"


def test_stale_session_incarnation_cannot_write_into_recreated_profile(
    home: Path,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    old_incarnation = "0" * 32
    (profile_dir / ".profile-incarnation").write_text(
        old_incarnation + "\n",
        encoding="utf-8",
    )
    SessionDB(db_path=profile_dir / "state.db").close()
    stale_session = {
        "attached_images": [],
        "image_counter": 0,
        "profile_home": str(profile_dir),
        "profile_incarnation": old_incarnation,
    }

    profiles.delete_profile("worker", yes=True)
    profiles.create_profile("worker", no_alias=True, no_skills=True)

    new_incarnation = (
        profile_dir.joinpath(".profile-incarnation").read_text(encoding="utf-8").strip()
    )
    assert new_incarnation != old_incarnation

    replacement_db = profile_dir / "state.db"
    assert not replacement_db.exists()
    with pytest.raises(FileNotFoundError, match="incarnation"):
        SessionDB(
            db_path=replacement_db,
            expected_profile_incarnation=old_incarnation,
        )
    assert not replacement_db.exists()
    with pytest.raises(FileNotFoundError, match="incarnation"):
        srv._queue_attached_image(stale_session, b"stale", ".png", prefix="stale")
    assert stale_session["attached_images"] == []
    images_dir = profile_dir / "images"
    assert not images_dir.exists() or list(images_dir.iterdir()) == []

    current_db = SessionDB(
        db_path=replacement_db,
        expected_profile_incarnation=new_incarnation,
    )
    current_db.close()
    assert replacement_db.exists()
    current_session = {
        "attached_images": [],
        "image_counter": 0,
        "profile_home": str(profile_dir),
        "profile_incarnation": new_incarnation,
    }
    current_image = srv._queue_attached_image(
        current_session,
        b"current",
        ".png",
        prefix="current",
    )
    assert current_image.read_bytes() == b"current"
    assert current_session["attached_images"] == [str(current_image)]


def _attempt_cross_process_profile_recreate(home: Path) -> subprocess.CompletedProcess[str]:
    script = (
        "from hermes_cli import profile_lifecycle, profiles; "
        "profile_lifecycle._PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS=0.2; "
        "\ntry:\n profiles.delete_profile('worker', yes=True)\n"
        "except TimeoutError:\n raise SystemExit(0)\n"
        "profile=profiles.create_profile('worker', no_alias=True, no_skills=True); "
        "(profile / 'attachments').mkdir(exist_ok=True); raise SystemExit(1)"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "HERMES_HOME": str(home)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _assert_attachment_write_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = profiles.create_profile("worker", no_alias=True, no_skills=True)
    incarnation = srv._capture_profile_incarnation(profile_dir)
    workspace = home / "workspace"
    workspace.mkdir()
    session = {
        "cwd": str(workspace),
        "profile_home": str(profile_dir),
        "profile_incarnation": incarnation,
        "session_key": "attachment-race",
    }
    entered_write = threading.Event()
    release_write = threading.Event()
    real_write_bytes = Path.write_bytes
    stored: list[Path] = []
    errors: list[BaseException] = []

    def blocking_write_bytes(path: Path, payload: bytes) -> int:
        if path.parent.name == "attachments" and path.parent.parent == profile_dir:
            entered_write.set()
            if not release_write.wait(timeout=5):
                raise TimeoutError("attachment write barrier was not released")
        return real_write_bytes(path, payload)

    monkeypatch.setattr(Path, "write_bytes", blocking_write_bytes)

    def attach() -> None:
        try:
            path, uploaded = srv._stage_session_file_attachment(
                session,
                raw_path="",
                data_url="data:text/plain;base64,c2FmZQ==",
                name="report.txt",
            )
            assert uploaded is True
            stored.append(path)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=attach)
    thread.start()
    assert entered_write.wait(timeout=5)
    try:
        recreate = _attempt_cross_process_profile_recreate(home)
    finally:
        release_write.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert recreate.returncode == 0, recreate.stdout + recreate.stderr
    assert errors == []
    assert len(stored) == 1
    assert stored[0].read_bytes() == b"safe"


def test_attachment_write_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_attachment_write_holds_profile_lifecycle_lease(home, monkeypatch)


@pytest.mark.macos_only
def test_macos_attachment_write_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_attachment_write_holds_profile_lifecycle_lease(home, monkeypatch)


@pytest.mark.windows_only
def test_windows_attachment_write_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_attachment_write_holds_profile_lifecycle_lease(home, monkeypatch)


def _assert_sessiondb_bind_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = profiles.create_profile("worker", no_alias=True, no_skills=True)
    incarnation = srv._capture_profile_incarnation(profile_dir)
    db_path = profile_dir / "state.db"
    entered_connect = threading.Event()
    release_connect = threading.Event()
    real_connect = hermes_state._connect_tracked_db
    opened: list[SessionDB] = []
    errors: list[BaseException] = []

    def blocking_connect(path, *args, **kwargs):
        if Path(path) == db_path and not entered_connect.is_set():
            entered_connect.set()
            if not release_connect.wait(timeout=5):
                raise TimeoutError("SessionDB connect barrier was not released")
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", blocking_connect)

    def open_db() -> None:
        try:
            opened.append(
                SessionDB(
                    db_path=db_path,
                    expected_profile_incarnation=incarnation,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=open_db)
    thread.start()
    assert entered_connect.wait(timeout=5)
    try:
        recreate = _attempt_cross_process_profile_recreate(home)
    finally:
        release_connect.set()
        thread.join(timeout=5)

    try:
        assert not thread.is_alive()
        assert recreate.returncode == 0, recreate.stdout + recreate.stderr
        assert errors == []
        assert len(opened) == 1
        assert opened[0].expected_profile_incarnation == incarnation
    finally:
        for db in opened:
            db.close()


def test_sessiondb_bind_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_sessiondb_bind_holds_profile_lifecycle_lease(home, monkeypatch)


@pytest.mark.macos_only
def test_macos_sessiondb_bind_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_sessiondb_bind_holds_profile_lifecycle_lease(home, monkeypatch)


@pytest.mark.windows_only
def test_windows_sessiondb_bind_holds_profile_lifecycle_lease(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_sessiondb_bind_holds_profile_lifecycle_lease(home, monkeypatch)


def test_compute_host_rejects_old_incarnation_before_agent_build(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    from tui_gateway.compute_host import ComputeHost

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    old_incarnation = "1" * 32
    profile_dir.joinpath(".profile-incarnation").write_text(
        old_incarnation + "\n",
        encoding="utf-8",
    )
    SessionDB(db_path=profile_dir / "state.db").close()
    profiles.delete_profile("worker", yes=True)
    profiles.create_profile("worker", no_alias=True, no_skills=True)

    builds: list[str] = []
    monkeypatch.setattr(
        srv,
        "_make_agent",
        lambda *_args, **_kwargs: builds.append("built"),
    )
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        with pytest.raises(FileNotFoundError, match="incarnation"):
            host._ensure_server_session(
                srv,
                {
                    "sid": "stale-host",
                    "session_key": "stale-host",
                    "profile_home": str(profile_dir),
                    "profile_incarnation": old_incarnation,
                },
            )
        with pytest.raises(FileNotFoundError, match="incarnation"):
            host._ensure_server_session(
                srv,
                {
                    "sid": "unstamped-host",
                    "session_key": "unstamped-host",
                    "profile_home": str(profile_dir),
                },
            )
    finally:
        host.close()

    assert builds == []
    assert "stale-host" not in srv._sessions
    assert "unstamped-host" not in srv._sessions


def test_named_launch_home_sessions_capture_the_launch_incarnation(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = profiles.create_profile("worker", no_alias=True, no_skills=True)
    incarnation = profile_dir.joinpath(".profile-incarnation").read_text(
        encoding="utf-8"
    ).strip()
    monkeypatch.setattr(srv, "_hermes_home", profile_dir)

    record = srv._deferred_session_record(
        "named-launch",
        cols=80,
        cwd=str(home),
        history=[],
        lease=None,
        profile_home=None,
    )

    assert record["profile_home"] is None
    assert record["profile_incarnation"] == incarnation
    assert srv._profile_home_rejected(None, incarnation) is False
    assert srv._profile_home_rejected(None, None) is True


def test_delete_retry_recovers_incarnation_after_partial_rmtree(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = profiles.create_profile("worker", no_alias=True, no_skills=True)
    incarnation = profile_dir.joinpath(".profile-incarnation").read_text(
        encoding="utf-8"
    ).strip()
    real_rmtree = profiles._rmtree_with_retry
    attempts = 0

    def partial_then_remove(path: Path, onerror) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            path.joinpath(".profile-incarnation").unlink()
            raise OSError("injected partial removal")
        real_rmtree(path, onerror)

    monkeypatch.setattr(profiles, "_rmtree_with_retry", partial_then_remove)

    with pytest.raises(RuntimeError, match="Could not remove profile directory"):
        profiles.delete_profile("worker", yes=True)

    tombstone = profile_lifecycle.profile_deletion_marker(profile_dir)
    assert tombstone.read_text(encoding="utf-8").strip() == incarnation
    assert not profile_dir.joinpath(".profile-incarnation").exists()

    profiles.delete_profile("worker", yes=True)

    assert attempts == 2
    assert not profile_dir.exists()


def test_live_session_reuse_is_scoped_to_profile_incarnation(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = profiles.create_profile("worker", no_alias=True, no_skills=True)
    current_incarnation = profile_dir.joinpath(".profile-incarnation").read_text(
        encoding="utf-8"
    ).strip()
    old_record = {
        "session_key": "shared-key",
        "profile_home": str(profile_dir),
        "profile_incarnation": "0" * 32,
    }
    current_record = {
        "session_key": "shared-key",
        "profile_home": str(profile_dir),
        "profile_incarnation": current_incarnation,
        "cwd": str(home),
    }
    monkeypatch.setattr(srv, "_register_session_cwd", lambda _record: None)
    with srv._sessions_lock:
        srv._sessions["old-runtime"] = old_record
    try:
        reused = srv._claim_or_reuse_live(
            "new-runtime",
            "shared-key",
            current_record,
            None,
        )
        assert reused is None
        assert srv._sessions["new-runtime"] is current_record
        assert srv._sessions["old-runtime"] is old_record
    finally:
        with srv._sessions_lock:
            srv._sessions.pop("old-runtime", None)
            srv._sessions.pop("new-runtime", None)


def test_live_session_reuse_skips_stale_incarnation_before_current_winner(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = profiles.create_profile("worker", no_alias=True, no_skills=True)
    current_incarnation = profile_dir.joinpath(".profile-incarnation").read_text(
        encoding="utf-8"
    ).strip()
    old_record = {
        "session_key": "shared-key",
        "profile_home": str(profile_dir),
        "profile_incarnation": "0" * 32,
    }
    current_record = {
        "session_key": "shared-key",
        "profile_home": str(profile_dir),
        "profile_incarnation": current_incarnation,
    }
    fresh_record = {
        "session_key": "shared-key",
        "profile_home": str(profile_dir),
        "profile_incarnation": current_incarnation,
        "cwd": str(home),
    }
    monkeypatch.setattr(srv, "_register_session_cwd", lambda _record: None)
    with srv._sessions_lock:
        # Dict insertion order makes the stale generation the first home/key
        # match. It must not hide the later live runtime for this generation.
        srv._sessions["old-runtime"] = old_record
        srv._sessions["current-runtime"] = current_record
    try:
        reused = srv._claim_or_reuse_live(
            "new-runtime",
            "shared-key",
            fresh_record,
            None,
        )
        assert reused == ("current-runtime", current_record)
        assert "new-runtime" not in srv._sessions
    finally:
        with srv._sessions_lock:
            srv._sessions.pop("old-runtime", None)
            srv._sessions.pop("current-runtime", None)
            srv._sessions.pop("new-runtime", None)


def test_new_profile_stays_unpublished_until_initialization_completes(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    paused = threading.Event()
    release = threading.Event()
    errors = []

    def pause_migration(_profile_dir: Path) -> None:
        paused.set()
        release.wait(timeout=10)

    monkeypatch.setattr(profiles, "_migrate_profile_config_if_outdated", pause_migration)

    def create() -> None:
        try:
            profiles.create_profile(
                "worker",
                no_alias=True,
                no_skills=True,
                description="published only when ready",
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=create)
    thread.start()
    assert paused.wait(timeout=2)

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True
    assert profiles.profile_exists("worker") is False
    assert all(row.name != "worker" for row in profiles.list_profiles())
    assert srv._profile_home_rejected(profile_dir) is True

    release.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert errors == []
    assert profiles.profile_exists("worker") is True
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is False
    assert profiles.read_profile_meta(profile_dir)["description"] == "published only when ready"


@pytest.mark.parametrize("prior_tombstone", [False, True])
def test_failed_profile_initialization_publishes_no_partial_home(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_tombstone: bool,
) -> None:
    profile_dir = home / "profiles" / "worker"
    if prior_tombstone:
        profile_lifecycle.mark_profile_deleting(profile_dir)

    def fail_meta(*_args, **_kwargs):
        raise RuntimeError("simulated metadata failure")

    monkeypatch.setattr(profiles, "write_profile_meta", fail_meta)

    with pytest.raises(RuntimeError, match="metadata failure"):
        profiles.create_profile(
            "worker",
            no_alias=True,
            no_skills=True,
            description="must be atomic",
        )

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is prior_tombstone
    staging_root = home / "profiles" / ".profile-creating"
    assert not staging_root.exists() or list(staging_root.iterdir()) == []


def test_rename_tombstones_old_home_and_publishes_new_home(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_dir = home / "profiles" / "worker"
    new_dir = home / "profiles" / "research"
    old_dir.mkdir(parents=True)
    SessionDB(db_path=old_dir / "state.db").close()

    class Agent:
        closed = False

        def close(self) -> None:
            self.closed = True

    agent = Agent()
    released_memory_homes: list[Path] = []
    from plugins.memory.holographic.store import MemoryStore

    monkeypatch.setattr(
        MemoryStore,
        "release_all_under",
        lambda path: released_memory_homes.append(Path(path)) or 1,
    )
    with srv._sessions_lock:
        srv._sessions["rename-live"] = {
            "agent": agent,
            "history": [],
            "profile_home": str(old_dir),
            "session_key": "",
        }
    profiles.set_active_profile("worker")

    renamed = profiles.rename_profile("worker", "research")

    assert renamed == new_dir
    with srv._sessions_lock:
        assert "rename-live" not in srv._sessions
    assert agent.closed is True
    assert released_memory_homes == [old_dir]
    assert profile_lifecycle.profile_home_is_tombstoned(old_dir) is True
    assert profile_lifecycle.profile_home_is_tombstoned(new_dir) is False
    assert srv._profile_home_rejected(old_dir) is True
    assert srv._profile_home_rejected(new_dir) is False
    assert profiles.profile_exists("worker") is False
    assert profiles.profile_exists("research") is True
    assert profiles.get_active_profile() == "research"
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        SessionDB(db_path=old_dir / "state.db")


def test_unrelated_profiles_named_directory_keeps_normal_sessiondb_semantics(home: Path) -> None:
    db_path = home.parent / "project" / "profiles" / "worker" / "state.db"

    db = SessionDB(db_path=db_path)
    db.close()

    assert db_path.is_file()


def test_named_profile_sessiondb_never_mkdir_after_validation_race(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    db_path = profile_dir / "state.db"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=db_path).close()

    def remove_after_validation(_home: Path) -> bool:
        shutil.rmtree(profile_dir)
        return False

    monkeypatch.setattr(
        hermes_state,
        "named_profile_home_is_unavailable",
        remove_after_validation,
    )

    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        SessionDB(db_path=db_path)
    assert not profile_dir.exists()


def test_named_profile_config_never_mkdir_after_validation_race(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    checked = False

    def remove_after_validation(_profile_home: Path | str) -> bool:
        nonlocal checked
        if not checked:
            checked = True
            shutil.rmtree(profile_dir)
        return False

    monkeypatch.setattr(
        "hermes_constants.named_profile_home_is_unavailable",
        remove_after_validation,
    )
    token = set_hermes_home_override(profile_dir)
    config._HERMES_HOME_ENSURED.pop(str(profile_dir), None)
    try:
        with pytest.raises(FileNotFoundError, match="disappeared during initialization"):
            config.ensure_hermes_home()
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()


def test_config_memo_never_crosses_profile_directory_generation(home: Path) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    token = set_hermes_home_override(profile_dir)
    try:
        config.ensure_hermes_home()
        old_identity = config._HERMES_HOME_ENSURED[str(profile_dir)]
        shutil.rmtree(profile_dir)
        profile_dir.mkdir()

        config.ensure_hermes_home()

        assert config._HERMES_HOME_ENSURED[str(profile_dir)] != old_identity
        assert (profile_dir / "cron").is_dir()
    finally:
        config._HERMES_HOME_ENSURED.pop(str(profile_dir), None)
        reset_hermes_home_override(token)


def test_deletion_tombstone_blocks_stale_sessiondb_open_in_another_process(home: Path) -> None:
    profile_dir = home / "profiles" / "worker"
    db_path = profile_dir / "state.db"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=db_path).close()
    profiles.delete_profile("worker", yes=True)

    script = (
        "import os; from pathlib import Path; from hermes_state import SessionDB; "
        "path=Path(os.environ['TARGET_DB']); "
        "\ntry:\n SessionDB(db_path=path)\nexcept FileNotFoundError:\n raise SystemExit(0)\n"
        "raise SystemExit(1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "HERMES_HOME": str(home), "TARGET_DB": str(db_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not profile_dir.exists()


def test_failed_partial_delete_stays_tombstoned(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=profile_dir / "state.db").close()

    def fail_remove(*_args, **_kwargs) -> None:
        raise PermissionError("simulated locked profile remainder")

    monkeypatch.setattr(profiles, "_rmtree_with_retry", fail_remove)

    with pytest.raises(RuntimeError, match="Could not remove profile directory"):
        profiles.delete_profile("worker", yes=True)

    assert profile_dir.is_dir()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True
    assert profiles.profile_exists("worker") is False
    assert srv._profile_home_rejected(profile_dir) is True
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        srv._profile_home("worker")
    assert "worker" not in profiles.list_profile_names()
    assert all(row.name != "worker" for row in profiles.list_profiles())
    from hermes_cli.web_server_profiles import _fallback_profile_dicts

    assert all(row["name"] != "worker" for row in _fallback_profile_dicts(profiles))
    with pytest.raises(FileNotFoundError):
        profiles.resolve_profile_env("worker")
    token = set_hermes_home_override(profile_dir)
    identity = config._hermes_home_identity(profile_dir, include_ctime=True)
    assert identity is not None
    config._HERMES_HOME_ENSURED[str(profile_dir)] = identity
    try:
        with pytest.raises(FileNotFoundError, match="missing or being deleted"):
            config.ensure_hermes_home()
    finally:
        config._HERMES_HOME_ENSURED.pop(str(profile_dir), None)
        reset_hermes_home_override(token)
    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        SessionDB(db_path=profile_dir / "state.db")


def _assert_profile_mutation_lock_is_cross_process(home: Path) -> None:
    profile_dir = home / "profiles" / "locked"
    script = (
        "from hermes_cli import profile_lifecycle, profiles; "
        "profile_lifecycle._PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS=0.1; "
        "\ntry:\n profiles.create_profile('locked', no_alias=True, no_skills=True)\n"
        "except TimeoutError:\n raise SystemExit(0)\nraise SystemExit(1)"
    )

    with profile_lifecycle._cross_process_profile_mutation_lock():
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "HERMES_HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    assert not profile_dir.exists()
    assert profiles.create_profile("locked", no_alias=True, no_skills=True) == profile_dir


def test_profile_mutation_lock_is_cross_process(home: Path) -> None:
    _assert_profile_mutation_lock_is_cross_process(home)


@pytest.mark.windows_only
def test_windows_profile_mutation_lock_uses_native_byte_range_lock(home: Path) -> None:
    _assert_profile_mutation_lock_is_cross_process(home)


def test_interactive_delete_confirmation_does_not_hold_lifecycle_lock(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)

    class MustNotEnter:
        def __enter__(self):
            raise AssertionError("profile lifecycle lock acquired before confirmation")

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        profile_lifecycle,
        "_cross_process_profile_mutation_lock",
        MustNotEnter,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "cancel")

    assert profiles.delete_profile("worker", yes=False) == profile_dir
    assert profile_dir.is_dir()


def test_delete_confirmation_never_deletes_replacement_generation(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    (profile_dir / "old-generation").write_text("old", encoding="utf-8")

    def replace_then_confirm(_prompt: str) -> str:
        shutil.rmtree(profile_dir)
        profile_dir.mkdir()
        (profile_dir / "replacement-generation").write_text("new", encoding="utf-8")
        return "worker"

    monkeypatch.setattr("builtins.input", replace_then_confirm)

    with pytest.raises(RuntimeError, match="changed while deletion was being confirmed"):
        profiles.delete_profile("worker", yes=False)

    assert (profile_dir / "replacement-generation").read_text(encoding="utf-8") == "new"


def test_untracked_live_sessiondb_makes_delete_fail_closed_until_retry(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db = SessionDB(db_path=profile_dir / "state.db")
    monkeypatch.setattr(
        profile_lifecycle,
        "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS",
        0.1,
    )

    with pytest.raises(RuntimeError, match="still in use"):
        profiles.delete_profile("worker", yes=True)

    assert profile_dir.is_dir()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is False
    assert profiles.profile_exists("worker") is True
    assert srv._profile_home_rejected(profile_dir) is False
    db.close()

    assert profiles.delete_profile("worker", yes=True) == profile_dir
    assert not profile_dir.exists()


def test_external_profile_handle_blocks_rename_until_released(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_dir = home / "profiles" / "worker"
    old_dir.mkdir(parents=True)
    db_path = old_dir / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setattr(
        profile_lifecycle,
        "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS",
        0.1,
    )
    script = (
        "import os,sys; from pathlib import Path; from hermes_state import SessionDB; "
        "db=SessionDB(db_path=Path(os.environ['TARGET_DB'])); "
        "print('READY', flush=True); sys.stdin.readline(); db.close()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env={**os.environ, "HERMES_HOME": str(home), "TARGET_DB": str(db_path)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"

        with pytest.raises(RuntimeError, match="external process"):
            profiles.rename_profile("worker", "research")

        assert old_dir.is_dir()
        assert profile_lifecycle.profile_home_is_tombstoned(old_dir) is False
        assert srv._profile_home_rejected(old_dir) is False
    finally:
        process.communicate("close\n", timeout=10)

    assert profiles.rename_profile("worker", "research") == home / "profiles" / "research"


def test_post_move_rename_failure_still_publishes_new_profile(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_dir = home / "profiles" / "worker"
    new_dir = home / "profiles" / "research"
    old_dir.mkdir(parents=True)

    def fail_finish(*_args) -> None:
        raise RuntimeError("simulated alias update failure")

    monkeypatch.setattr(profiles, "_finish_profile_rename", fail_finish)
    profiles.set_active_profile("worker")

    with pytest.raises(RuntimeError, match="alias update failure"):
        profiles.rename_profile("worker", "research")

    assert not old_dir.exists()
    assert new_dir.is_dir()
    assert profile_lifecycle.profile_home_is_tombstoned(new_dir) is False
    assert profiles.profile_exists("research") is True
    assert srv._profile_home_rejected(new_dir) is False
    assert profiles.get_active_profile() == "research"


def test_failed_rename_preserves_prior_destination_tombstone(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_dir = home / "profiles" / "worker"
    new_dir = home / "profiles" / "research"
    old_dir.mkdir(parents=True)
    profile_lifecycle.mark_profile_deleting(new_dir)

    original_rename = Path.rename

    def fail_target_rename(path: Path, target: Path) -> Path:
        if path == old_dir and target == new_dir:
            raise OSError("simulated rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_target_rename)

    with pytest.raises(OSError, match="rename failure"):
        profiles.rename_profile("worker", "research")

    assert old_dir.is_dir()
    assert not new_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(new_dir) is True


def test_concurrent_profile_use_cannot_restore_retired_name(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_dir = home / "profiles" / "worker"
    old_dir.mkdir(parents=True)
    profiles.set_active_profile("worker")
    finishing = threading.Event()
    release = threading.Event()
    rename_errors: list[Exception] = []
    use_errors: list[Exception] = []

    def block_finish(*_args) -> None:
        finishing.set()
        release.wait(timeout=10)

    monkeypatch.setattr(profiles, "_finish_profile_rename", block_finish)

    def rename() -> None:
        try:
            profiles.rename_profile("worker", "research")
        except Exception as exc:
            rename_errors.append(exc)

    def reuse_old_name() -> None:
        try:
            profiles.set_active_profile("worker")
        except Exception as exc:
            use_errors.append(exc)

    rename_thread = threading.Thread(target=rename)
    rename_thread.start()
    assert finishing.wait(timeout=2)
    use_thread = threading.Thread(target=reuse_old_name)
    use_thread.start()
    release.set()
    rename_thread.join(timeout=3)
    use_thread.join(timeout=3)

    assert not rename_errors
    assert len(use_errors) == 1
    assert isinstance(use_errors[0], FileNotFoundError)
    assert profiles.get_active_profile() == "research"


def test_failed_reimport_preserves_prior_deletion_tombstone(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_lifecycle.mark_profile_deleting(profile_dir)
    srv.retire_profile_home(profile_dir)

    def fail_extract(*_args) -> None:
        raise ValueError("simulated invalid archive")

    monkeypatch.setattr(
        profiles,
        "safe_extract_targz",
        fail_extract,
    )

    with pytest.raises(ValueError, match="invalid archive"):
        profiles._import_profile_into_home(
            home / "unused.tar.gz",
            "worker",
            "worker",
            profile_dir,
        )

    assert not profile_dir.exists()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True
    assert srv._profile_home_rejected(profile_dir) is True


def test_failed_fresh_import_hides_partial_destination(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"

    def fake_extract(_archive: Path, staging_root: Path) -> None:
        extracted = staging_root / "worker"
        extracted.mkdir()
        (extracted / "config.yaml").write_text("model: test\n", encoding="utf-8")

    def partial_move(_source: str, destination: str) -> None:
        partial = Path(destination)
        partial.mkdir()
        (partial / "config.yaml").write_text("partial", encoding="utf-8")
        raise OSError("simulated cross-filesystem copy failure")

    monkeypatch.setattr(profiles, "safe_extract_targz", fake_extract)
    monkeypatch.setattr(shutil, "move", partial_move)

    with pytest.raises(OSError, match="copy failure"):
        profiles._import_profile_into_home(
            home / "unused.tar.gz",
            "worker",
            "worker",
            profile_dir,
        )

    assert profile_dir.is_dir()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True
    assert profiles.profile_exists("worker") is False


def test_delete_waits_for_active_turn_before_removing_profile(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    release = threading.Event()
    started = threading.Event()

    def active_turn() -> None:
        started.set()
        release.wait(timeout=10)

    thread = threading.Thread(target=active_turn)
    thread.start()
    assert started.wait(timeout=2)

    class Agent:
        def close(self) -> None:
            return None

    with srv._sessions_lock:
        srv._sessions["active-turn"] = {
            "_run_thread": thread,
            "agent": Agent(),
            "history": [],
            "profile_home": str(profile_dir),
            "session_key": "",
        }
    monkeypatch.setattr(srv, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="active session turn"):
        profiles.delete_profile("worker", yes=True)

    assert profile_dir.is_dir()
    assert profile_lifecycle.profile_home_is_tombstoned(profile_dir) is True
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert profiles.delete_profile("worker", yes=True) == profile_dir


def test_delete_fences_and_closes_deferred_agent_build(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=profile_dir / "state.db").close()
    started = threading.Event()
    release = threading.Event()

    class Agent:
        closed = False

        def __init__(self, session_db) -> None:
            self._session_db = session_db
            self._owns_session_db = False

        def close(self) -> None:
            self.closed = True

    built = []
    notify_registrations = []

    def make_agent(_sid, _key, **kwargs):
        agent = Agent(kwargs.get("session_db"))
        built.append(agent)
        started.set()
        release.wait(timeout=10)
        return agent

    monkeypatch.setattr(srv, "_make_agent", make_agent)
    monkeypatch.setattr(
        "tools.approval.register_gateway_notify",
        lambda *args, **kwargs: notify_registrations.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "tui_gateway.entry.ensure_mcp_discovery_started",
        lambda: None,
    )
    monkeypatch.setattr(srv, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.0)
    session = {
        "agent": None,
        "agent_ready": threading.Event(),
        "history": [],
        "profile_home": str(profile_dir),
        "profile_incarnation": srv._capture_profile_incarnation(profile_dir),
        "session_key": "deferred-build",
    }
    with srv._sessions_lock:
        srv._sessions["deferred-build"] = session
    srv._start_agent_build("deferred-build", session)
    assert started.wait(timeout=2)

    with pytest.raises(RuntimeError, match="active session turn"):
        profiles.delete_profile("worker", yes=True)

    assert profile_dir.is_dir()
    release.set()
    build_thread = session["_agent_build_thread"]
    build_thread.join(timeout=3)
    assert not build_thread.is_alive()
    assert built and built[0].closed is True
    assert notify_registrations == []
    assert "deferred-build" not in srv._sessions
    assert profiles.delete_profile("worker", yes=True) == profile_dir
