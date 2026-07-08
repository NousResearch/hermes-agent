import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tui_gateway import server


class LiveTransport:
    _closed = False

    def __init__(self):
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True


@pytest.fixture(autouse=True)
def _clean_server_state(monkeypatch):
    server._sessions.clear()
    server._desktop_session_initiated_restart.clear()
    server._compute_host_inflight_turns.clear()
    server._desktop_auto_resume_signal_handlers_installed = False
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"desktop_auto_resume": True}})
    yield
    server._sessions.clear()
    server._desktop_session_initiated_restart.clear()
    server._compute_host_inflight_turns.clear()


def _create_db(tmp_path: Path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("desktop-session", source="desktop", model="test-model")
    return db


def _live_session(session_key="desktop-session", *, running=True, transport=None):
    return {
        "session_key": session_key,
        "history": [{"role": "user", "content": "deploy it"}],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": running,
        "transport": transport if transport is not None else LiveTransport(),
        "inflight_turn": {"user": "deploy it", "assistant": "", "streaming": True},
        "created_at": time.time(),
        "last_active": time.time(),
    }


def test_default_config_and_read_site_keep_desktop_auto_resume_dormant(monkeypatch):
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["dashboard"]["desktop_auto_resume"] is False

    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    cfg = server._load_desktop_auto_resume_config()
    assert cfg["enabled"] is False


def test_sessiondb_persists_marker_breaker_and_monotonic_consumed_reason(tmp_path):
    db = _create_db(tmp_path)
    now = 1000.0

    db.upsert_desktop_resume_marker(
        "desktop-session",
        reason=server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED,
        prompt="continue",
        created_at=now,
        ttl_seconds=3600,
        reason_priority=server.DESKTOP_REASON_PRIORITIES[
            server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
        ],
    )
    db.upsert_desktop_resume_marker(
        "desktop-session",
        reason=server.DESKTOP_REASON_RESTART_CONSUMED,
        prompt="do not continue",
        created_at=now + 1,
        ttl_seconds=3600,
        reason_priority=server.DESKTOP_REASON_PRIORITIES[
            server.DESKTOP_REASON_RESTART_CONSUMED
        ],
    )
    db.upsert_desktop_resume_marker(
        "desktop-session",
        reason=server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED,
        prompt="lower priority must not win",
        created_at=now + 2,
        ttl_seconds=3600,
        reason_priority=server.DESKTOP_REASON_PRIORITIES[
            server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
        ],
    )
    db.close()

    reopened = SessionDB(db_path=tmp_path / "state.db")
    marker = reopened.get_desktop_resume_marker("desktop-session")
    assert marker["reason"] == server.DESKTOP_REASON_RESTART_CONSUMED
    assert marker["prompt"] == "do not continue"

    claimed = reopened.claim_desktop_auto_resume(
        "desktop-session",
        marker_created_at=marker["created_at"],
        now=now + 3,
        freshness_window_seconds=3600,
        replay_window_seconds=300,
        replay_threshold=3,
    )
    assert claimed["claimed"] is False
    assert claimed["reason"] == server.DESKTOP_REASON_RESTART_CONSUMED


def test_shutdown_marker_write_is_default_off_and_requires_live_transport(monkeypatch, tmp_path):
    db = _create_db(tmp_path)
    server._sessions["live"] = _live_session()
    monkeypatch.setattr(server, "_session_db", lambda _session: _DbContext(db))
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    assert server._write_desktop_resume_markers_for_shutdown("sigterm") == 0
    assert db.get_desktop_resume_marker("desktop-session") is None

    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"desktop_auto_resume": True}})
    server._sessions["live"]["transport"] = server._detached_ws_transport
    assert server._write_desktop_resume_markers_for_shutdown("sigterm") == 0
    assert db.get_desktop_resume_marker("desktop-session") is None


class _DbContext:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, *_args):
        return False


def test_shutdown_marker_write_covers_running_and_compute_host_turns(monkeypatch, tmp_path):
    db = _create_db(tmp_path)
    db.create_session("isolated-session", source="desktop", model="test-model")
    live = _live_session("desktop-session", running=True)
    isolated = _live_session("isolated-session", running=False)
    server._sessions["live"] = live
    server._sessions["isolated"] = isolated
    server._compute_host_inflight_turns["isolated"] = {
        "session_key": "isolated-session",
        "text": "isolated prompt",
    }
    monkeypatch.setattr(server, "_session_db", lambda _session: _DbContext(db))

    assert server._write_desktop_resume_markers_for_shutdown("sigterm") == 2
    live_marker = db.get_desktop_resume_marker("desktop-session")
    assert live_marker["reason"] == server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
    assert "Do NOT blindly re-execute old tool calls" in live_marker["prompt"]
    assert "deploy it" in live_marker["prompt"]
    isolated_marker = db.get_desktop_resume_marker("isolated-session")
    assert isolated_marker["reason"] == server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
    assert "isolated prompt" in isolated_marker["prompt"]


def test_tool_start_marks_self_restart_but_not_safe_restart_inspection():
    server._sessions["sid"] = _live_session()

    server._on_tool_start(
        "sid",
        "tc-read",
        "terminal",
        {"command": "cat ~/.hermes/skills/safe-gateway-restart/scripts/safe-restart.py"},
    )
    assert "sid" not in server._desktop_session_initiated_restart

    server._on_tool_start(
        "sid",
        "tc-run",
        "terminal",
        {"command": "python ~/.hermes/skills/safe-gateway-restart/scripts/safe-restart.py"},
    )
    assert "sid" in server._desktop_session_initiated_restart


def test_self_restart_reason_is_excluded_from_continuation(monkeypatch, tmp_path):
    db = _create_db(tmp_path)
    server._sessions["sid"] = _live_session(running=True)
    server._desktop_session_initiated_restart.add("sid")
    monkeypatch.setattr(server, "_session_db", lambda _session: _DbContext(db))

    assert server._write_desktop_resume_markers_for_shutdown("sigterm") == 1
    marker = db.get_desktop_resume_marker("desktop-session")
    assert marker["reason"] == server.DESKTOP_REASON_RESTART_CONSUMED_INTERRUPTED

    started = []
    monkeypatch.setattr(
        server,
        "_start_desktop_auto_resume_turn",
        lambda *args, **kwargs: started.append(args) or True,
    )
    payload = {"session_id": "sid", "session_key": "desktop-session", "status": "idle"}
    server._sessions["sid"]["running"] = False

    updated = server._maybe_trigger_desktop_auto_resume_after_resume(
        "rid", "desktop-session", "sid", server._sessions["sid"], payload, db
    )
    assert started == []
    assert updated["desktop_auto_resume"]["status"] == "excluded"


def test_external_marker_claim_launches_once_and_clears_atomically(monkeypatch, tmp_path):
    db = _create_db(tmp_path)
    session = _live_session(running=False)
    server._sessions["sid"] = session
    db.upsert_desktop_resume_marker(
        "desktop-session",
        reason=server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED,
        prompt="continue safely",
        created_at=time.time(),
        ttl_seconds=3600,
        reason_priority=server.DESKTOP_REASON_PRIORITIES[
            server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
        ],
    )
    started = []
    monkeypatch.setattr(
        server,
        "_start_desktop_auto_resume_turn",
        lambda _rid, _sid, _session, prompt: started.append(prompt) or True,
    )

    payload = {"session_id": "sid", "session_key": "desktop-session", "status": "idle"}
    first = server._maybe_trigger_desktop_auto_resume_after_resume(
        "rid", "desktop-session", "sid", session, dict(payload), db
    )
    second = server._maybe_trigger_desktop_auto_resume_after_resume(
        "rid", "desktop-session", "sid", session, dict(payload), db
    )

    assert started == ["continue safely"]
    assert first["desktop_auto_resume"]["status"] == "started"
    assert "desktop_auto_resume" not in second
    assert db.get_desktop_resume_marker("desktop-session") is None


def test_stale_marker_surfaces_signal_without_silent_continuation(monkeypatch, tmp_path):
    db = _create_db(tmp_path)
    session = _live_session(running=False)
    server._sessions["sid"] = session
    db.upsert_desktop_resume_marker(
        "desktop-session",
        reason=server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED,
        prompt="continue",
        created_at=10.0,
        ttl_seconds=48 * 3600,
        reason_priority=server.DESKTOP_REASON_PRIORITIES[
            server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
        ],
    )
    monkeypatch.setattr(server, "_desktop_auto_resume_now", lambda: 10_000.0)
    started = []
    monkeypatch.setattr(
        server,
        "_start_desktop_auto_resume_turn",
        lambda *args, **kwargs: started.append(args) or True,
    )

    payload = {"session_id": "sid", "session_key": "desktop-session", "status": "idle"}
    updated = server._maybe_trigger_desktop_auto_resume_after_resume(
        "rid", "desktop-session", "sid", session, payload, db
    )

    assert started == []
    assert updated["desktop_auto_resume"]["status"] == "stale_marker_present"
    assert db.get_desktop_resume_marker("desktop-session") is not None


def test_state_db_unavailable_disables_auto_continue_without_crash(monkeypatch):
    session = _live_session(running=False)
    started = []
    monkeypatch.setattr(
        server,
        "_start_desktop_auto_resume_turn",
        lambda *args, **kwargs: started.append(args) or True,
    )

    payload = {"session_id": "sid", "session_key": "desktop-session", "status": "idle"}
    assert server._maybe_trigger_desktop_auto_resume_after_resume(
        "rid", "desktop-session", "sid", session, payload, None
    ) == payload
    assert started == []


def test_breaker_survives_real_process_respawns(tmp_path):
    db_path = tmp_path / "state.db"
    setup_db = SessionDB(db_path=db_path)
    setup_db.create_session("desktop-session", source="desktop", model="test-model")
    setup_db.close()

    script = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path
        from hermes_state import SessionDB
        from tui_gateway import server

        db = SessionDB(db_path=Path(sys.argv[1]))
        now = float(sys.argv[2])
        db.upsert_desktop_resume_marker(
            'desktop-session',
            reason=server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED,
            prompt='continue',
            created_at=now,
            ttl_seconds=3600,
            reason_priority=server.DESKTOP_REASON_PRIORITIES[
                server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
            ],
        )
        marker = db.get_desktop_resume_marker('desktop-session')
        result = db.claim_desktop_auto_resume(
            'desktop-session',
            marker_created_at=marker['created_at'],
            now=now,
            freshness_window_seconds=3600,
            replay_window_seconds=300,
            replay_threshold=3,
        )
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    results = []
    for i in range(4):
        proc = subprocess.run(
            [sys.executable, "-c", script, str(db_path), str(1000 + i)],
            text=True,
            capture_output=True,
            check=True,
            env=env,
            timeout=30,
        )
        results.append((proc.stdout + proc.stderr).strip())

    assert '"claimed": true' in results[0]
    assert '"claimed": true' in results[1]
    assert '"claimed": true' in results[2]
    assert '"claimed": false' in results[3]
    assert '"suspended": true' in results[3]


@pytest.mark.parametrize(
    ("signum", "expect_marker"),
    [(signal.SIGTERM, True), (signal.SIGKILL, False)],
)
def test_signal_shutdown_writes_marker_but_sigkill_cannot(tmp_path, signum, expect_marker):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("POSIX signals required")

    home = tmp_path / ("home-term" if expect_marker else "home-kill")
    home.mkdir()
    script = "\n".join(
        [
            "import os",
            "import sys",
            "import threading",
            "import time",
            "from pathlib import Path",
            "home = Path(os.environ['HERMES_HOME'])",
            "(home / 'config.yaml').write_text('dashboard:\\n  desktop_auto_resume: true\\n', encoding='utf-8')",
            "from hermes_state import SessionDB",
            "from tui_gateway import server",
            "server._cfg_cache = None",
            "server._cfg_mtime = None",
            "server._cfg_path = None",
            "db = SessionDB()",
            "db.create_session('desktop-session', source='desktop', model='test-model')",
            "class LiveTransport:",
            "    _closed = False",
            "    def write(self, _obj):",
            "        return True",
            "server._sessions['sid'] = {",
            "    'session_key': 'desktop-session',",
            "    'history': [{'role': 'user', 'content': 'keep going'}],",
            "    'history_lock': threading.Lock(),",
            "    'history_version': 0,",
            "    'running': True,",
            "    'transport': LiveTransport(),",
            "    'inflight_turn': {'user': 'keep going', 'assistant': '', 'streaming': True},",
            "    'created_at': time.time(),",
            "    'last_active': time.time(),",
            "}",
            "server._install_desktop_auto_resume_signal_handlers()",
            "print('READY', file=sys.stderr, flush=True)",
            "time.sleep(60)",
        ]
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 20
        line = ""
        while time.monotonic() < deadline:
            line = proc.stderr.readline()
            if "READY" in line:
                break
        if "READY" not in line:
            stdout, stderr = proc.communicate(timeout=5)
            raise AssertionError(
                f"child did not become ready; stdout={stdout!r} stderr={stderr!r}"
            )
        os.kill(proc.pid, signum)
        proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    db = SessionDB(db_path=home / "state.db")
    marker = db.get_desktop_resume_marker("desktop-session")
    assert (marker is not None) is expect_marker


def test_get_db_boot_sweep_purges_expired_markers_and_breakers(monkeypatch, tmp_path):
    """AC-10 (RC-5): the TTL sweep must actually FIRE on first DB open — the
    method exists but is inert unless wired into the boot maintenance window.
    Seed an already-expired marker + an aged-out breaker, reset the _get_db
    singleton pointed at this db, and assert _get_db() purges them on open.
    Mutation-proof: remove the sweep call in _get_db → expired rows survive → RED.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("stale-desktop", source="desktop", model="test-model")
    now = 10_000.0
    # An already-EXPIRED marker (expires_at in the past) — the stale/never-resumed
    # path that would otherwise accrete forever. No claim: a claim would consume
    # the marker, and we want the SWEEP to be what removes it.
    seed.upsert_desktop_resume_marker(
        "stale-desktop",
        reason=server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED,
        prompt="continue",
        created_at=now - 100_000.0,
        ttl_seconds=1.0,  # expires_at ~= now - 100000, already past
        reason_priority=server.DESKTOP_REASON_PRIORITIES[
            server.DESKTOP_REASON_EXTERNAL_RESTART_INTERRUPTED
        ],
    )
    # An aged-out breaker row (updated far beyond the 48h TTL) on a DIFFERENT
    # session so it doesn't touch the marker above. Seed it directly since a live
    # claim would stamp updated_at=now (not aged). The breaker table FKs to
    # sessions(id), so the session row must exist first.
    seed.create_session("aged-breaker", source="desktop", model="test-model")
    seed._execute_write(
        lambda conn: conn.execute(
            "INSERT INTO desktop_resume_breakers "
            "(session_id, window_started_at, replay_count, updated_at) VALUES (?, ?, ?, ?)",
            ("aged-breaker", now - 1_000_000.0, 2, now - 1_000_000.0),
        )
    )
    seed.close()

    # Sanity: BEFORE the boot sweep, both rows are present (get_* / a direct read
    # do not filter on expiry — only the sweep DELETEs them).
    pre = SessionDB(db_path=db_path)
    assert pre.get_desktop_resume_marker("stale-desktop") is not None
    assert (
        pre._conn.execute(
            "SELECT COUNT(*) FROM desktop_resume_breakers WHERE session_id = ?",
            ("aged-breaker",),
        ).fetchone()[0]
        == 1
    )
    pre.close()

    # Reset the _get_db singleton and point SessionDB() at our seeded db.
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_db_error", None, raising=False)
    import hermes_state as _hs
    monkeypatch.setattr(_hs, "SessionDB", lambda *a, **k: SessionDB(db_path=db_path))

    db = server._get_db()
    assert db is not None
    # Boot sweep fired → the expired marker AND the aged breaker are gone.
    assert db.get_desktop_resume_marker("stale-desktop") is None
    assert (
        db._conn.execute(
            "SELECT COUNT(*) FROM desktop_resume_breakers WHERE session_id = ?",
            ("aged-breaker",),
        ).fetchone()[0]
        == 0
    )
    db.close()
    monkeypatch.setattr(server, "_db", None, raising=False)
