"""Gateway sessions must persist their per-command cwd to ``state.db``.

The Projects sidebar groups ``sessions`` rows by ``cwd``/``git_repo_root``.
CLI/TUI persisted it; messaging-gateway chats (``agent:…`` session keys)
only tracked cwd in memory, so every Telegram/Discord/Slack chat grouped
under the launch dir forever (#29531).

Ownership is fenced to the EXACT durable ``HERMES_SESSION_ID`` that
commissioned the terminal effect — never re-resolved from the stable
``session_key`` after the effect completed (a reset reuses the key across
durable session ids, so a stale effect must not re-home its successor).

These tests drive the REAL ``tt.terminal_tool`` against a temp
``HERMES_HOME`` — never the user's database.
"""

import json
import os
import subprocess

import pytest

import tools.terminal_tool as tt
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB
from tools.environments.local import LocalEnvironment

GATEWAY_KEY = "agent:main:telegram:group:-100:alice"


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch, tmp_path):
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    # Route every DB write in this test into a throwaway HERMES_HOME.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def env(tmp_path):
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    environment.init_session()
    yield environment
    environment.cleanup()


def _db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _seed_session(tmp_path, session_id, session_key):
    """Mint the gateway row the way ``async_session_store`` would."""
    db = _db(tmp_path)
    db.create_session(session_id, "telegram", session_key=session_key)
    db.close()


def _row(tmp_path, session_id):
    db = _db(tmp_path)
    try:
        return db.get_session(session_id)
    finally:
        db.close()


class ToolDriver:
    """Drive ``tt.terminal_tool`` as the gateway would: with the session
    context (routing key + durable session id) bound around the call."""

    def __init__(self, monkeypatch, env):
        monkeypatch.setattr(tt, "_active_environments", {"default": env})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": env.cwd, "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        self._env = env

    def run(self, command, task_id, timeout=None, workdir=None, session_id=None):
        set_session_vars(
            platform="telegram",
            session_key=task_id,
            session_id=session_id or task_id,
        )
        try:
            return json.loads(
                tt.terminal_tool(
                    command=command, task_id=task_id, timeout=timeout, workdir=workdir
                )
            )
        finally:
            clear_session_vars([])


@pytest.fixture
def driver(monkeypatch, env):
    return ToolDriver(monkeypatch, env)


def _git_repo(path):
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "feature/pet", str(path)], check=True)
    # A commit is needed for HEAD to resolve; an unborn branch fails the probe.
    (path / ".gitkeep").write_text("")
    subprocess.run(["git", "-C", str(path), "add", ".gitkeep"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"],
        check=True,
    )
    return path


class TestGatewaySessionsPersistCwd:
    def test_happy_path_persists_cwd_and_git_meta(self, driver, tmp_path):
        repo = _git_repo(tmp_path / "projects" / "pet-app")
        _seed_session(tmp_path, "20260828_100000_abc12345", GATEWAY_KEY)

        result = driver.run(f"cd {repo} && pwd", GATEWAY_KEY,
                            session_id="20260828_100000_abc12345")
        assert result["exit_code"] == 0

        row = _row(tmp_path, "20260828_100000_abc12345")
        assert os.path.realpath(row["cwd"]) == os.path.realpath(str(repo))
        assert os.path.realpath(row["git_repo_root"]) == os.path.realpath(str(repo))
        assert row["git_branch"] == "feature/pet"
        assert row["git_metadata_generation"] == 1

    def test_short_gateway_key_persists_too(self, driver, tmp_path):
        """``agent:main`` style keys are gateway keys — pin the scope guard."""
        _seed_session(tmp_path, "row-short", "agent:main")

        result = driver.run("pwd", "agent:main", session_id="row-short")
        assert result["exit_code"] == 0
        assert _row(tmp_path, "row-short")["cwd"] is not None


class TestStaleCompletionFencing:
    """P1: the write targets the exact durable session, never the key's
    currently-open row. A stale effect from a reset conversation must be a
    no-op for its successor."""

    def test_stale_effect_does_not_rehome_successor(self, driver, tmp_path):
        repo_old = _git_repo(tmp_path / "projects" / "old-work")
        repo_new = _git_repo(tmp_path / "projects" / "new-work")

        # S1 under K finishes a command in repo_old; its persistence is
        # "in flight" (the effect commissioned by S1).
        _seed_session(tmp_path, "row-old", GATEWAY_KEY)
        driver.run(f"cd {repo_old} && pwd", GATEWAY_KEY, session_id="row-old")
        assert os.path.realpath(
            _row(tmp_path, "row-old")["cwd"]
        ) == os.path.realpath(str(repo_old))

        # Reset: /new closes S1 and mints S2 under the SAME key K.
        db = _db(tmp_path)
        db.end_session("row-old", "reset")
        db.create_session("row-new", "telegram", session_key=GATEWAY_KEY)
        db.close()

        # The STALE S1 completion persists now, after the reset. It still
        # runs inside S1's task context (that's what produced the effect),
        # so the write must land on S1 (or be dropped), never re-home S2.
        set_session_vars(
            platform="telegram", session_key=GATEWAY_KEY, session_id="row-old"
        )
        try:
            tt.persist_gateway_session_cwd_to_db(GATEWAY_KEY, str(repo_new))
        finally:
            clear_session_vars([])
        assert _row(tmp_path, "row-new")["cwd"] is None
        assert os.path.realpath(
            _row(tmp_path, "row-old")["cwd"]
        ) == os.path.realpath(str(repo_new))

        # And the stale write must still not be able to touch S2 even when
        # it carries a fresh cwd: ownership is by id, not by key.
        result = driver.run(f"cd {repo_new} && pwd", GATEWAY_KEY,
                            session_id="row-new")
        assert result["exit_code"] == 0
        assert os.path.realpath(
            _row(tmp_path, "row-new")["cwd"]
        ) == os.path.realpath(str(repo_new))


class TestSkips:
    def test_workdir_override_does_not_persist(self, driver, tmp_path):
        _seed_session(tmp_path, "row-workdir", GATEWAY_KEY)
        transient = tmp_path / "transient"
        transient.mkdir()

        result = driver.run("pwd", GATEWAY_KEY, workdir=str(transient),
                            session_id="row-workdir")
        assert result["exit_code"] == 0
        assert _row(tmp_path, "row-workdir")["cwd"] is None

    def test_interrupted_command_does_not_persist(self, driver, tmp_path):
        _seed_session(tmp_path, "row-interrupted", GATEWAY_KEY)

        result = driver.run("sleep 20", GATEWAY_KEY, timeout=2,
                            session_id="row-interrupted")
        # Killed before the cwd marker: no observation, no persistence.
        assert result["exit_code"] != 0
        assert _row(tmp_path, "row-interrupted")["cwd"] is None

    def test_cli_session_key_does_not_persist(self, driver, tmp_path):
        result = driver.run("pwd", "cli:default", session_id="row-cli")
        assert result["exit_code"] == 0
        # The scope guard declined before the write: no DB was ever opened.
        assert not (tmp_path / "state.db").exists()

    def test_tui_session_key_does_not_persist(self, driver, tmp_path):
        result = driver.run("pwd", "tui:abc123", session_id="row-tui")
        assert result["exit_code"] == 0
        assert not (tmp_path / "state.db").exists()


class TestSharedEnvSafety:
    def test_concurrent_sessions_stamp_only_their_own_cwd(self, driver, tmp_path):
        """Two chats sharing one env each persist only where THEY finished."""
        repo_a = _git_repo(tmp_path / "projects" / "alpha")
        repo_b = _git_repo(tmp_path / "projects" / "beta")
        key_a = "agent:main:telegram:group:-100:alice"
        key_b = "agent:main:discord:dm:bob"
        _seed_session(tmp_path, "row-a", key_a)
        _seed_session(tmp_path, "row-b", key_b)

        driver.run(f"cd {repo_a} && pwd", key_a, session_id="row-a")
        driver.run(f"cd {repo_b} && pwd", key_b, session_id="row-b")

        row_a = _row(tmp_path, "row-a")
        row_b = _row(tmp_path, "row-b")
        assert os.path.realpath(row_a["cwd"]) == os.path.realpath(str(repo_a))
        assert os.path.realpath(row_b["cwd"]) == os.path.realpath(str(repo_b))
        assert os.path.realpath(row_a["git_repo_root"]) == os.path.realpath(str(repo_a))
        assert os.path.realpath(row_b["git_repo_root"]) == os.path.realpath(str(repo_b))


class TestDegradedInputs:
    def test_non_git_cwd_persists_cwd_with_null_git_meta(self, driver, tmp_path):
        plain = tmp_path / "not-a-repo"
        plain.mkdir()
        _seed_session(tmp_path, "row-plain", GATEWAY_KEY)

        result = driver.run(f"cd {plain} && pwd", GATEWAY_KEY,
                            session_id="row-plain")
        assert result["exit_code"] == 0

        row = _row(tmp_path, "row-plain")
        assert os.path.realpath(row["cwd"]) == os.path.realpath(str(plain))
        assert row["git_branch"] is None
        assert row["git_repo_root"] is None

    def test_db_failure_does_not_break_the_command(self, driver, tmp_path, monkeypatch):
        """A locked/corrupt DB must never fail the user's command."""
        _seed_session(tmp_path, "row-broken", GATEWAY_KEY)

        def _boom(*args, **kwargs):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(SessionDB, "update_session_cwd", _boom)

        result = driver.run("echo alive", GATEWAY_KEY, session_id="row-broken")
        assert result["exit_code"] == 0
        assert "alive" in result["output"]
