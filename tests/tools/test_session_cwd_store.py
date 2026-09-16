"""Session-cwd record store (cwd rearchitecture, step 1: dual-write).

The store is the future single source of truth for per-session working
directories. Step 1 only guarantees the WRITE side: every path that learns a
session's live cwd must record it under the raw session key. Readers still
use the legacy env.cwd ladder; these tests pin the invariants the later
read-side flip will rely on.
"""

import pytest

import tools.terminal_tool as tt


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch):
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})


class TestRecordSemantics:
    def test_records_are_keyed_by_raw_session_key(self):
        tt.record_session_cwd("sess-a", "/wt/a")
        tt.record_session_cwd("sess-b", "/wt/b")
        # No cross-talk: each session reads back exactly its own record.
        assert tt.get_session_cwd("sess-a") == "/wt/a"
        assert tt.get_session_cwd("sess-b") == "/wt/b"
        assert tt.get_session_cwd("sess-c") is None


    def test_clear_drops_only_the_named_session(self):
        tt.record_session_cwd("sess-a", "/wt/a")
        tt.record_session_cwd("sess-b", "/wt/b")
        tt.clear_session_cwd("sess-a")
        assert tt.get_session_cwd("sess-a") is None
        assert tt.get_session_cwd("sess-b") == "/wt/b"


class TestDualWriteSites:
    def test_register_cwd_override_seeds_the_session_record(self):
        """A registered workspace cwd IS the session's cwd until a `cd`."""
        tt.register_task_env_overrides("desktop-sess", {"cwd": "/wt/desktop"})
        assert tt.get_session_cwd("desktop-sess") == "/wt/desktop"


    def test_reregistration_updates_the_record(self):
        """ACP session/load switching project roots mid-session."""
        tt.register_task_env_overrides("acp-sess", {"cwd": "/proj/one"})
        tt.register_task_env_overrides("acp-sess", {"cwd": "/proj/two"})
        assert tt.get_session_cwd("acp-sess") == "/proj/two"


class TestPostCommandDualWrite:
    """The env's post-command cwd tracking must mirror into the session record."""

    def _run(self, monkeypatch, task_id, env):
        import json
        monkeypatch.setattr(tt, "_active_environments", {task_id: env})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": "/default", "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        return json.loads(tt.terminal_tool(command="cd /new/dir", task_id=task_id))

    def test_cd_result_is_recorded_under_the_session_key(self, monkeypatch):
        class FakeEnv:
            env = {}
            cwd = "/start"
            def execute(self, command, **kwargs):
                # Simulate the env's own post-command tracking (marker parse):
                # the marker is what moves cwd AND what flags the observation.
                self.cwd = "/new/dir"
                return {"output": "", "returncode": 0, "cwd_observed": True}

        result = self._run(monkeypatch, "sess-a", FakeEnv())
        assert result["exit_code"] == 0
        assert tt.get_session_cwd("sess-a") == "/new/dir"
        # And ONLY that session's record was touched.
        assert tt.get_session_cwd("sess-b") is None

    def test_envs_without_cwd_tracking_record_nothing(self, monkeypatch):
        class FakeEnv:
            env = {}
            def execute(self, command, **kwargs):
                return {"output": "", "returncode": 0}

        result = self._run(monkeypatch, "sess-a", FakeEnv())
        assert result["exit_code"] == 0
        assert tt.get_session_cwd("sess-a") is None


class TestFileToolsReadTheRecord:
    """Step 2: file-tool path resolution prefers the session's own record."""

    def test_two_sessions_resolve_into_their_own_recorded_cwds(self, tmp_path, monkeypatch):
        import tools.file_tools as ft
        import tools.file_tools_paths as ftp

        wt_a = tmp_path / "wt_a"
        wt_b = tmp_path / "wt_b"
        for d in (wt_a, wt_b):
            d.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(ft, "_file_ops_cache", {})
        monkeypatch.setattr(tt, "_active_environments", {})

        # Each session ran commands that recorded its own cwd. No env alive,
        # no registered overrides — just the records.
        tt.record_session_cwd("sess-a", str(wt_a))
        tt.record_session_cwd("sess-b", str(wt_b))

        assert ftp._resolve_path_for_task("f.py", task_id="sess-a") == (wt_a / "f.py")
        assert ftp._resolve_path_for_task("f.py", task_id="sess-b") == (wt_b / "f.py")

    def test_record_beats_foreign_env_cwd_without_ownership_metadata(self, tmp_path, monkeypatch):
        """The leak-A scenario, solved structurally: the shared env's cwd is
        never consulted for path resolution — only the session's own record."""
        import tools.file_tools as ft
        import tools.file_tools_paths as ftp

        wt_a = tmp_path / "wt_a"
        wt_b = tmp_path / "wt_b"
        for d in (wt_a, wt_b):
            d.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(ft, "_file_ops_cache", {})

        class _Env:
            cwd = str(wt_b)  # another session's leftover cd on the shared env

        monkeypatch.setattr(tt, "_active_environments", {"default": _Env()})
        tt.record_session_cwd("sess-a", str(wt_a))

        resolved = ftp._resolve_path_for_task("f.py", task_id="sess-a")
        assert resolved == (wt_a / "f.py")
        assert not str(resolved).startswith(str(wt_b))


class TestDelegateSeedsChildRecord:
    def test_child_record_seeded_from_parent_then_isolated(self):
        tt.record_session_cwd("parent-task", "/parent/worktree")
        # what delegate_tool does at spawn:
        tt.record_session_cwd("child-1", tt.get_session_cwd("parent-task"))

        assert tt.get_session_cwd("child-1") == "/parent/worktree"
        # child cds somewhere; parent record must be untouched.
        tt.record_session_cwd("child-1", "/child/scratch")
        assert tt.get_session_cwd("parent-task") == "/parent/worktree"
        assert tt.get_session_cwd("child-1") == "/child/scratch"


class TestReapedEnvFallbackIsFillOnly:
    """file_tools' reaped-env rescue (#26211) must not overwrite the record.

    The cached file_ops' ``cwd`` is a snapshot of the SHARED env, so it can
    belong to another session — the same class of error as the
    interrupted-command bug (#85658). The rescue may only fill an ABSENT
    record.
    """

    def _reap(self, monkeypatch, tmp_path, task_id, stale_cwd):
        import tools.file_tools as ft
        import tools.file_tools_paths as ftp

        class _StaleFileOps:
            cwd = stale_cwd

        # Cached file_ops whose env was reaped: cache entry present,
        # _active_environments empty. The cache is keyed by the COLLAPSED
        # container id ("default" for plain sessions) — that collapse is
        # exactly why the snapshot can belong to another session.
        container_id = tt._resolve_container_task_id(task_id)
        monkeypatch.setattr(ft, "_file_ops_cache", {container_id: _StaleFileOps()})
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        ft._get_file_ops(task_id)

    def test_existing_record_survives_the_rescue(self, tmp_path, monkeypatch):
        tt.record_session_cwd("sess-a", "/my/worktree")
        self._reap(monkeypatch, tmp_path, "sess-a", "/other/sessions/dir")
        assert tt.get_session_cwd("sess-a") == "/my/worktree"

    def test_absent_record_is_filled(self, tmp_path, monkeypatch):
        """#26211 stays fixed: a recordless session still gets the rescue."""
        self._reap(monkeypatch, tmp_path, "sess-b", "/last/known/dir")
        assert tt.get_session_cwd("sess-b") == "/last/known/dir"


class TestCommandCwdReadsTheRecord:
    """_resolve_command_cwd: workdir > session record > default. Nothing else."""

    def test_record_beats_default(self):
        tt.record_session_cwd("sess-a", "/my/worktree")
        resolved = tt._resolve_command_cwd(
            workdir=None,
            default_cwd="/config/default",
            session_key="sess-a",
        )
        assert resolved == "/my/worktree"


    def test_other_sessions_record_is_not_consulted(self):
        tt.record_session_cwd("sess-b", "/other/worktree")
        resolved = tt._resolve_command_cwd(
            workdir=None,
            default_cwd="/config/default",
            session_key="sess-a",
        )
        assert resolved == "/config/default"

    def test_cd_then_next_command_runs_in_the_new_dir(self, monkeypatch):
        """E2E through terminal_tool: the record round-trips cd state."""
        import json

        class FakeEnv:
            env = {}
            cwd = "/start"
            def execute(self, command, **kwargs):
                self.last_cwd_arg = kwargs.get("cwd")
                if command.startswith("cd "):
                    self.cwd = command[3:]
                    # A completed cd emits the cwd marker; the parse sets both.
                    return {"output": "", "returncode": 0, "cwd_observed": True}
                return {"output": "", "returncode": 0}

        fake = FakeEnv()
        monkeypatch.setattr(tt, "_active_environments", {"sess-a": fake})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": "/default", "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )

        json.loads(tt.terminal_tool(command="cd /project", task_id="sess-a"))
        assert tt.get_session_cwd("sess-a") == "/project"
        json.loads(tt.terminal_tool(command="pwd", task_id="sess-a"))
        assert fake.last_cwd_arg == "/project"
