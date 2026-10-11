"""A local session's recorded cwd can disappear mid-session.

The session wrapper runs ``builtin cd -- <cwd> || exit 126``, so a recorded cwd
that no longer exists aborts every later command *before* it runs: the command
never executes, the session looks wedged, and only a gateway restart clears it.
The resolver must fall back to ``default_cwd`` so the session self-heals.

A recorded *remote* path (ssh) must not be probed against the host filesystem —
``os.path.isdir`` on the gateway's own disk says nothing about the remote host.

The fallback is only as good as the ``default_cwd`` it is handed, and the
production caller hands it ``plan.cwd`` — which ``_plan_execution`` builds FROM
the session record. Resolving a stale record there made the record its own
fallback, so the guard below returned the very path that was gone and the
session stayed wedged. ``_plan_execution`` therefore validates the record at
the planner boundary; the end-to-end class drives the real tool so a
regression that only fixes the helper cannot pass.
"""

import json
import os
import shutil

import pytest

import tools.terminal_tool as tt
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch):
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.delenv("TERMINAL_ENV", raising=False)


@pytest.fixture
def env(tmp_path):
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    environment.init_session()
    yield environment
    environment.cleanup()


def _resolve(session_key, default_cwd, env_type):
    return tt._resolve_command_cwd(
        workdir=None,
        default_cwd=default_cwd,
        session_key=session_key,
        env_type=env_type,
    )


class TestLocalStaleCwd:
    def test_stale_local_cwd_falls_back_to_a_usable_default(self, tmp_path):
        stale = tmp_path / "gone"
        stale.mkdir()
        fallback = tmp_path / "fallback"
        fallback.mkdir()
        tt.record_session_cwd("sess-stale", str(stale))
        stale.rmdir()
        assert _resolve("sess-stale", str(fallback), "local") == str(fallback)

    def test_stale_record_is_left_alone_when_the_default_is_gone_too(self, tmp_path):
        """Swapping one missing directory for another buys nothing — the
        wrapper's ``cd`` aborts either way — so the record stays the answer.
        This also keeps the resolver a pure function of its inputs for callers
        that pass synthetic paths."""
        stale = tmp_path / "gone"
        stale.mkdir()
        tt.record_session_cwd("sess-stale-2", str(stale))
        stale.rmdir()
        assert _resolve("sess-stale-2", "/default", "local") == str(stale)

    def test_live_local_cwd_is_kept(self, tmp_path):
        live = tmp_path / "live"
        live.mkdir()
        tt.record_session_cwd("sess-live", str(live))
        assert _resolve("sess-live", "/default", "local") == str(live)

    def test_unrecorded_session_uses_default(self):
        assert _resolve("sess-none", "/default", "local") == "/default"


class TestRemoteCwdIsNotProbedLocally:
    def test_ssh_recorded_path_survives_even_if_absent_here(self):
        tt.record_session_cwd("sess-ssh", "/remote/only/path")
        assert _resolve("sess-ssh", "~", "ssh") == "/remote/only/path"


class TestPlannerKeepsTheConfiguredDefault:
    """``_plan_execution`` must not resolve a stale record into ``plan.cwd``.

    ``plan.cwd`` IS the ``default_cwd`` the per-command resolver is handed, so
    a stale record resolved here makes the helper's fallback a no-op.
    """

    def _plan(self, tmp_path, monkeypatch, task_id):
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        # ``_host_local`` pins the backend to ``local`` without pulling in the
        # refusal-scope machinery; the cwd resolution below is unconditional.
        return tt._plan_execution(
            "pwd", task_id=task_id, timeout=None, background=False, _host_local=True,
        )

    def test_stale_record_falls_back_to_the_configured_default(self, tmp_path, monkeypatch):
        stale = tmp_path / "gone"
        stale.mkdir()
        tt.record_session_cwd("sess-plan", str(stale))
        stale.rmdir()

        plan = self._plan(tmp_path, monkeypatch, "sess-plan")

        assert os.path.realpath(plan.cwd) == os.path.realpath(str(tmp_path))

    def test_live_record_still_wins_over_the_configured_default(self, tmp_path, monkeypatch):
        live = tmp_path / "live"
        live.mkdir()
        tt.record_session_cwd("sess-plan-live", str(live))

        plan = self._plan(tmp_path, monkeypatch, "sess-plan-live")

        assert os.path.realpath(plan.cwd) == os.path.realpath(str(live))


class TestTerminalToolEndToEnd:
    """Drive ``tt.terminal_tool`` itself: real planner, real guard, real shell.

    The helper tests above hand ``_resolve_command_cwd`` an independent
    ``default_cwd`` literal, which is not what production passes — a fix that
    only patches the helper passes them and still leaves the session wedged.
    """

    def _tool(self, monkeypatch, env, command, task_id, default_cwd, timeout=None):
        # One shared env for every session, like the real local backend
        # (_resolve_container_task_id collapses cwd-only sessions to "default").
        monkeypatch.setattr(tt, "_active_environments", {"default": env})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": default_cwd, "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        return json.loads(
            tt.terminal_tool(command=command, task_id=task_id, timeout=timeout)
        )

    @pytest.mark.platforms("linux")
    def test_deleted_recorded_cwd_does_not_wedge_the_session(self, env, tmp_path, monkeypatch):
        """Reproduce the wedge: the session's own directory disappears, then
        the very next command must still run — in the configured default."""
        gone = tmp_path / "gone"
        gone.mkdir()

        result = self._tool(monkeypatch, env, f"cd {gone} && pwd", "sess", str(tmp_path))
        assert result["exit_code"] == 0, result
        assert tt.get_session_cwd("sess") == str(gone)

        # Mid-session the directory vanishes: an unmounted drive, a deleted
        # scratch dir. The wrapper's ``cd`` would abort every later command.
        shutil.rmtree(gone)

        result = self._tool(monkeypatch, env, "pwd", "sess", str(tmp_path))

        assert result["exit_code"] == 0, result
        assert os.path.realpath(result["output"].strip()) == os.path.realpath(str(tmp_path))
        # The record self-heals, so later commands do not re-resolve the ghost.
        assert os.path.realpath(tt.get_session_cwd("sess") or "") == os.path.realpath(str(tmp_path))
        # ... and the caller is told it ran somewhere it did not name.
        assert str(gone) in (result.get("cwd_fallback") or "")

    @pytest.mark.platforms("linux")
    def test_live_recorded_cwd_is_still_honored_end_to_end(self, env, tmp_path, monkeypatch):
        """The guard must not steal a session's working directory from it."""
        mine = tmp_path / "mine"
        mine.mkdir()

        result = self._tool(monkeypatch, env, f"cd {mine} && pwd", "sess", str(tmp_path))
        assert result["exit_code"] == 0, result

        result = self._tool(monkeypatch, env, "pwd", "sess", str(tmp_path))

        assert result["exit_code"] == 0, result
        assert os.path.realpath(result["output"].strip()) == os.path.realpath(str(mine))
        assert "cwd_fallback" not in result
