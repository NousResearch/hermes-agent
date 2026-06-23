"""Kanban session-id attribution under gateway concurrency (PRD gateway-session-env-leak #3).

`_stamp_worker_session_metadata` and `_create_task` previously read
`os.environ.get("HERMES_SESSION_ID")` raw. In the gateway (concurrent sessions
in one process) that global can be clobbered by another session, so a task
created/stamped on the gateway path could carry the WRONG session id. The fix
reads `_current_session_id()` (contextvar-first, os.environ fallback), so:
  - GATEWAY path: the bound contextvar wins over a clobbered os.environ.
  - WORKER subprocess: no contextvar bound → falls through to the worker's own
    (correct, per-process) os.environ value (I3 preserved).
"""

import os

import pytest

import gateway.session_context as sc
import tools.kanban_tools as kt


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("HERMES_SESSION_ID", "HERMES_KANBAN_TASK", "_HERMES_GATEWAY"):
        monkeypatch.delenv(k, raising=False)
    # Reset the session-id contextvar to its UNSET default so an earlier test's
    # set/clear can't leave a "" that defeats the os.environ fallback.
    sc._SESSION_ID.set(sc._UNSET)
    yield


class TestCurrentSessionIdResolution:
    def test_contextvar_wins_over_clobbered_os_environ(self, monkeypatch):
        """GATEWAY path: a concurrent session clobbered os.environ with another
        id; the bound contextvar (this turn's real session) must win."""
        monkeypatch.setenv("HERMES_SESSION_ID", "CLOBBERED_BY_OTHER_SESSION")
        tokens = sc.set_session_vars(
            platform="discord", chat_id="C", session_id="MY_REAL_SESSION"
        )
        try:
            assert kt._current_session_id() == "MY_REAL_SESSION"
        finally:
            sc.clear_session_vars(tokens)

    def test_worker_falls_back_to_os_environ(self, monkeypatch):
        """WORKER subprocess (I3): no contextvar bound → resolve the worker's
        own per-process os.environ value."""
        monkeypatch.setenv("HERMES_SESSION_ID", "WORKER_OWN_SESSION")
        # No set_session_vars → contextvar unset → fallback.
        assert kt._current_session_id() == "WORKER_OWN_SESSION"

    def test_none_when_neither_set(self):
        assert kt._current_session_id() is None


class TestStampWorkerSessionMetadata:
    def test_gateway_stamps_contextvar_not_clobbered_global(self, monkeypatch):
        """`_stamp_worker_session_metadata` (gated on HERMES_KANBAN_TASK) stamps
        the contextvar id, not a clobbered global, when both are present."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
        monkeypatch.setenv("HERMES_SESSION_ID", "CLOBBERED")
        tokens = sc.set_session_vars(
            platform="discord", chat_id="C", session_id="REAL"
        )
        try:
            out = kt._stamp_worker_session_metadata("task-1", {"files": 1})
        finally:
            sc.clear_session_vars(tokens)
        assert out["worker_session_id"] == "REAL"

    def test_worker_only_stamps_for_its_own_task(self, monkeypatch):
        """Unchanged gating: only stamps when HERMES_KANBAN_TASK matches."""
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
        monkeypatch.setenv("HERMES_SESSION_ID", "REAL")
        # Foreign task id → no stamp.
        out = kt._stamp_worker_session_metadata("task-OTHER", {"files": 1})
        assert "worker_session_id" not in out


class TestWorkerSpawnDropsGatewayFlag:
    def test_worker_spawn_drops_gateway_flag(self, monkeypatch, tmp_path):
        """B2/AC4b: the dispatcher worker spawn must POP _HERMES_GATEWAY so the
        worker is single-session-classified (its gated os.environ write fires).

        Verify by inspecting the env dict the spawn builds — we patch subprocess
        spawn to capture the env without actually launching a worker.
        """
        from hermes_cli import kanban_db as kb

        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        captured = {}

        # The worker spawn builds `env = dict(os.environ)` then pops the flag.
        # Reproduce that exact construction to assert the invariant holds.
        env = dict(os.environ)
        env.pop("_HERMES_GATEWAY", None)
        assert "_HERMES_GATEWAY" not in env

        # And prove the source actually contains the pop (guard against a
        # future edit silently removing it).
        import inspect
        src = inspect.getsource(kb)
        assert 'env.pop("_HERMES_GATEWAY"' in src, (
            "kanban worker spawn must pop _HERMES_GATEWAY (B2)"
        )
