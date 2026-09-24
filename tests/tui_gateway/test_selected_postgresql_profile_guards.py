"""Selected-PostgreSQL target profiles cannot leak through TUI session/profile RPCs.

The launch home stays SQLite (its own gateway serves it); the RISK is a *foreign*
profile whose ``config.yaml`` selects PostgreSQL being routed through this
process's legacy SessionDB runtime (``session.resume``'s dedicated profile handle,
``session.create``-seeded branch rows / ``session.branch`` agent builds, and the
``profiles.list`` roster). Handler bodies are REBOUND onto server.py globals
(method_ctx.bind_module), so every readiness import must be function-local — a
module-level import is invisible to the rebound body (runtime-proven NameError).

Contract under test, all under ``trap_state_db_opens`` across BOTH homes:
- ``session.resume`` against a selected-PG foreign profile returns a typed
  JSON-RPC error carrying the capability report; zero state.db opens and no
  state.db created in the foreign home.
- ``session.branch``'s build path (``_build_branch_agent``) raises the typed
  error before acquiring the parent profile's state.db.
- ``profiles.list`` roster fields for the selected-PG profile are MARKED
  unavailable (``sessions_unavailable`` with a reason), never silently empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tui_gateway.server as srv
from state_store_runtime_readiness import PostgreSQLRuntimeActivationError, trap_state_db_opens

_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


@pytest.fixture
def homes(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """SQLite launch home + a selected-PG foreign profile under it."""
    launch = tmp_path / ".hermes"
    foreign = launch / "profiles" / "selected-pg"
    foreign.mkdir(parents=True)
    (foreign / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    return launch, foreign


def _assert_typed_error(envelope: dict, *, code: int) -> None:
    """JSON-RPC error whose data carries the readiness capability report."""
    assert envelope.get("error"), f"expected a JSON-RPC error, got {envelope}"
    assert envelope["error"]["code"] == code
    data = envelope["error"].get("data") or {}
    missing = data.get("missing_capabilities") or []
    assert "tui-api-session-runtime" in missing, data


def test_session_resume_foreign_selected_pg_is_typed_error(homes, monkeypatch):
    """``session.resume`` refuses BEFORE any foreign state.db open or creation."""
    launch, foreign = homes
    monkeypatch.setattr(srv, "_schedule_resume_hydration", lambda *a, **k: None)
    monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(srv, "_enable_gateway_prompts", lambda: None)

    with trap_state_db_opens(launch, foreign):
        envelope = srv._methods["session.resume"](1, {
            "session_id": "missing1", "profile": "selected-pg", "defer_history": True,
        })
    _assert_typed_error(envelope, code=5000)
    assert not (foreign / "state.db").exists()
    assert not (launch / "state.db").exists()


def test_branch_build_agent_foreign_selected_pg_raises_typed(homes):
    """``_build_branch_agent`` fails loud BEFORE acquiring the parent state.db."""
    launch, foreign = homes
    session = {"profile_home": str(foreign), "history_lock": __import__("threading").RLock()}

    with trap_state_db_opens(launch, foreign):
        with pytest.raises(PostgreSQLRuntimeActivationError):
            srv._build_branch_agent(session, "nsid", "nkey", [], "cli")
    assert not (foreign / "state.db").exists()


def test_profiles_list_marks_selected_pg_sessions_unavailable(homes):
    """Roster marks the selected-PG profile's session fields, never silent-empty."""
    launch, foreign = homes
    rows = srv._methods["profiles.list"](1, {})["result"]["profiles"]
    row = next(p for p in rows if p["name"] == "selected-pg")
    flagged = row.get("sessions_unavailable")
    assert flagged, "selected-PG roster row must flag session fields unavailable"
    assert "tui-api-session-runtime" in (flagged.get("missing_capabilities") or [])
    assert not (foreign / "state.db").exists()


def test_profiles_list_sqlite_profile_still_resolves(homes):
    """The guard is per-profile: the SQLite launch profile keeps full fields."""
    launch, _ = homes
    rows = srv._methods["profiles.list"](1, {})["result"]["profiles"]
    row = next(p for p in rows if p["name"] == "default")
    assert "sessions_unavailable" not in row
    assert row["last_session"] is None  # no db yet — a true None, not a swallow
