"""Agent-facing ``kanban_reassign``: the CLI's own assign kernel, reached as a tool.

Covers, per the work order:
  - it REUSES ``hermes_cli.kanban_db.reassign_task`` / ``assign_task`` — the
    exact functions ``hermes kanban reassign`` runs — so the lifecycle and
    active-worker guards are the CLI's, not a reimplementation
  - the destination profile is validated against the authoritative profile
    enumeration BEFORE the board is opened, so a typo leaves the board
    byte-for-byte unchanged
  - a claimed running card is refused and changes nothing; ``reclaim`` is the
    documented escape hatch
  - board isolation: only the board this call opens is written
  - orchestrator-only gating, schema/registry registration
"""
from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------- Fixtures

def _install_profile(home, name):
    """A live named profile: ``config.yaml`` is the identity marker that makes
    a directory a profile (``named_profile_has_identity``)."""
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    return profile_dir


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    """Orchestrator-context board (no dispatcher worker env) with one peer profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_SESSION_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_DELEGATED_CHILD_CONTEXT",
                "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    _install_profile(home, "peer")

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _db():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    return kb, kbc


def _dispatch(tool, args):
    """Real registry dispatch, tolerant of str or dict handler results."""
    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    from tools.registry import registry
    out = registry.dispatch(tool, args)
    return out if isinstance(out, dict) else json.loads(out)


def _new_task(kb, kbc, title="card", assignee="default", claimed=False):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, assignee=assignee)
        if claimed:
            assert kb.claim_task(conn, tid) is not None
    return tid


def _state(kb, kbc, tid):
    """Everything a refused call must not disturb."""
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        return {
            "assignee": task.assignee,
            "status": task.status,
            "claim_lock": task.claim_lock,
            "consecutive_failures": task.consecutive_failures,
            "events": [tuple(r) for r in conn.execute(
                "SELECT kind, payload, run_id FROM task_events WHERE task_id = ? ORDER BY id",
                (tid,))],
        }


# --------------------------------------------------------------------------- happy path

def test_reassign_moves_the_card_and_reads_the_audit_event_back(board_env):
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default")

    out = _dispatch("kanban_reassign", {"task_id": tid, "assignee": "peer"})
    assert out["ok"] is True, out
    assert out["assignee"] == "peer"
    assert out["previous_assignee"] == "default"
    # Readback of the STORED event, not a restatement of the request.
    assert out["audit"]["kind"] == "assigned"
    assert out["audit"]["payload"] == {"assignee": "peer", "from": "default"}

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).assignee == "peer"
        kinds = [r[0] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
    assert kinds.count("assigned") == 1


def test_reassign_accepts_the_default_profile(board_env):
    """``default`` resolves through the profile root, not the current home."""
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="peer")
    out = _dispatch("kanban_reassign", {"task_id": tid, "assignee": "default"})
    assert out["ok"] is True, out
    assert out["assignee"] == "default"


def test_reassign_accepts_a_name_the_cli_would_normalize(board_env):
    """The kernel normalizes case/whitespace exactly like ``hermes -p`` does."""
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default")
    out = _dispatch("kanban_reassign", {"task_id": tid, "assignee": "  Peer  "})
    assert out["ok"] is True, out
    assert out["assignee"] == "peer"


# --------------------------------------------------------------------------- refusals

def test_reassign_refuses_an_unknown_destination_and_changes_nothing(board_env):
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default")
    before = _state(kb, kbc, tid)

    out = _dispatch("kanban_reassign", {"task_id": tid, "assignee": "nope"})
    assert out.get("ok") is not True
    # Wording contract: a profile is "not found", never "not installed".
    assert "profile 'nope' was not found" in out["error"], out
    assert "not installed" not in out["error"], out
    # The roster is named back so the model can correct itself.
    assert "default" in out["error"] and "peer" in out["error"]
    assert out["error"].endswith("Nothing changed.")
    assert _state(kb, kbc, tid) == before, "a refused reassign mutated the board"


def test_reassign_refuses_a_missing_assignee_before_touching_the_board(board_env):
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default")
    before = _state(kb, kbc, tid)
    out = _dispatch("kanban_reassign", {"task_id": tid})
    assert out.get("ok") is not True
    assert "assignee is required" in out["error"]
    assert _state(kb, kbc, tid) == before


def test_reassign_refuses_a_claimed_running_card_and_writes_nothing(board_env):
    """The CLI's active-worker guard, verbatim: ``assign_task`` refuses inside
    the write transaction before any UPDATE, so no ``assigned`` event lands."""
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default", claimed=True)
    before = _state(kb, kbc, tid)
    assert before["claim_lock"] is not None

    out = _dispatch("kanban_reassign", {"task_id": tid, "assignee": "peer"})
    assert out.get("ok") is not True
    assert "still running under a live claim" in out["error"], out
    assert "reclaim=true" in out["error"]
    assert out["error"].endswith("Nothing changed.")
    assert _state(kb, kbc, tid) == before, "a refused reassign mutated the board"


def test_reassign_with_reclaim_releases_the_claim_first(board_env):
    """``reclaim=true`` is the documented \"this profile's model is broken\" path."""
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default", claimed=True)

    out = _dispatch("kanban_reassign",
                    {"task_id": tid, "assignee": "peer", "reclaim": True,
                     "reason": "profile model is broken"})
    assert out["ok"] is True, out
    assert out["assignee"] == "peer"
    assert out["claim_reclaimed"] is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.assignee == "peer"
        assert task.claim_lock is None
        kinds = [r[0] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
    # reclaim + assign both recorded, and the LAST event is the assignment.
    assert "assigned" in kinds


# --------------------------------------------------------------------------- isolation / gating

def test_reassign_is_scoped_to_the_board_it_opens(board_env, tmp_path):
    """A card on the default board is invisible to another board's connection —
    and vice versa. Only the board this call opens is written."""
    kb, kbc = _db()
    kb.create_board("peerboard", name="Peer board")
    default_tid = _new_task(kb, kbc, assignee="default")
    with kbc.connect(board="peerboard") as conn:
        other_tid = kb.create_task(conn, title="other board card", assignee="default")
    default_before = _state(kb, kbc, default_tid)

    # Default-board card addressed at the other board: not found, nothing written.
    out = _dispatch("kanban_reassign",
                    {"task_id": default_tid, "assignee": "peer", "board": "peerboard"})
    assert out.get("ok") is not True
    assert "not found" in out["error"]
    assert _state(kb, kbc, default_tid) == default_before

    # And the other board's card needs its own board argument.
    out = _dispatch("kanban_reassign", {"task_id": other_tid, "assignee": "peer"})
    assert out.get("ok") is not True
    assert "not found" in out["error"]
    with kbc.connect(board="peerboard") as conn:
        assert kb.get_task(conn, other_tid).assignee == "default"

    out = _dispatch("kanban_reassign",
                    {"task_id": other_tid, "assignee": "peer", "board": "peerboard"})
    assert out["ok"] is True, out
    with kbc.connect(board="peerboard") as conn:
        assert kb.get_task(conn, other_tid).assignee == "peer"
    assert _state(kb, kbc, default_tid) == default_before


def test_reassign_is_orchestrator_only(board_env, monkeypatch):
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default")
    before = _state(kb, kbc, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    out = _dispatch("kanban_reassign", {"task_id": tid, "assignee": "peer"})
    assert out.get("ok") is not True
    assert "orchestrator-only" in out["error"]
    assert _state(kb, kbc, tid) == before

    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    assert "kanban_reassign" in kt._ORCHESTRATOR_TOOLS


def test_reassign_is_registered_in_the_kanban_toolset(board_env):
    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    from tools.registry import registry
    from toolsets import TOOLSETS

    schema = registry.get_schema("kanban_reassign")
    assert schema is not None, "kanban_reassign is not registered"
    assert schema["parameters"]["required"] == ["task_id", "assignee"]
    props = schema["parameters"]["properties"]
    assert props["assignee"]["type"] == "string"
    assert props["reclaim"]["type"] == "boolean"
    assert "kanban_reassign" in TOOLSETS["kanban"]["tools"]
    # Hidden from task workers, shown to orchestrator profiles.
    assert "kanban_reassign" in kt._ORCHESTRATOR_TOOLS


def test_reassign_rejects_unknown_parameters_without_mutation(board_env):
    kb, kbc = _db()
    tid = _new_task(kb, kbc, assignee="default")
    before = _state(kb, kbc, tid)
    out = _dispatch("kanban_reassign", {"task_id": tid, "assignee": "peer",
                                        "misspelled_argument": True})
    assert out.get("ok") is not True
    assert "unknown parameter" in out["error"]
    assert _state(kb, kbc, tid) == before
