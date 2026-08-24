"""Stranded bot-session adoption (#93296 follow-up).

Pre-#93296 misrouting accumulated profile-bot sessions in the DEFAULT
profile's state.db. Post-fix, profile-scoped resumes correctly target the
profile's own store — which never saw the session — so the same chat 4001'd
for the opposite reason. These tests pin the heal: the unit-level adoption
primitive (SessionDB.adopt_session_lineage_from) and its invariants.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB

STRANDED_ID = "20260823_043331_c93770"


@pytest.fixture()
def stores(tmp_path):
    default_db = SessionDB(db_path=tmp_path / "state.db")
    profile_home = tmp_path / "profiles" / "developer"
    profile_home.mkdir(parents=True)
    profile_db = SessionDB(db_path=profile_home / "state.db")
    yield default_db, profile_db
    default_db.close()
    profile_db.close()


def _seed_stranded(db, session_id=STRANDED_ID, turns=3, title="Bot Chat", **kwargs):
    db.create_session(session_id, source="tui", **kwargs)
    db.set_session_title(session_id, title)
    for i in range(1, turns + 1):
        db.append_message(session_id, "user", f"question {i}")
        db.append_message(session_id, "assistant", f"answer {i}")


def test_adoption_moves_session_and_messages(stores):
    default_db, profile_db = stores
    _seed_stranded(default_db)

    result = profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)

    assert result["adopted"] is True
    assert result["imported"] == 1
    row = profile_db.get_session(STRANDED_ID)
    assert row is not None
    msgs = profile_db.get_messages(STRANDED_ID)
    assert len(msgs) == 6
    assert msgs[0]["content"] == "question 1"
    assert msgs[-1]["content"] == "answer 3"


def test_donor_is_archived_not_deleted(stores):
    default_db, profile_db = stores
    _seed_stranded(default_db)

    profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)

    donor = default_db.get_session(STRANDED_ID)
    assert donor is not None, "donor row must survive (archived, never deleted)"
    assert donor["archived"]
    assert donor["end_reason"] == "adopted_by_profile"
    # bytes stay recoverable
    assert len(default_db.get_messages(STRANDED_ID)) == 6


def test_adoption_archive_is_not_recoverable_resurrectable(stores):
    """Canonical-lookup resurrection must NOT undo an adoption."""
    default_db, profile_db = stores
    _seed_stranded(default_db)
    profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)

    assert "adopted_by_profile" not in SessionDB.RECOVERABLE_END_REASONS
    assert default_db.unarchive_recoverable_session(STRANDED_ID) is False
    assert default_db.get_session(STRANDED_ID)["archived"]


def test_adoption_is_idempotent(stores):
    default_db, profile_db = stores
    _seed_stranded(default_db)

    first = profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)
    second = profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)

    assert first["adopted"] and second["adopted"]
    assert second["imported"] == 0 and second["skipped"] == 1
    assert len(profile_db.get_messages(STRANDED_ID)) == 6


def test_missing_donor_session_is_reported_not_raised(stores):
    default_db, profile_db = stores
    result = profile_db.adopt_session_lineage_from(default_db, "nope")
    assert result["adopted"] is False
    assert "not found" in result["error"]


def test_compression_lineage_adopts_as_a_unit(stores):
    """A compacted conversation is parent(end_reason=compression) -> child.

    Adoption must carry BOTH segments so the profile store can follow the
    continuation chain, and must retire both donor rows.
    """
    default_db, profile_db = stores
    parent, child = "sess-parent", "sess-child"
    _seed_stranded(default_db, session_id=parent, turns=2)
    default_db.end_session(parent, "compression")
    default_db.create_session(child, source="tui", parent_session_id=parent)
    default_db.set_session_title(child, "Bot Chat")
    default_db.append_message(child, "user", "post-compaction question")
    default_db.append_message(child, "assistant", "post-compaction answer")

    result = profile_db.adopt_session_lineage_from(default_db, parent)

    assert result["adopted"] is True
    assert result["imported"] == 2
    assert profile_db.get_session(parent) is not None
    assert profile_db.get_session(child) is not None
    assert profile_db.get_session(child)["parent_session_id"] == parent
    for sid in (parent, child):
        donor = default_db.get_session(sid)
        assert donor["archived"], f"{sid} must be retired in donor store"


def test_adoption_does_not_touch_unrelated_sessions(stores):
    default_db, profile_db = stores
    _seed_stranded(default_db)
    _seed_stranded(default_db, session_id="other-session", title="Other Chat")

    profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)

    other = default_db.get_session("other-session")
    assert not other["archived"]
    assert profile_db.get_session("other-session") is None


# -------------------------------------------------------------------------
# Handler-level: the real session.resume JSON-RPC path (server.handle_request)
# -------------------------------------------------------------------------
#
# Mirrors tests/tui_gateway/test_session_profile_db.py's harness: import the
# real server (with env_loader/banner mocked at first import), wire _get_db()
# to the default store, and drive session.resume with profile= + lazy=True so
# the resume registers a live record WITHOUT building an agent.

import importlib
from unittest.mock import MagicMock, patch


@pytest.fixture()
def gateway(tmp_path, monkeypatch):
    from pathlib import Path as _P

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(_P, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)

    default_db = SessionDB(db_path=home / "state.db")
    mod._db = default_db

    profile_home = home / "profiles" / "developer"
    profile_home.mkdir(parents=True)

    # session.resume resolves the profile via hermes_cli.profiles
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda name: str(profile_home)
    )

    yield mod, default_db, profile_home

    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None
    default_db.close()


def test_profile_resume_adopts_stranded_default_store_session(gateway):
    """The live repro: resume a session by id on a profile whose store has
    never seen it, while the id exists in the default store. Pre-heal this
    was a hard 4007; now the lineage is adopted and the resume succeeds."""
    mod, default_db, profile_home = gateway
    _seed_stranded(default_db)

    resp = mod.handle_request(
        {
            "id": "1",
            "method": "session.resume",
            "params": {
                "session_id": STRANDED_ID,
                "profile": "developer",
                "lazy": True,
            },
        }
    )

    assert not resp.get("error"), f"resume failed: {resp.get('error')}"
    result = resp["result"]
    assert result.get("resumed") == STRANDED_ID
    assert result.get("message_count") == 6

    # Durably adopted into the profile's own store...
    pdb = SessionDB(db_path=profile_home / "state.db")
    try:
        assert pdb.get_session(STRANDED_ID) is not None
        assert len(pdb.get_messages(STRANDED_ID)) == 6
    finally:
        pdb.close()
    # ...and retired (archived, never deleted) in the default store.
    donor = default_db.get_session(STRANDED_ID)
    assert donor["archived"]
    assert donor["end_reason"] == "adopted_by_profile"


def test_profile_resume_of_truly_unknown_session_still_4007s(gateway):
    """Adoption must not weaken the not-found contract: an id in NEITHER
    store keeps failing with 4007 exactly as before."""
    mod, _default_db, _profile_home = gateway

    resp = mod.handle_request(
        {
            "id": "2",
            "method": "session.resume",
            "params": {
                "session_id": "definitely-not-anywhere",
                "profile": "developer",
                "lazy": True,
            },
        }
    )

    assert resp.get("error")
    assert resp["error"]["code"] == 4007


def test_launch_profile_resume_path_is_untouched(gateway):
    """A resume WITHOUT profile scope (owns_db=False) never consults the
    adoption fallback — unknown ids fail 4007 on the shared handle."""
    mod, _default_db, _profile_home = gateway

    resp = mod.handle_request(
        {
            "id": "3",
            "method": "session.resume",
            "params": {"session_id": "unknown-launch-id", "lazy": True},
        }
    )

    assert resp.get("error")
    assert resp["error"]["code"] == 4007


# -------------------------------------------------------------------------
# Review-hardening regressions (deleg_e8230ed7): title-collision safety,
# divergence guard, donor_retired truthfulness, no re-adoption of retired
# donors, and a non-vacuous launch-profile gating test.
# -------------------------------------------------------------------------


def test_title_lookup_is_never_used_for_adoption(gateway):
    """H1: a profile resume by a TITLE (not id) that collides with an
    unrelated default-store session must NOT adopt/retire it. Only exact-id
    donors qualify."""
    mod, default_db, profile_home = gateway
    # Unrelated default-profile conversation titled like every bot chat.
    _seed_stranded(default_db, session_id="innocent-default", title="Bot Chat")

    resp = mod.handle_request(
        {
            "id": "10",
            "method": "session.resume",
            "params": {
                # resolves nothing by id; would have matched by title pre-fix
                "session_id": "Bot Chat",
                "profile": "developer",
                "lazy": True,
            },
        }
    )

    assert resp.get("error") and resp["error"]["code"] == 4007
    innocent = default_db.get_session("innocent-default")
    assert not innocent["archived"], "unrelated session must never be retired"
    pdb = SessionDB(db_path=profile_home / "state.db")
    try:
        assert pdb.get_session("innocent-default") is None
    finally:
        pdb.close()


def test_archived_donor_is_not_readopted(gateway):
    """M4: after profile A adopts (donor archived), a second profile resuming
    the same id must NOT clone the conversation from the archived donor."""
    mod, default_db, profile_home = gateway
    _seed_stranded(default_db)
    # Simulate a prior completed adoption's retirement stamp.
    default_db.reopen_session(STRANDED_ID)
    default_db.end_session(STRANDED_ID, "adopted_by_profile")
    default_db.set_session_archived(STRANDED_ID, True)

    resp = mod.handle_request(
        {
            "id": "11",
            "method": "session.resume",
            "params": {
                "session_id": STRANDED_ID,
                "profile": "developer",
                "lazy": True,
            },
        }
    )

    assert resp.get("error") and resp["error"]["code"] == 4007
    pdb = SessionDB(db_path=profile_home / "state.db")
    try:
        assert pdb.get_session(STRANDED_ID) is None
    finally:
        pdb.close()


def test_launch_profile_resume_never_adopts_even_when_donor_exists(gateway):
    """Non-vacuous owns_db gating (reviewer 3): seed a REAL donor in the
    default store, resume WITHOUT profile scope under an unknown id — the
    fallback must not run, and the donor must stay untouched."""
    mod, default_db, _profile_home = gateway
    _seed_stranded(default_db)

    resp = mod.handle_request(
        {
            "id": "12",
            "method": "session.resume",
            "params": {"session_id": "unknown-launch-id", "lazy": True},
        }
    )

    assert resp.get("error") and resp["error"]["code"] == 4007
    donor = default_db.get_session(STRANDED_ID)
    assert not donor["archived"]
    assert donor["end_reason"] is None


def test_divergent_donor_is_not_retired(stores):
    """H2: donor gained messages after a partial adoption — re-adoption must
    NOT retire it (the newer messages would become unreachable)."""
    default_db, profile_db = stores
    _seed_stranded(default_db)
    first = profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)
    assert first["adopted"] and first["donor_retired"]

    # Donor keeps living (e.g. user kept chatting there) — un-retire + append.
    default_db.set_session_archived(STRANDED_ID, False)
    default_db.append_message(STRANDED_ID, "user", "late question")
    default_db.append_message(STRANDED_ID, "assistant", "late answer")

    second = profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)

    assert second["adopted"] is True  # profile copy still serves
    assert second["donor_retired"] is False
    donor = default_db.get_session(STRANDED_ID)
    assert not donor["archived"], "diverged donor must stay reachable"
    assert len(default_db.get_messages(STRANDED_ID)) == 8


def test_donor_retired_reports_false_on_retirement_failure(stores, monkeypatch):
    """M1: donor_retired must not lie when retirement fails."""
    default_db, profile_db = stores
    _seed_stranded(default_db)

    def _boom(_sid, _reason):
        raise RuntimeError("locked")

    monkeypatch.setattr(default_db, "end_session", _boom)
    result = profile_db.adopt_session_lineage_from(default_db, STRANDED_ID)

    assert result["adopted"] is True
    assert result["donor_retired"] is False
    assert not default_db.get_session(STRANDED_ID)["archived"]
