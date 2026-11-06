"""``session.resume`` heals a historical minted-shape row's never-persisted ``session_key``.

Regression for #149905: in a long-lived store, thousands of session rows carry
``session_key IS NULL`` — some written by older builds that never stamped the column,
some minted-but-never-persisted until resume materialized them. They resume fine by id,
but the canonical gateway rebuilds a messaging lane from the row's ``session_key``
(``_mount_plan`` routes by it), so a NULL-keyed weixin chat reopens read-only forever,
and repair-routing files the row as a permanently unrepairable orphan. On resume, heal
the row: when the target is a well-formed server-minted key and the found row's key is
NULL, stamp ``session_key = id`` — the key the server minted for this chat was never
persisted, and the id IS that key.

The invariants below are the safety case, not garnish: a legal key is never overwritten
(NULL-only), non-minted ids (cron lineage) stay keyless by design, and legacy-local rows
(cli/oneshot/...) MUST keep NULL — a non-empty ``session_key`` is exactly what makes
``validate_legacy_row`` refuse canonical adoption of that history, so healing those would
trade a readable zombie row for a not_found.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from hermes_state_local_migration import validate_legacy_row
from tui_gateway import server

MINTED_KEY = "20250912_041703_9f3ab2"  # minted shape; the column never got the stamp
OTHER_KEY = "20250901_101010_beef01"  # a legal key that belongs to a DIFFERENT id
CRON_ID = "7c1d2e93"  # runtime-id shape: cron lineage is keyless by design, not wounded


@pytest.fixture()
def real_db(monkeypatch, tmp_path):
    """Real SessionDB behind ``_get_db``, with the heavy resume machinery off."""
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda _key, _home=None: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(tmp_path))
    monkeypatch.setattr(server, "_child_run_active", lambda _key, _home=None: False)
    known = set(server._sessions)
    yield db
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)
    db.close()


def _resume(**params):
    return server.handle_request({"id": "1", "method": "session.resume", "params": params})


def _assert_resumed_ok(resp, target):
    assert "error" not in resp, resp
    assert resp["result"]["resumed"] == target
    assert resp["result"]["session_id"], "a live runtime must be registered"


def test_resume_heals_null_key_on_minted_row(real_db):
    """T1 — the #149905 wound: a minted-shape row with NULL session_key resumes, and the
    resume backfills the key so the canonical gateway can rebuild its lane."""
    real_db.create_session(MINTED_KEY, source="weixin", model="test-model")
    assert real_db.get_session(MINTED_KEY)["session_key"] is None, "precondition: the wounded row"

    resp = _resume(session_id=MINTED_KEY)

    _assert_resumed_ok(resp, MINTED_KEY)
    row = real_db.get_session(MINTED_KEY)
    assert row["session_key"] == MINTED_KEY, "resume must stamp the never-persisted key"

    # Idempotent: re-resuming a healed row succeeds again and never rewrites the key.
    resp = _resume(session_id=MINTED_KEY)
    _assert_resumed_ok(resp, MINTED_KEY)
    assert real_db.get_session(MINTED_KEY)["session_key"] == MINTED_KEY


def test_resume_never_touches_a_legal_key(real_db):
    """T2 — NULL-only: a row whose key was persisted resumes with its key byte-identical."""
    real_db.create_session(MINTED_KEY, source="weixin", model="test-model", session_key=OTHER_KEY)

    resp = _resume(session_id=MINTED_KEY)

    _assert_resumed_ok(resp, MINTED_KEY)
    assert real_db.get_session(MINTED_KEY)["session_key"] == OTHER_KEY


def test_resume_leaves_non_minted_keyless_row_alone(real_db):
    """T3 — cron lineage: a non-minted id with NULL key resumes as before and stays keyless."""
    real_db.create_session(CRON_ID, source="cron", model="test-model")

    resp = _resume(session_id=CRON_ID)

    _assert_resumed_ok(resp, CRON_ID)
    row = real_db.get_session(CRON_ID)
    assert row["session_key"] is None
    assert row["source"] == "cron"


def test_resume_keeps_legacy_local_row_adoptable(real_db):
    """T4 — cli history: healing it would flip ``validate_legacy_row`` to not_found and
    close the canonical adoption door, so a minted-shape cli row must stay keyless."""
    real_db.create_session(MINTED_KEY, source="cli", model="test-model")

    resp = _resume(session_id=MINTED_KEY)

    _assert_resumed_ok(resp, MINTED_KEY)
    row = dict(real_db.get_session(MINTED_KEY))
    assert row["session_key"] is None
    assert row["source"] == "cli"
    # The adoption contract this test protects: the row is still legal legacy history.
    validate_legacy_row(real_db, row)


def test_resume_materializes_minted_key_with_its_key(real_db):
    """T5 — a minted-but-never-persisted key materializes WITH its key: a freshly
    materialized row must not be born with the same NULL-key wound it exists to heal."""
    resp = _resume(session_id=MINTED_KEY)

    _assert_resumed_ok(resp, MINTED_KEY)
    row = real_db.get_session(MINTED_KEY)
    assert row is not None
    assert row["session_key"] == MINTED_KEY
