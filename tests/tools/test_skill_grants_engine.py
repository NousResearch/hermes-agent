"""Phase 3: skill grant engine + skill_request tool.

Covers grant_skill safety gates, task-completion revocation, TTL sweep, and the
skill_request tool's library-existence guard. Isolated to a temp ledger.
"""
import time

import pytest

from hermes_cli import profile_activity_ledger as pal
import tools.skill_grants as sg
import tools.skills_tool as st


@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite"
    monkeypatch.setattr(pal, "ledger_db_path", lambda: db)
    monkeypatch.setattr(pal, "_governance_root", lambda: tmp_path)
    return db


def test_grant_safe_skill(temp_ledger):
    d = sg.grant_skill("octacon", "arxiv", "t_1", "need refs")
    assert d["granted"] is True and d["decision"] == "grant_task_only"
    assert sg.has_active_grant("octacon", "arxiv") is True


def test_grant_requires_task_id(temp_ledger):
    d = sg.grant_skill("octacon", "arxiv", "", "no task")
    assert d["granted"] is False and "task_id" in d["reason"]
    # nothing recorded for an unrevocable grant
    assert sg.has_active_grant("octacon", "arxiv") is False


def test_never_grant_denied(temp_ledger):
    d = sg.grant_skill("octacon", "governance", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]
    assert sg.has_active_grant("octacon", "governance") is False


def test_frequency_cap_denies(temp_ledger):
    for i in range(sg.FREQUENCY_LIMIT):
        sg.grant_skill("octacon", "arxiv", f"t_{i}")
    d = sg.grant_skill("octacon", "arxiv", "t_over")
    assert d["granted"] is False and "frequency" in d["reason"]


def test_revoke_grants_for_task(temp_ledger):
    sg.grant_skill("octacon", "arxiv", "t_done")
    sg.grant_skill("octacon", "maps", "t_done")
    sg.grant_skill("octacon", "blogwatcher", "t_other")
    n = sg.revoke_grants_for_task("t_done", "completed")
    assert n == 2
    assert sg.has_active_grant("octacon", "arxiv") is False
    assert sg.has_active_grant("octacon", "maps") is False
    # a grant on a different task is untouched
    assert sg.has_active_grant("octacon", "blogwatcher") is True


def test_ttl_sweep_revokes_old_grants(temp_ledger, monkeypatch):
    # Borrow stamped 48h ago
    old = int(time.time() - 48 * 3600)
    pal.append_event(source="t", event_type="skill.borrowed", target_profile="remii",
                     object_type="skill", object_id="arxiv", event_id="old1",
                     occurred_at=old, payload={"task_id": "t_x"})
    sg.grant_skill("remii", "maps", "t_y")  # fresh, must survive
    revoked = sg.sweep_expired_grants(ttl_hours=24)
    assert revoked == 1
    assert sg.has_active_grant("remii", "arxiv") is False
    assert sg.has_active_grant("remii", "maps") is True


def test_revoke_by_event_id(temp_ledger):
    d = sg.grant_skill("octacon", "arxiv", "t_1")
    res = sg.revoke_by_event_id(d["event_id"])
    assert res["status"] == "revoked"
    assert sg.revoke_by_event_id("nope").get("error")


def test_skill_request_rejects_unknown_skill(temp_ledger, monkeypatch):
    monkeypatch.setattr(st, "_skill_exists", lambda name: False)
    monkeypatch.setattr(st, "_current_profile", lambda: "octacon")
    import json

    out = json.loads(st.skill_request("nonexistent", "t_1", "because"))
    assert out["success"] is False and out["granted"] is False
    assert "not in the library" in out["error"]


def test_skill_request_grants_existing_skill(temp_ledger, monkeypatch):
    monkeypatch.setattr(st, "_skill_exists", lambda name: True)
    monkeypatch.setattr(st, "_current_profile", lambda: "octacon")
    import json

    out = json.loads(st.skill_request("arxiv", "t_1", "need it"))
    assert out["success"] is True and out["granted"] is True
    assert sg.has_active_grant("octacon", "arxiv") is True


# ---------------------------------------------------------------------------
# C006: bounded-query contract for has_active_grant + governance whitelist.
# ---------------------------------------------------------------------------

def test_bounded_grant_old_grant_later_revoke_within_window(temp_ledger):
    """C006-F1: a grant borrowed 3d ago and revoked 2d ago must be inactive
    under the default bounded lookback (both events within the 7-day window)."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="skill.borrowed", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="b-old-rev",
        occurred_at=now - 3 * 86400, payload={"task_id": "t_old"},
    )
    pal.append_event(
        source="t", event_type="skill.revoked", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="r-old-rev",
        occurred_at=now - 2 * 86400,
        payload={"borrow_event_id": "b-old-rev", "task_result": "completed"},
    )
    assert sg.has_active_grant("octacon", "arxiv") is False


def test_bounded_grant_old_still_active_within_window(temp_ledger):
    """C006-F1: a grant borrowed 3d ago with no revocation must remain active
    under the default bounded lookback (borrow within the 7-day window)."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="skill.borrowed", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="b-old-active",
        occurred_at=now - 3 * 86400, payload={"task_id": "t_old"},
    )
    assert sg.has_active_grant("octacon", "arxiv") is True


def test_bounded_grant_crosses_boundary_revoked(temp_ledger):
    """C006-F1: a grant borrowed 8d ago (outside the 7-day window) and revoked
    6d ago (inside) — the bounded query misses the borrow and correctly
    returns False (the grant was revoked)."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="skill.borrowed", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="b-cross",
        occurred_at=now - 8 * 86400, payload={"task_id": "t_cross"},
    )
    pal.append_event(
        source="t", event_type="skill.revoked", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="r-cross",
        occurred_at=now - 6 * 86400,
        payload={"borrow_event_id": "b-cross", "task_result": "completed"},
    )
    assert sg.has_active_grant("octacon", "arxiv") is False


def test_bounded_grant_boundary_ordering(temp_ledger):
    """C006-F1: events exactly at the since boundary (>=) are included."""
    import time as _t
    now = int(_t.time())
    boundary = now - sg._GRANT_SCAN_WINDOW_SECONDS
    pal.append_event(
        source="t", event_type="skill.borrowed", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="b-bnd",
        occurred_at=boundary, payload={"task_id": "t_bnd"},
    )
    assert sg.has_active_grant(
        "octacon", "arxiv", since=boundary
    ) is True
    pal.append_event(
        source="t", event_type="skill.revoked", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="r-bnd",
        occurred_at=boundary,
        payload={"borrow_event_id": "b-bnd", "task_result": "completed"},
    )
    assert sg.has_active_grant(
        "octacon", "arxiv", since=boundary
    ) is False


def test_bounded_grant_task_id_scoping(temp_ledger):
    """C006-F1: revoking grants for one task must not affect grants for a
    different task under the bounded query."""
    sg.grant_skill("octacon", "arxiv", "t_a")
    sg.grant_skill("octacon", "maps", "t_b")
    n = sg.revoke_grants_for_task("t_a", "completed")
    assert n == 1
    assert sg.has_active_grant("octacon", "arxiv") is False
    assert sg.has_active_grant("octacon", "maps") is True


def test_bounded_grant_explicit_since(temp_ledger):
    """C006-F1: an explicit since parameter bounds the query."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="skill.borrowed", target_profile="octacon",
        object_type="skill", object_id="arxiv", event_id="b-explicit",
        occurred_at=now - 10 * 86400, payload={"task_id": "t_explicit"},
    )
    assert sg.has_active_grant(
        "octacon", "arxiv", since=now - 5 * 86400
    ) is False
    assert sg.has_active_grant(
        "octacon", "arxiv", since=now - 15 * 86400
    ) is True


def test_governance_whitelist_allows_denji(temp_ledger):
    """C006-F5: Denji (governance profile) is allowed to borrow
    profile-config-mutator and soyl-editor."""
    d = sg.grant_skill("denji", "profile-config-mutator", "t_1")
    assert d["granted"] is True


def test_governance_whitelist_denies_non_denji(temp_ledger):
    """C006-F5: a non-governance profile is denied profile-config-mutator
    even though the whitelist allows it for denji."""
    d = sg.grant_skill("octacon", "profile-config-mutator", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]


def test_governance_whitelist_denies_unlisted_governance_skill(temp_ledger):
    """C006-F5: denji is still denied NEVER_GRANT skills not in its
    whitelist (e.g. governance)."""
    d = sg.grant_skill("denji", "governance", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]
