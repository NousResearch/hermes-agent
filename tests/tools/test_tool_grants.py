"""Phase 4: tool grant engine + tool_request tool.

Covers grant_tool safety gates, task-completion revocation, and the
tool_request tool's registry-existence guard. Mirrors
tests/tools/test_skill_grants_engine.py. Isolated to a temp ledger.
"""
import time

import pytest

from hermes_cli import profile_activity_ledger as pal
import tools.tool_grants as tg
import tools.skills_tool as st


@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite"
    monkeypatch.setattr(pal, "ledger_db_path", lambda: db)
    monkeypatch.setattr(pal, "_governance_root", lambda: tmp_path)
    return db


def test_grant_safe_tool(temp_ledger):
    d = tg.grant_tool("octacon", "web_search", "t_1", "need to look something up")
    assert d["granted"] is True and d["decision"] == "grant_task_only"
    assert tg.has_active_grant("octacon", "web_search") is True


def test_grant_requires_task_id(temp_ledger):
    d = tg.grant_tool("octacon", "web_search", "", "no task")
    assert d["granted"] is False and "task_id" in d["reason"]
    assert tg.has_active_grant("octacon", "web_search") is False


def test_never_grant_curated_name_denied(temp_ledger):
    d = tg.grant_tool("octacon", "kanban_profile_edit", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]
    assert tg.has_active_grant("octacon", "kanban_profile_edit") is False


def test_never_grant_terminal_toolset_denied(temp_ledger):
    d = tg.grant_tool("octacon", "terminal", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]
    assert tg.has_active_grant("octacon", "terminal") is False


def test_never_grant_process_denied(temp_ledger):
    # process (the other terminal-toolset tool) must be denied via the
    # hardcoded floor even if toolset resolution is unavailable.
    d = tg.grant_tool("octacon", "process", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]


def test_never_grant_floor_holds_without_toolset_resolution(temp_ledger, monkeypatch):
    # Simulate toolsets import/resolution failure: the floor must still deny
    # arbitrary command execution.
    monkeypatch.setattr(tg, "_effective_never_grant", lambda: set(tg.NEVER_GRANT_TOOLS))
    d = tg.grant_tool("octacon", "terminal", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]


def test_unknown_tool_denied(temp_ledger):
    d = tg.grant_tool("octacon", "this_tool_does_not_exist", "t_1")
    assert d["granted"] is False
    assert "not a registered tool" in d["reason"]


def test_frequency_cap_denies(temp_ledger):
    for i in range(tg.FREQUENCY_LIMIT):
        tg.grant_tool("octacon", "web_search", f"t_{i}")
    d = tg.grant_tool("octacon", "web_search", "t_over")
    assert d["granted"] is False and "frequency" in d["reason"]


def test_revoke_grants_for_task(temp_ledger):
    tg.grant_tool("octacon", "web_search", "t_done")
    tg.grant_tool("octacon", "web_extract", "t_done")
    tg.grant_tool("octacon", "memory", "t_other")
    n = tg.revoke_grants_for_task("t_done", "completed")
    assert n == 2
    assert tg.has_active_grant("octacon", "web_search") is False
    assert tg.has_active_grant("octacon", "web_extract") is False
    # a grant on a different task is untouched
    assert tg.has_active_grant("octacon", "memory") is True


def test_ttl_sweep_revokes_old_grants(temp_ledger, monkeypatch):
    old = int(time.time() - 48 * 3600)
    pal.append_event(source="t", event_type="tool.borrowed", target_profile="remii",
                     object_type="tool", object_id="web_search", event_id="old1",
                     occurred_at=old, payload={"task_id": "t_x"})
    tg.grant_tool("remii", "memory", "t_y")  # fresh, must survive
    revoked = tg.sweep_expired_grants(ttl_hours=24)
    assert revoked == 1
    assert tg.has_active_grant("remii", "web_search") is False
    assert tg.has_active_grant("remii", "memory") is True


def test_revoke_by_event_id(temp_ledger):
    d = tg.grant_tool("octacon", "web_search", "t_1")
    res = tg.revoke_by_event_id(d["event_id"])
    assert res["status"] == "revoked"
    assert tg.revoke_by_event_id("nope").get("error")


def test_tool_request_rejects_unknown_tool(temp_ledger, monkeypatch):
    monkeypatch.setattr(st, "_current_profile", lambda: "octacon")
    import json

    out = json.loads(st.tool_request("this_tool_does_not_exist", "t_1", "because"))
    assert out["success"] is True  # the helper itself didn't error
    assert out["granted"] is False
    assert "not a registered tool" in out["reason"]


def test_tool_request_grants_existing_tool(temp_ledger, monkeypatch):
    monkeypatch.setattr(st, "_current_profile", lambda: "octacon")
    import json

    out = json.loads(st.tool_request("web_search", "t_1", "need it"))
    assert out["success"] is True and out["granted"] is True
    assert tg.has_active_grant("octacon", "web_search") is True


# ---------------------------------------------------------------------------
# C006: bounded-query contract for has_active_grant + governance whitelist +
# never-grant toolset resolution logging.
# ---------------------------------------------------------------------------

def test_bounded_grant_old_grant_later_revoke_within_window(temp_ledger):
    """C006-F1: a grant borrowed 3d ago and revoked 2d ago must be inactive
    under the default bounded lookback (both events within the 7-day window)."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="tool.borrowed", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="b-old-rev",
        occurred_at=now - 3 * 86400, payload={"task_id": "t_old"},
    )
    pal.append_event(
        source="t", event_type="tool.revoked", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="r-old-rev",
        occurred_at=now - 2 * 86400,
        payload={"borrow_event_id": "b-old-rev", "task_result": "completed"},
    )
    assert tg.has_active_grant("octacon", "web_search") is False


def test_bounded_grant_old_still_active_within_window(temp_ledger):
    """C006-F1: a grant borrowed 3d ago with no revocation must remain active
    under the default bounded lookback (borrow within the 7-day window)."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="tool.borrowed", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="b-old-active",
        occurred_at=now - 3 * 86400, payload={"task_id": "t_old"},
    )
    assert tg.has_active_grant("octacon", "web_search") is True


def test_bounded_grant_crosses_boundary_revoked(temp_ledger):
    """C006-F1: a grant borrowed 8d ago (outside the 7-day window) and revoked
    6d ago (inside) — the bounded query misses the borrow and correctly
    returns False (the grant was revoked)."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="tool.borrowed", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="b-cross",
        occurred_at=now - 8 * 86400, payload={"task_id": "t_cross"},
    )
    pal.append_event(
        source="t", event_type="tool.revoked", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="r-cross",
        occurred_at=now - 6 * 86400,
        payload={"borrow_event_id": "b-cross", "task_result": "completed"},
    )
    # Borrow is outside window → not found → False.  Correct: it was revoked.
    assert tg.has_active_grant("octacon", "web_search") is False


def test_bounded_grant_boundary_ordering(temp_ledger):
    """C006-F1: events exactly at the since boundary (>=) are included."""
    import time as _t
    now = int(_t.time())
    # Place a borrow exactly at the boundary (now - window).
    boundary = now - tg._GRANT_SCAN_WINDOW_SECONDS
    pal.append_event(
        source="t", event_type="tool.borrowed", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="b-bnd",
        occurred_at=boundary, payload={"task_id": "t_bnd"},
    )
    # Exercise the inclusive ledger boundary directly; passing the same
    # boundary avoids a wall-clock rollover between setup and assertion.
    assert tg.has_active_grant(
        "octacon", "web_search", since=boundary
    ) is True
    # Now revoke it, also at the boundary.
    pal.append_event(
        source="t", event_type="tool.revoked", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="r-bnd",
        occurred_at=boundary,
        payload={"borrow_event_id": "b-bnd", "task_result": "completed"},
    )
    assert tg.has_active_grant(
        "octacon", "web_search", since=boundary
    ) is False


def test_bounded_grant_task_id_scoping(temp_ledger):
    """C006-F1: revoking grants for one task must not affect grants for a
    different task under the bounded query."""
    tg.grant_tool("octacon", "web_search", "t_a")
    tg.grant_tool("octacon", "web_extract", "t_b")
    n = tg.revoke_grants_for_task("t_a", "completed")
    assert n == 1
    assert tg.has_active_grant("octacon", "web_search") is False
    assert tg.has_active_grant("octacon", "web_extract") is True


def test_bounded_grant_explicit_since(temp_ledger):
    """C006-F1: an explicit since parameter bounds the query — events before
    since are excluded."""
    import time as _t
    now = int(_t.time())
    pal.append_event(
        source="t", event_type="tool.borrowed", target_profile="octacon",
        object_type="tool", object_id="web_search", event_id="b-explicit",
        occurred_at=now - 10 * 86400, payload={"task_id": "t_explicit"},
    )
    # With a 5-day lookback, the 10-day-old borrow is missed → False.
    assert tg.has_active_grant(
        "octacon", "web_search", since=now - 5 * 86400
    ) is False
    # With a 15-day lookback, the 10-day-old borrow is found → True.
    assert tg.has_active_grant(
        "octacon", "web_search", since=now - 15 * 86400
    ) is True


def test_effective_never_grant_warns_on_unresolvable_toolset(temp_ledger, monkeypatch, caplog):
    """C006-F4: unresolved never-grant toolset resolution must log WARNING,
    not DEBUG, while the hardcoded deny floor still holds."""
    import logging

    def boom(_name, *args, **kwargs):
        raise RuntimeError("toolset module unavailable")

    monkeypatch.setattr("tools.tool_grants.resolve_toolset", boom, raising=False)
    # Patch at the import site used inside _effective_never_grant.
    import toolsets as _ts_mod
    monkeypatch.setattr(_ts_mod, "resolve_toolset", boom)

    with caplog.at_level(logging.WARNING, logger="tools.tool_grants"):
        deny = tg._effective_never_grant()
    # Hardcoded floor must hold.
    assert "terminal" in deny
    assert "kanban_profile_edit" in deny
    # A WARNING must have been emitted for the failed resolution.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("resolve_toolset" in (r.getMessage() or "") or "toolsets" in (r.getMessage() or "") for r in warnings), \
        "expected a WARNING log for unresolved toolset"


def test_governance_whitelist_allows_denji_tool(temp_ledger, monkeypatch):
    """C006-F5: GOVERNANCE_GRANT_WHITELIST lets a governance profile bypass
    the NEVER_GRANT floor for whitelisted tools."""
    monkeypatch.setattr(tg, "GOVERNANCE_GRANT_WHITELIST",
                        {"denji": {"terminal"}})
    d = tg.grant_tool("denji", "terminal", "t_1")
    assert d["granted"] is True


def test_governance_whitelist_denies_non_governance_profile(temp_ledger, monkeypatch):
    """C006-F5: a non-governance profile is still denied even when the
    whitelist has entries for other profiles."""
    monkeypatch.setattr(tg, "GOVERNANCE_GRANT_WHITELIST",
                        {"denji": {"terminal"}})
    d = tg.grant_tool("octacon", "terminal", "t_1")
    assert d["granted"] is False
    assert "NEVER_GRANT" in d["reason"]
