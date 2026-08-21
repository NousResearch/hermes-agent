"""Tests for tools.wisdom_share_pass (PRD 1, M0)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.wisdom_share_pass import (
    RECENCY_FULL_DAYS,
    RECENCY_ZERO_DAYS,
    SCORE_FLOOR_USES,
    TOP_N,
    _recency_weight,
    decline_candidate,
    load_state,
    maybe_run_share_pass,
    run_share_pass,
    save_state,
    score_skill,
)

NOW = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def _iso(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _rec(uses=0, patches=0, last=None):
    return {
        "use_count": uses,
        "patch_count": patches,
        "last_used_at": last,
        "view_count": 0,
        "last_viewed_at": None,
        "last_patched_at": None,
    }


# ---------------------------------------------------------------------------
# Recency weight
# ---------------------------------------------------------------------------


class TestRecencyWeight:
    def test_today_full_weight(self):
        assert _recency_weight(_iso(0), NOW) == 1.0

    def test_boundary_full(self):
        assert _recency_weight(_iso(RECENCY_FULL_DAYS), NOW) == 1.0

    def test_boundary_zero(self):
        assert _recency_weight(_iso(RECENCY_ZERO_DAYS), NOW) == 0.0

    def test_beyond_zero(self):
        assert _recency_weight(_iso(RECENCY_ZERO_DAYS + 30), NOW) == 0.0

    def test_midpoint(self):
        mid = (RECENCY_FULL_DAYS + RECENCY_ZERO_DAYS) // 2
        w = _recency_weight(_iso(mid), NOW)
        assert 0.4 < w < 0.6

    def test_none(self):
        assert _recency_weight(None, NOW) == 0.0

    def test_garbage(self):
        assert _recency_weight("not-a-date", NOW) == 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoreSkill:
    def test_uses_only(self):
        rec = _rec(uses=10, last=_iso(1))
        assert score_skill(rec, NOW) == pytest.approx(10.0)

    def test_patches_add(self):
        rec = _rec(uses=10, patches=4, last=_iso(1))
        assert score_skill(rec, NOW) == pytest.approx(12.0)

    def test_old_skill_zero_recency(self):
        rec = _rec(uses=100, last=_iso(90))
        # recency=0 → uses contribute 0; only patches count
        assert score_skill(rec, NOW) == pytest.approx(0.0)

    def test_no_uses_no_patches(self):
        assert score_skill(_rec(), NOW) == 0.0


# ---------------------------------------------------------------------------
# The pass (with mocked usage data)
# ---------------------------------------------------------------------------


def _mock_usage(tmp_path, monkeypatch, data, state=None):
    """Point load_usage and the state file at test fixtures."""
    monkeypatch.setattr(
        "tools.wisdom_share_pass.load_state", lambda: state or _default_state()
    )
    saved = []

    def _save(d):
        saved.append(d)

    monkeypatch.setattr("tools.wisdom_share_pass.save_state", _save)
    monkeypatch.setattr(
        "tools.skill_usage.load_usage", lambda: data
    )
    # provenance: everything is agent-authored in tests
    monkeypatch.setattr("tools.skill_usage.provenance", lambda name: "agent")
    return saved


def _default_state():
    return {"last_run_at": None, "declined": [], "last_candidates": [], "run_count": 0}


class TestRunSharePass:
    def test_empty_usage(self, tmp_path, monkeypatch):
        _mock_usage(tmp_path, monkeypatch, {})
        result = run_share_pass(now=NOW)
        assert result["ok"] is True
        assert result["candidates"] == []
        assert result["skipped_below_floor"] == 0

    def test_below_floor_excluded(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=2, last=_iso(1))}  # 2 < SCORE_FLOOR_USES
        _mock_usage(tmp_path, monkeypatch, data)
        result = run_share_pass(now=NOW)
        assert result["candidates"] == []
        assert result["skipped_below_floor"] == 1

    def test_at_floor_included(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=SCORE_FLOOR_USES, last=_iso(1))}
        _mock_usage(tmp_path, monkeypatch, data)
        result = run_share_pass(now=NOW)
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["name"] == "skill-a"

    def test_top_n_cap(self, tmp_path, monkeypatch):
        data = {
            f"skill-{i}": _rec(uses=100 - i, last=_iso(1)) for i in range(TOP_N + 3)
        }
        _mock_usage(tmp_path, monkeypatch, data)
        result = run_share_pass(now=NOW)
        assert len(result["candidates"]) == TOP_N

    def test_ranking_order(self, tmp_path, monkeypatch):
        data = {
            "low": _rec(uses=5, last=_iso(1)),
            "high": _rec(uses=50, last=_iso(1)),
            "mid": _rec(uses=20, last=_iso(1)),
        }
        _mock_usage(tmp_path, monkeypatch, data)
        result = run_share_pass(now=NOW)
        names = [c["name"] for c in result["candidates"]]
        assert names == ["high", "mid", "low"]

    def test_declined_excluded(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=50, last=_iso(1))}
        state = _default_state()
        state["declined"] = ["skill-a"]
        _mock_usage(tmp_path, monkeypatch, data, state)
        result = run_share_pass(now=NOW)
        assert result["candidates"] == []
        assert result["skipped_declined"] == ["skill-a"]

    def test_evidence_line_present(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, patches=2, last=_iso(3))}
        _mock_usage(tmp_path, monkeypatch, data)
        result = run_share_pass(now=NOW)
        ev = result["candidates"][0]["evidence"]
        assert "used 10 times" in ev
        assert "patched 2 times" in ev
        assert "3 days ago" in ev

    def test_state_persisted(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        saved = _mock_usage(tmp_path, monkeypatch, data)
        run_share_pass(now=NOW)
        assert len(saved) == 1
        assert saved[0]["last_candidates"] == ["skill-a"]
        assert saved[0]["run_count"] == 1

    def test_never_raises_on_garbage(self, tmp_path, monkeypatch):
        data = {"skill-a": "not-a-dict", "skill-b": _rec(uses=10, last=_iso(1))}
        _mock_usage(tmp_path, monkeypatch, data)
        result = run_share_pass(now=NOW)
        assert result["ok"] is True
        assert len(result["candidates"]) == 1

    def test_dry_run_flag(self, tmp_path, monkeypatch):
        _mock_usage(tmp_path, monkeypatch, {})
        result = run_share_pass(dry_run=True, now=NOW)
        assert result["dry_run"] is True


# ---------------------------------------------------------------------------
# Scheduling gate
# ---------------------------------------------------------------------------


class TestMaybeRunSharePass:
    def test_first_run_fires(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        _mock_usage(tmp_path, monkeypatch, data)
        result = maybe_run_share_pass(now=NOW)
        assert result is not None
        assert result["ok"] is True

    def test_second_run_within_interval_skipped(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        state = _default_state()
        state["last_run_at"] = NOW.isoformat()
        _mock_usage(tmp_path, monkeypatch, data, state)
        # 1 hour later — within the 168h interval
        result = maybe_run_share_pass(now=NOW + timedelta(hours=1))
        assert result is None

    def test_run_after_interval(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        state = _default_state()
        state["last_run_at"] = (NOW - timedelta(hours=169)).isoformat()
        _mock_usage(tmp_path, monkeypatch, data, state)
        result = maybe_run_share_pass(now=NOW)
        assert result is not None
        assert result["ok"] is True

    def test_corrupt_timestamp_runs_anyway(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        state = _default_state()
        state["last_run_at"] = "not-a-date"
        _mock_usage(tmp_path, monkeypatch, data, state)
        result = maybe_run_share_pass(now=NOW)
        assert result is not None


# ---------------------------------------------------------------------------
# Decline persistence
# ---------------------------------------------------------------------------


class TestDecline:
    def test_decline_persists(self, tmp_path, monkeypatch):
        saved_states = []
        monkeypatch.setattr(
            "tools.wisdom_share_pass.load_state", lambda: _default_state()
        )
        monkeypatch.setattr(
            "tools.wisdom_share_pass.save_state",
            lambda d: saved_states.append(d),
        )
        decline_candidate("skill-x")
        assert saved_states[0]["declined"] == ["skill-x"]

    def test_decline_idempotent(self, tmp_path, monkeypatch):
        state = _default_state()
        state["declined"] = ["skill-x"]
        saved_states = []
        monkeypatch.setattr("tools.wisdom_share_pass.load_state", lambda: state)
        monkeypatch.setattr(
            "tools.wisdom_share_pass.save_state",
            lambda d: saved_states.append(d),
        )
        decline_candidate("skill-x")
        assert len(saved_states) == 0  # no save when already declined


# ---------------------------------------------------------------------------
# M1 calibration: idle cap + never-tracked
# ---------------------------------------------------------------------------


class TestIdleCap:
    def test_active_skill_no_cap(self):
        rec = _rec(uses=100, last=_iso(10))
        assert score_skill(rec, NOW) == pytest.approx(100.0)

    def test_idle_skill_capped(self):
        rec = _rec(uses=100, last=_iso(45))
        # recency=0 at 45 days (past RECENCY_ZERO_DAYS=60? no, 45 < 60 so recency > 0)
        # But idle cap applies (45 > IDLE_DAYS_THRESHOLD=30)
        s = score_skill(rec, NOW)
        assert s < 50.0  # capped

    def test_idle_boundary(self):
        rec = _rec(uses=100, last=_iso(30))
        # Exactly at threshold — no cap
        s = score_skill(rec, NOW)
        assert s > 0

    def test_patches_not_capped(self):
        rec = _rec(uses=100, patches=20, last=_iso(45))
        s = score_skill(rec, NOW)
        # Patches contribute fully regardless of idle cap
        assert s >= 20 * 0.5


class TestNeverTracked:
    def test_untracked_listed(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        _mock_usage(tmp_path, monkeypatch, data)
        monkeypatch.setattr(
            "tools.skill_usage.list_agent_created_skill_names",
            lambda: ["skill-a", "skill-b", "skill-c"],
        )
        result = run_share_pass(now=NOW)
        assert result["never_tracked"] == ["skill-b", "skill-c"]

    def test_declined_untracked_excluded(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        state = _default_state()
        state["declined"] = ["skill-b"]
        _mock_usage(tmp_path, monkeypatch, data, state)
        monkeypatch.setattr(
            "tools.skill_usage.list_agent_created_skill_names",
            lambda: ["skill-a", "skill-b", "skill-c"],
        )
        result = run_share_pass(now=NOW)
        assert result["never_tracked"] == ["skill-c"]

    def test_all_tracked_empty_list(self, tmp_path, monkeypatch):
        data = {"skill-a": _rec(uses=10, last=_iso(1))}
        _mock_usage(tmp_path, monkeypatch, data)
        monkeypatch.setattr(
            "tools.skill_usage.list_agent_created_skill_names",
            lambda: ["skill-a"],
        )
        result = run_share_pass(now=NOW)
        assert result["never_tracked"] == []
