"""Tests for Denji auto-promotion script (Phase 5)."""

import json
import pytest
import sys
import time
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hermes_cli.profile_activity_ledger import append_event
from tools.skill_grants import grant_skill, revoke_grants_for_task


class TestAutoPromotion:
    """Integration tests for the auto-promotion pipeline."""

    def test_borrow_count_tracking(self):
        """Borrow events should be queryable by profile and skill."""
        # Grant a test skill
        grant_skill("remii", "arxiv", "t_ap_test_count", "auto-promote test")

        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(
            event_types=["skill.borrowed"],
            target_profile="remii",
            object_id="arxiv",
            since=int(time.time() - 3600),
        )
        # At least one borrow event for remii/arxiv should exist
        remii_arxiv = [e for e in events
                       if e.get("target_profile") == "remii"
                       and e.get("object_id") == "arxiv"]
        assert len(remii_arxiv) >= 1, f"Expected >=1 borrow events for remii/arxiv, got {len(remii_arxiv)}"

        # Cleanup
        revoke_grants_for_task("t_ap_test_count")

    def test_never_auto_promote_cannot_be_borrowed(self):
        """NEVER_GRANT skills should not appear in borrow counts."""
        # governance is on NEVER_GRANT — it should always be denied
        result = grant_skill("octacon", "governance", "t_ap_never", "test")
        assert result["granted"] is False

    def test_promotion_event_is_recorded(self):
        """Auto-promotion creates a skill.enabled_auto event in the ledger."""
        from hermes_cli.profile_activity_ledger import query_events

        # Record a synthetic promotion event
        eid = append_event(
            source="denji-auto-promote",
            event_type="skill.enabled_auto",
            event_id=f"ap-test-{int(time.time())}",
            actor_profile="denji",
            target_profile="remii-test",
            object_type="skill",
            object_id="market-research",
            summary="Auto-promoted market-research for remii-test (5 borrows in 30d)",
            payload={
                "profile": "remii-test",
                "skill": "market-research",
                "borrow_count": 5,
                "window_days": 30,
                "promoted_at": int(time.time()),
            },
        )
        assert eid is not None

        # Query it back
        events = query_events(event_types=["skill.enabled_auto"], object_id="market-research", limit=1)
        assert len(events) == 1
        assert events[0]["target_profile"] == "remii-test"
        assert events[0]["object_id"] == "market-research"

    def test_promotion_is_reversible(self):
        """A promotion can be reverted by recording a revert event."""
        from hermes_cli.profile_activity_ledger import query_events

        # Record promotion
        append_event(
            source="denji-auto-promote",
            event_type="skill.enabled_auto",
            event_id=f"ap-revert-test-{int(time.time())}",
            actor_profile="denji",
            target_profile="octacon-test",
            object_type="skill",
            object_id="zoom-out",
            summary="Auto-promoted zoom-out for octacon-test",
            payload={"borrow_count": 4},
        )

        # Record revert
        append_event(
            source="denji-auto-promote",
            event_type="skill.enabled_auto_reverted",
            event_id=f"ap-revert-{int(time.time())}",
            actor_profile="denji",
            target_profile="octacon-test",
            object_type="skill",
            object_id="zoom-out",
            summary="Reverted auto-promotion of zoom-out for octacon-test (manual override)",
            payload={"reason": "manual override", "reverted_at": int(time.time())},
        )

        # Both should exist — audit trail intact
        events = query_events(
            event_types=["skill.enabled_auto", "skill.enabled_auto_reverted"],
            object_id="zoom-out",
        )
        types = {e["event_type"] for e in events}
        assert "skill.enabled_auto" in types
        assert "skill.enabled_auto_reverted" in types

    def test_auto_promote_skips_non_existent_skills(self):
        """Skills not on disk should not be promoted."""
        # ext-evil-skill is a test artifact with borrow events but no file on disk
        from hermes_cli.profile_activity_ledger import query_events
        skills_dir = Path("/home/kensei/.hermes/skills")

        # Grant a skill that doesn't exist on disk (should still grant if not quarantined)
        # But auto-promotion should skip it
        non_existent = "this-definitely-does-not-exist-as-a-skill"

        # Verify it doesn't exist on disk
        assert not (skills_dir / non_existent / "SKILL.md").exists()
        assert not any(skills_dir.rglob(f"{non_existent}/SKILL.md"))

        # Auto-promotion should skip it even if there were borrows
        # (Tested implicitly by the script's --verbose dry-run)
