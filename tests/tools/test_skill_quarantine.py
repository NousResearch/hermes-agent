"""Tests for skill quarantine — ledger-based quarantine status tracking (Phase 4)."""

import pytest
from pathlib import Path

from tools.skill_quarantine import (
    quarantine_skill,
    promote_skill,
    reject_skill,
    is_quarantined,
    get_quarantine_info,
    list_quarantined,
)


class TestQuarantineLifecycle:
    """Basic quarantine → promote → clear cycle."""

    def test_quarantine_blocks_grant(self):
        """A quarantined skill should return True from is_quarantined."""
        quarantine_skill("ext-malware-scanner", source="github",
                         identifier="evilcorp/skills/malware-scanner")
        assert is_quarantined("ext-malware-scanner") is True

    def test_promotion_clears_quarantine(self):
        """Promoting a skill should clear its quarantine status."""
        quarantine_skill("ext-useful-skill", source="github")
        assert is_quarantined("ext-useful-skill") is True
        promote_skill("ext-useful-skill", "denji", "Reviewed — safe, rewritten natively")
        assert is_quarantined("ext-useful-skill") is False

    def test_rejected_stays_quarantined(self):
        """Rejecting should NOT clear quarantine — it stays blocked."""
        quarantine_skill("ext-dangerous-skill", source="url")
        reject_skill("ext-dangerous-skill", "skill-research", "Prompt injection found")
        assert is_quarantined("ext-dangerous-skill") is True

    def test_review_out_of_order_cleared(self):
        """A reviewed event supersedes a quarantine even if events are interleaved."""
        quarantine_skill("ext-alpha", source="github")
        quarantine_skill("ext-beta", source="github")  # interleave
        promote_skill("ext-alpha", "denji")
        assert is_quarantined("ext-alpha") is False
        assert is_quarantined("ext-beta") is True

    def test_unknown_skill_not_quarantined(self):
        """A skill never recorded should not be quarantined."""
        assert is_quarantined("no-such-skill") is False

    def test_quarantine_info_for_quarantined(self):
        """get_quarantine_info returns structured data for active quarantine."""
        quarantine_skill("ext-info-test", source="skills-sh",
                         identifier="someone/skills-sh/some-skill",
                         review_required_by="skill-research")
        info = get_quarantine_info("ext-info-test")
        assert info is not None
        assert info["skill_name"] == "ext-info-test"
        assert info["source"] == "skills-sh"
        assert info["status"] == "skill.quarantined"
        assert info["review_required_by"] == "skill-research"

    def test_quarantine_info_returns_none_for_cleared(self):
        """After promotion, get_quarantine_info should return None."""
        quarantine_skill("ext-cleared-skill", source="github")
        promote_skill("ext-cleared-skill", "denji")
        assert get_quarantine_info("ext-cleared-skill") is None

    def test_list_quarantined(self):
        """list_quarantined returns only unresolved quarantines."""
        # Create a few quarantined skills
        quarantine_skill("ext-q-list-1", source="github")
        quarantine_skill("ext-q-list-2", source="url")
        # Promote one
        promote_skill("ext-q-list-1", "denji")
        # Reject another
        reject_skill("ext-q-list-3-shouldnt-exist", "denji", "no")  # doesn't exist yet
        quarantine_skill("still-quarantined", source="skills-sh")

        quarantined = list_quarantined()
        names = {q["skill_name"] for q in quarantined}
        assert "ext-q-list-2" in names
        assert "still-quarantined" in names
        assert "ext-q-list-1" not in names  # promoted

    def test_quarantine_records_event_id(self):
        """quarantine_skill returns a valid event_id."""
        eid = quarantine_skill("ext-eid-test", source="url")
        assert eid.startswith("quarantine-")
        assert len(eid) > 20

    def test_promote_records_event_id(self):
        """promote_skill returns a valid event_id."""
        quarantine_skill("ext-promote-eid", source="github")
        eid = promote_skill("ext-promote-eid", "skill-research", "clean")
        assert eid.startswith("review-")

    def test_reject_records_event_id(self):
        """reject_skill returns a valid event_id."""
        quarantine_skill("ext-reject-eid", source="url")
        eid = reject_skill("ext-reject-eid", "denji", "bad")
        assert eid.startswith("reject-")
