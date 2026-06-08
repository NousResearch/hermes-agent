from __future__ import annotations

from wisdom.apply import create_application_proposals
from wisdom.capture import capture_text


def test_application_proposals_are_internal_only(wisdom_db, wisdom_config):
    outcome = capture_text("Investing thought: risk beats thesis confidence.", config=wisdom_config, db=wisdom_db)
    capture = outcome.capture
    assert capture is not None
    apps = create_application_proposals(wisdom_db, capture.id)
    assert {app.application_type for app in apps} == {"investment_rule", "checklist", "decision_rule"}
    assert all(app.status == "proposed" for app in apps)


def test_apply_is_idempotent(wisdom_db, wisdom_config):
    outcome = capture_text("Health note: sleep affects cognition.", config=wisdom_config, db=wisdom_db)
    capture = outcome.capture
    assert capture is not None
    first = create_application_proposals(wisdom_db, capture.id)
    second = create_application_proposals(wisdom_db, capture.id)
    assert [app.id for app in first] == [app.id for app in second]
