from __future__ import annotations

from pathlib import Path

import yaml

from wisdom.capture import capture_text
from wisdom.review import build_review_items, related_captures, score_capture
from wisdom.service import accept, apply, dismiss, related, review


def test_review_quality_fixture_cases_are_deterministic(wisdom_db, wisdom_config):
    cases = _load_cases()
    captures_by_case = {}
    for case in cases:
        outcome = capture_text(case["text"], config=wisdom_config, db=wisdom_db)
        assert outcome.status == "captured"
        assert outcome.capture is not None
        captures_by_case[case["id"]] = outcome.capture
        assert outcome.capture.category == case["expected_category"]
        assert wisdom_db.get_capture(outcome.capture.id).original_text == case["text"]

    for case in cases:
        capture = captures_by_case[case["id"]]
        quality = score_capture(capture)
        if "minimum_quality" in case:
            assert quality.overall >= float(case["minimum_quality"])
        if "maximum_quality" in case:
            assert quality.overall <= float(case["maximum_quality"])

        proposals = apply(capture.id, config=wisdom_config, db=wisdom_db)
        assert {proposal.application_type for proposal in proposals} == set(case["expected_application_types"])

    business_capture = captures_by_case["business_windshield"]
    related_ids = {item.capture.id for item in related(business_capture.id, config=wisdom_config, db=wisdom_db)}
    assert captures_by_case["business_forward_reports"].id in related_ids


def test_review_queue_filters_and_review_actions(wisdom_db, wisdom_config):
    business = _capture("Business idea: Reports need windshields, not rear-view mirrors.", wisdom_db, wisdom_config)
    investing = _capture("Investing thought: I confuse thesis confidence with position sizing.", wisdom_db, wisdom_config)
    noisy = _capture("Note this: random vague thing maybe later.", wisdom_db, wisdom_config)

    assert business.review_status == "unreviewed"

    dismissed = dismiss(noisy.id, config=wisdom_config, db=wisdom_db)
    assert dismissed is not None
    assert dismissed.review_status == "dismissed"

    accepted = accept(investing.id, config=wisdom_config, db=wisdom_db)
    assert accepted is not None
    assert accepted.review_status == "accepted"

    queue = review(config=wisdom_config, db=wisdom_db, limit=10)
    queue_ids = [item.capture.id for item in queue.items]
    assert business.id in queue_ids
    assert investing.id in queue_ids
    assert noisy.id not in queue_ids
    assert queue.counts["needs_review"] == 2

    business_only = review(category="business", config=wisdom_config, db=wisdom_db, limit=10)
    assert [item.capture.category for item in business_only.items] == ["business"]

    high_potential = review(mode="high-potential", config=wisdom_config, db=wisdom_db, limit=10)
    assert high_potential.mode == "high_potential"
    assert all(item.quality.overall >= 0.55 for item in high_potential.items)

    apply(investing.id, config=wisdom_config, db=wisdom_db)
    applied = wisdom_db.get_capture(investing.id)
    assert applied.review_status == "applied"
    assert applied.original_text == "Investing thought: I confuse thesis confidence with position sizing."

    unapplied = review(mode="unapplied", config=wisdom_config, db=wisdom_db, limit=10)
    assert investing.id not in [item.capture.id for item in unapplied.items]

    assert wisdom_db.archive_capture(business.id) is True
    archived_queue = review(config=wisdom_config, db=wisdom_db, limit=10)
    assert business.id not in [item.capture.id for item in archived_queue.items]


def test_related_suggestions_use_overlap_not_embeddings(wisdom_db, wisdom_config):
    first = _capture("Business idea: Clients need windshields, not rear-view mirrors.", wisdom_db, wisdom_config)
    second = _capture("Business idea: Client reports should show the road ahead.", wisdom_db, wisdom_config)
    unrelated = _capture("Health note: Poor sleep changes decision quality.", wisdom_db, wisdom_config)

    suggestions = related_captures(wisdom_db, first.id, limit=5)
    assert any(item.capture.id == second.id for item in suggestions)
    assert all(item.reasons for item in suggestions)
    assert unrelated.id not in {item.capture.id for item in suggestions}


def test_application_quality_templates_are_domain_specific(wisdom_db, wisdom_config):
    investing = _capture("Investing thought: I confuse a good thesis with a good position size.", wisdom_db, wisdom_config)
    investing_apps = apply(investing.id, config=wisdom_config, db=wisdom_db)
    checklist = next(app for app in investing_apps if app.application_type == "checklist")
    assert "liquidity" in checklist.body.lower()
    assert "survivability" in checklist.body.lower()
    assert "forces exit" in checklist.body.lower()

    business = _capture("Business idea: Reports are rear-view mirrors when clients need windshields.", wisdom_db, wisdom_config)
    business_apps = apply(business.id, config=wisdom_config, db=wisdom_db)
    assert "client-facing line" in next(app.body for app in business_apps if app.application_type == "client_language")

    health = _capture("Health note: Poor sleep changes my decision quality.", wisdom_db, wisdom_config)
    health_apps = apply(health.id, config=wisdom_config, db=wisdom_db)
    assert "7 days" in next(app.body for app in health_apps if app.application_type == "health_experiment")

    life = _capture("Life thought: I build systems to avoid making decisions.", wisdom_db, wisdom_config)
    life_apps = apply(life.id, config=wisdom_config, db=wisdom_db)
    assert {app.application_type for app in life_apps} == {"principle", "writing_idea", "decision_rule"}


def test_review_priority_ranks_business_and_investing_above_noise(wisdom_db, wisdom_config):
    noisy = _capture("Note this: random vague thing maybe later.", wisdom_db, wisdom_config)
    business = _capture("Business idea: Reports are rear-view mirrors when clients need windshields.", wisdom_db, wisdom_config)
    investing = _capture("Investing thought: Position sizing must follow survivability, liquidity, and forced-exit risk.", wisdom_db, wisdom_config)

    items = build_review_items(wisdom_db, mode="needs_review", limit=10)
    ranked_ids = [item.capture.id for item in items]
    assert ranked_ids.index(business.id) < ranked_ids.index(noisy.id)
    assert ranked_ids.index(investing.id) < ranked_ids.index(noisy.id)


def _capture(text: str, wisdom_db, wisdom_config):
    outcome = capture_text(text, config=wisdom_config, db=wisdom_db)
    assert outcome.capture is not None
    return outcome.capture


def _load_cases() -> list[dict]:
    fixture = Path(__file__).parent / "fixtures" / "review_quality_cases.yaml"
    data = yaml.safe_load(fixture.read_text())
    assert isinstance(data, list)
    return data
