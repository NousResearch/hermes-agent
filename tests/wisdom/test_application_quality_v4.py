from __future__ import annotations

from pathlib import Path

import yaml

from tools import wisdom_tool
from tools.registry import registry
from wisdom.capture import capture_text
from wisdom.review import score_capture
from wisdom.service import apply


def test_application_quality_fixture_cases_are_domain_specific(wisdom_db, wisdom_config, wisdom_home):
    cases = _load_cases()
    old_productivity_db = wisdom_home / "productivity" / "productivity.db"

    for case in cases:
        outcome = capture_text(case["text"], config=wisdom_config, db=wisdom_db)
        assert outcome.status == "captured"
        capture = outcome.capture
        assert capture is not None
        assert wisdom_db.get_capture(capture.id).original_text == case["text"]
        assert capture.category == case["expected_category"]
        assert capture.source_type == case["expected_source_type"]

        for key, value in (case.get("expected_metadata") or {}).items():
            assert capture.metadata.get(key) == value

        quality = score_capture(capture)
        if "minimum_quality" in case:
            assert quality.overall >= float(case["minimum_quality"])
        if "maximum_quality" in case:
            assert quality.overall <= float(case["maximum_quality"])

        applications = apply(capture.id, config=wisdom_config, db=wisdom_db)
        by_type = {application.application_type: application for application in applications}
        assert set(by_type) == set(case["expected_application_types"])

        for application_type, phrases in (case.get("required_phrases") or {}).items():
            body = by_type[application_type].body.lower()
            for phrase in phrases:
                assert phrase.lower() in body

    assert not old_productivity_db.exists()


def test_wisdom_capture_payload_exposes_source_context_metadata(wisdom_db, wisdom_config):
    text = (
        "Podcast note: Acquired - Costco episode.\n"
        "Context: trust and low-price positioning.\n"
        "Costco creates trust by staying aligned with members."
    )
    outcome = capture_text(text, config=wisdom_config, db=wisdom_db)
    assert outcome.capture is not None

    applications = apply(outcome.capture.id, config=wisdom_config, db=wisdom_db)
    assert any(application.application_type == "client_language" for application in applications)

    stored = wisdom_db.get_capture(outcome.capture.id)
    assert stored is not None
    assert stored.original_text == text
    assert stored.source_type == "podcast"
    assert stored.metadata["source"] == "Acquired - Costco episode"
    assert stored.metadata["context"] == "trust and low-price positioning"


def test_wisdom_tool_descriptions_cover_v4_natural_requests():
    assert wisdom_tool.WISDOM_TOOL_NAMES
    descriptions = {
        name: registry.get_entry(name).schema["description"].lower()
        for name in wisdom_tool.WISDOM_TOOL_NAMES
    }

    capture_description = descriptions["wisdom_capture"]
    for phrase in (
        "remember this",
        "save this",
        "podcast note",
        "book note",
        "investing thought",
        "source:",
        "context:",
    ):
        assert phrase in capture_description

    search_description = descriptions["wisdom_search"]
    for phrase in ("find that idea", "what have i said about"):
        assert phrase in search_description

    assert "show exact wording" in descriptions["wisdom_original"]

    apply_description = descriptions["wisdom_apply"]
    for phrase in (
        "turn that into client language",
        "make this a checklist",
        "make this an investment rule",
        "turn this into a health experiment",
        "make this a decision rule",
        "apply this to x10x",
    ):
        assert phrase in apply_description

    assert "what should i review" in descriptions["wisdom_review"]
    assert "accept that" in descriptions["wisdom_accept"]
    assert "dismiss that" in descriptions["wisdom_dismiss"]


def _load_cases() -> list[dict]:
    fixture = Path(__file__).parent / "fixtures" / "application_quality_cases.yaml"
    data = yaml.safe_load(fixture.read_text())
    assert isinstance(data, list)
    return data
