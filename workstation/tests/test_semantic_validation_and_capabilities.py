from __future__ import annotations

import json
from pathlib import Path
import pytest

from workstation.browser_bridge import browser_save_json
from workstation.browser_readiness import wait_for_condition
from workstation.capabilities import RuntimeCapabilityRegistry
from workstation.semantic_validation import FieldProvenance, SemanticValidator
from workstation.skill_learning import ExperienceLessonCompiler
from workstation.skills import TopicSkillLoader


def test_semantic_validation_and_field_provenance():
    """Scenario 10: Invariant check flags contradiction as SUSPECT; provenance is preserved."""
    validator = SemanticValidator(
        required_fields=["views", "likes"],
        field_ratios=[{"greater": "views", "smaller": "likes"}],
    )

    # Contradictory item: 50 views, 100 likes
    suspect_data = {
        "views": 50,
        "likes": 100,
        "instagram_section": {"section": "metrics", "value": 50},
        "facebook_section": {"section": "metrics", "value": 250},  # Conflicting cross-section!
    }

    result = validator.validate_item(suspect_data)
    assert result["suspect"] is True
    assert any("views (50) < likes (100)" in i for i in result["issues"])
    assert any("Conflicting values across section 'metrics'" in i for i in result["issues"])

    # Valid item with provenance
    prov = FieldProvenance(
        value=115839,
        source="instagram_insights",
        section="Visualizações",
        raw_ref="artifact://tasks/t1/raw_1.json",
        confidence=1.0,
    )
    p_dict = prov.to_dict()
    assert p_dict["value"] == 115839
    assert p_dict["section"] == "Visualizações"
    assert p_dict["raw_ref"] == "artifact://tasks/t1/raw_1.json"


def test_runtime_capability_registry():
    report = RuntimeCapabilityRegistry.inspect_environment()
    assert report.python_version != ""
    assert report.os_name != ""
    assert "sqlite3" in report.installed_modules
    assert report.installed_modules["sqlite3"] is True


def test_topic_skill_loading(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        """# Instagram Reporting Skill
## Overview
General description of the skill.

## Grid Enumeration
How to scroll and enumerate media IDs.
Scroll window.__sc and collect items.

## Parsing
Parse Visualizações and ignore Facebook section.
""",
        encoding="utf-8",
    )

    loader = TopicSkillLoader()
    res = loader.load_topic(skill_file, "grid_enumeration")
    assert res["found"] is True
    assert "Scroll window.__sc" in res["content"]
    # Does not include Parsing or Overview!
    assert "ignore Facebook section" not in res["content"]


def test_semantic_readiness_and_bridge(tmp_path):
    # Test readiness evaluation
    eval_calls = 0

    def mock_evaluate(script: str) -> str:
        nonlocal eval_calls
        eval_calls += 1
        # Becomes ready on second call
        if eval_calls >= 2:
            return json.dumps({"ready": True, "dom_ready": True, "text_length": 300, "selector_found": True, "all_text_found": True, "any_text_found": True, "missing": []})
        return json.dumps({"ready": False, "dom_ready": True, "text_length": 15, "selector_found": False, "all_text_found": False, "any_text_found": False, "missing": ["min_text_length"]})

    ready_res = wait_for_condition(mock_evaluate, min_text_length=250, timeout_seconds=2.0, poll_interval=0.01)
    assert ready_res["ready"] is True
    assert ready_res["text_length"] == 300

    # Test browser_save_json bridge
    dest_file = tmp_path / "workspace" / "grid.json"

    def mock_js_eval(script: str) -> str:
        return json.dumps({"success": True, "data": [{"id": "post_1", "views": 500}]})

    save_res = browser_save_json(mock_js_eval, "window.__sc", dest_file)
    assert save_res["success"] is True
    assert dest_file.exists()
    saved_data = json.loads(dest_file.read_text(encoding="utf-8"))
    assert saved_data[0]["id"] == "post_1"


def test_experience_lesson_compiler(tmp_path):
    compiler = ExperienceLessonCompiler(lessons_dir=tmp_path / "lessons")
    lesson = compiler.record_candidate_lesson(
        skill_name="instagram-reporting",
        issue_description="Facebook cross-post section overwriting Instagram metrics",
        raw_evidence_ref="artifact://tasks/t1/raw_anomalous.json",
        regression_fixture={"raw_text": "Facebook ... Visualizações: 250", "expected_views": 115839},
        proposed_fix_description="Filter out sections matching Facebook cross-post header before parsing",
    )
    assert lesson.lesson_id.startswith("lesson_")
    lessons = compiler.list_candidate_lessons("instagram-reporting")
    assert len(lessons) == 1
    assert lessons[0].issue_description == lesson.issue_description
