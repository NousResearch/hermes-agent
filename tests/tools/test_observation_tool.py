import json
import sqlite3
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.observation_tool import (
    MAX_LIST_ITEM_LENGTH,
    MAX_SEARCH_TEXT_LENGTH,
    MAX_TEXT_LENGTHS,
    OBSERVATION_SAVE_SCHEMA,
    OBSERVATION_SEARCH_SCHEMA,
    observation_save,
    observation_search,
)


@pytest.fixture
def hermes_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def _json_result(raw: str) -> dict:
    return json.loads(raw)


def _save(**overrides) -> dict:
    payload = {
        "type": "verification",
        "title": "Unique observation title",
        "narrative": "A concise evidence-linked observation narrative.",
        "confidence": "medium",
        "evidence_paths": ["/tmp/evidence.md"],
    }
    payload.update(overrides)
    return _json_result(observation_save(payload))


def _db_path(hermes_home: Path) -> Path:
    return hermes_home / "aion" / "observations" / "aion_observations.db"


def _stored_row(hermes_home: Path, observation_id: int) -> sqlite3.Row:
    conn = sqlite3.connect(_db_path(hermes_home))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM observations WHERE id = ?", (observation_id,)).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def test_observation_save_rejects_invalid_type(hermes_home):
    result = _save(type="bad")

    assert result["status"] == "error"
    assert "invalid type" in result["error"].lower()


def test_observation_save_requires_evidence_for_high_confidence(hermes_home):
    result = _save(confidence="high", evidence_paths=[])

    assert result["status"] == "error"
    assert "evidence" in result["error"].lower()


def test_observation_save_low_confidence_without_evidence_allowed_with_warning(hermes_home):
    result = _save(confidence="low", evidence_paths=[])

    assert result["status"] == "saved"
    assert result["observation"]["confidence"] == "low"
    assert any("evidence" in warning.lower() for warning in result["warnings"])


def test_observation_save_rejects_importance_outside_range(hermes_home):
    for importance in (0, 6):
        result = _save(importance=importance)
        assert result["status"] == "error"
        assert "importance" in result["error"].lower()


def test_observation_save_rejects_oversized_evidence_and_tags(hermes_home):
    oversized = "x" * 5000

    evidence_result = _save(evidence_paths=[oversized])
    tag_result = _save(tags=[oversized])

    assert evidence_result["status"] == "error"
    assert "evidence_paths" in evidence_result["error"]
    assert tag_result["status"] == "error"
    assert "tags" in tag_result["error"]


def test_observation_save_uses_hermes_home(hermes_home):
    result = _save(title="Profile isolation observation")

    expected_db = hermes_home / "aion" / "observations" / "aion_observations.db"
    assert result["status"] == "saved"
    assert Path(result["db_path"]) == expected_db
    assert expected_db.exists()


def test_observation_search_uses_same_profile_db(hermes_home):
    _save(title="Searchable alpha observation")

    result = _json_result(observation_search({"query": "alpha"}))

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["observations"][0]["title"] == "Searchable alpha observation"
    assert Path(result["db_path"]) == hermes_home / "aion" / "observations" / "aion_observations.db"


def test_observation_search_is_read_only_by_default(hermes_home):
    saved = _save(title="Read only search observation")
    obs_id = saved["observation"]["id"]
    before = _stored_row(hermes_home, obs_id)

    result = _json_result(observation_search({"query": "Read"}))
    after = _stored_row(hermes_home, obs_id)

    assert result["status"] == "ok"
    assert after["access_count"] == before["access_count"]
    assert after["updated_at"] == before["updated_at"]


def test_observation_search_on_fresh_profile_does_not_create_database(hermes_home):
    db_path = _db_path(hermes_home)

    result = _json_result(observation_search({"query": "anything"}))

    assert result["status"] == "ok"
    assert result["count"] == 0
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_observation_search_rejects_oversized_query(hermes_home):
    result = _json_result(observation_search({"query": "x" * 1000}))

    assert result["status"] == "error"
    assert "query" in result["error"].lower()


def test_observation_search_filters_by_domain_and_type(hermes_home):
    _save(title="Network router verified", type="verification", domain="TECHNICAL")
    _save(title="Network client decision", type="decision", domain="COMMERCIAL")

    result = _json_result(
        observation_search({"query": "Network", "domain": "TECHNICAL", "type": "verification"})
    )

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["observations"][0]["domain"] == "TECHNICAL"
    assert result["observations"][0]["type"] == "verification"


def test_observation_search_limits_results(hermes_home):
    for idx in range(30):
        _save(title=f"Limit candidate {idx}", narrative="Repeated limit search narrative.")

    result = _json_result(observation_search({"query": "candidate", "limit": 5}))

    assert result["status"] == "ok"
    assert result["count"] == 5
    assert len(result["observations"]) == 5


def test_observation_search_rejects_limit_above_cap(hermes_home):
    result = _json_result(observation_search({"limit": 500}))

    assert result["status"] == "error"
    assert "limit" in result["error"].lower()


def test_observation_search_query_does_not_inject_sql(hermes_home):
    _save(title="Injection resistant observation", narrative="SQL safety evidence.")

    result = _json_result(observation_search({"query": "' OR 1=1 --"}))

    assert result["status"] == "ok"
    assert isinstance(result["observations"], list)


def test_observation_search_rejects_tokenless_non_empty_query(hermes_home):
    _save(title="Tokenless query should not return everything")

    result = _json_result(observation_search({"query": "!!! -- ..."}))

    assert result["status"] == "error"
    assert "search token" in result["error"].lower()


def test_observation_save_schema_declares_runtime_text_caps():
    properties = OBSERVATION_SAVE_SCHEMA["parameters"]["properties"]

    for field, max_length in MAX_TEXT_LENGTHS.items():
        assert properties[field]["maxLength"] == max_length

    assert properties["evidence_paths"]["items"]["maxLength"] == MAX_LIST_ITEM_LENGTH
    assert properties["tags"]["items"]["maxLength"] == MAX_LIST_ITEM_LENGTH


def test_observation_search_schema_declares_runtime_query_cap():
    properties = OBSERVATION_SEARCH_SCHEMA["parameters"]["properties"]

    assert properties["query"]["maxLength"] == MAX_SEARCH_TEXT_LENGTH


def test_observation_tools_are_in_skills_toolset():
    import tools.observation_tool  # noqa: F401 - import registers tools
    from toolsets import resolve_toolset

    tools = resolve_toolset("skills")

    assert "observation_save" in tools
    assert "observation_search" in tools
