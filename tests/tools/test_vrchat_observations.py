from __future__ import annotations

import json

from tools.openclaw import vrchat_observations as observations


def test_build_observation_from_osc_chatbox():
    result = observations.build_observation_from_osc("/chatbox/input", ["hello", True, False])

    assert result["success"] is True
    assert result["observation"]["source"] == "textBox"
    assert result["observation"]["text"] == "hello"
    assert result["observation"]["trust"] == "vrchat_osc"


def test_build_observation_from_osc_blocks_avatar_parameters_by_default():
    result = observations.build_observation_from_osc("/avatar/parameters/Wave", [True])

    assert result["success"] is False
    assert result["observation"] is None
    assert result["ignored"] == "avatar_parameter_observation_disabled"


def test_build_observation_from_osc_allows_avatar_parameters_when_requested():
    result = observations.build_observation_from_osc(
        "/avatar/parameters/Wave",
        [True],
        allow_avatar_parameters=True,
    )

    assert result["success"] is True
    assert result["observation"]["source"] == "system"
    assert "Wave=True" in result["observation"]["text"]


def test_ingest_observations_queues_batch_and_rejects_unknown(tmp_path):
    queue_path = tmp_path / "observations.jsonl"

    result = observations.ingest_observations(
        [
            {"source": "speechToText", "text": "hello"},
            {"source": "visionObservation", "summary": "user waved"},
            {"source": "unknown", "text": "ignored"},
        ],
        queue_path=queue_path,
    )

    assert result["success"] is False
    assert len(result["queued"]) == 2
    assert result["rejected"] == [{"index": "2", "reason": "unsupported_source:unknown"}]
    saved = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert [item["source"] for item in saved] == ["speechToText", "visionObservation"]


def test_observation_queue_status_is_read_only(tmp_path):
    queue_path = tmp_path / "observations.jsonl"
    observations.ingest_observations(
        [
            {"source": "streamComment", "content": "nice"},
            {"source": "operator", "text": "observe"},
        ],
        queue_path=queue_path,
    )

    result = observations.observation_queue_status(queue_path=queue_path, max_preview=1)

    assert result["success"] is True
    assert result["exists"] is True
    assert result["count"] == 2
    assert len(result["preview"]) == 1
    assert queue_path.exists()


def test_parse_jsonl_observation_accepts_wrapped_event():
    result = observations.parse_jsonl_observation(
        json.dumps({"observation": {"source": "visionObservation", "summary": "hands visible"}})
    )

    assert result["success"] is True
    assert result["observation"]["source"] == "visionObservation"
    assert result["observation"]["text"] == "hands visible"


def test_parse_jsonl_observation_rejects_bad_source():
    result = observations.parse_jsonl_observation(json.dumps({"source": "rawOsc", "text": "no"}))

    assert result["success"] is False
    assert result["error"] == "unsupported_source:rawOsc"
