from __future__ import annotations

import json

from tools.openclaw import vrchat_autonomy as autonomy
from tools.openclaw import vrchat_conversation


def _write_profile(tmp_path, **overrides):
    profile = {
        **autonomy._default_profile(),
        "enabled": True,
        "mode": "private_test",
        "dry_run": True,
        "allow_voice": True,
        "allow_chatbox": True,
        "allow_movement": False,
        "audio_output_device": "CABLE Input",
        "output_device": "CABLE Input",
    }
    profile.update(overrides)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile_path


def test_conversation_dry_run_plans_chatbox_voice_and_neuro_route(tmp_path):
    profile_path = _write_profile(tmp_path)

    result = vrchat_conversation.run_multimodal_conversation_dry_run(
        profile_path=profile_path,
    )

    assert result["success"] is True
    assert result["persisted_observations"] is False
    assert result["has_chatbox"] is True
    assert result["has_voice"] is True
    assert result["planned_kinds"] == ["chatbox", "voice"]
    assert result["neuro_route"]["success"] is True
    assert result["neuro_route"]["turn"]["dry_run"] is True
    assert result["turn"]["execution_results"] == []
    assert result["safety"]["actuation_performed"] is False
    assert result["safety"]["speech_played"] is False


def test_conversation_dry_run_can_persist_observations(tmp_path):
    profile_path = _write_profile(tmp_path)
    queue_path = tmp_path / "observations.jsonl"

    result = vrchat_conversation.run_multimodal_conversation_dry_run(
        profile_path=profile_path,
        persist_observations=True,
        queue_path=queue_path,
    )

    assert result["success"] is True
    assert result["ingestion"]["persisted"] is True
    assert queue_path.exists()
    assert len(queue_path.read_text(encoding="utf-8").splitlines()) == 4


def test_conversation_dry_run_blocks_when_voice_disabled(tmp_path):
    profile_path = _write_profile(tmp_path, allow_voice=False)

    result = vrchat_conversation.run_multimodal_conversation_dry_run(
        profile_path=profile_path,
    )

    assert result["success"] is False
    assert "voice_plan_missing" in result["blockers"]
    assert result["safety"]["actuation_performed"] is False
