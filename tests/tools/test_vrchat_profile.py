from __future__ import annotations

import json

from tools.openclaw import vrchat_autonomy as autonomy
from tools.openclaw import vrchat_profile


def test_prepare_autonomy_profile_writes_enabled_dry_run_private_profile(tmp_path):
    profile_path = tmp_path / "profile.json"

    result = vrchat_profile.prepare_autonomy_profile(profile_path=profile_path)

    assert result["success"] is True
    assert result["written"] is True
    assert result["code"] == "DRY_RUN_PROFILE_READY"
    saved = json.loads(profile_path.read_text(encoding="utf-8"))
    assert saved["enabled"] is True
    assert saved["mode"] == "private_test"
    assert saved["dry_run"] is True
    assert saved["allow_voice"] is True
    assert saved["allow_chatbox"] is True
    assert saved["allow_movement"] is False
    assert saved["audio_output_device"] == "CABLE Input"
    assert saved["vrchat_microphone_device"] == "CABLE Output"
    assert saved["output_device"] == "CABLE Input"
    assert saved["live_actuation_ack"] == ""
    assert result["safety"]["actuation_performed"] is False


def test_prepare_autonomy_profile_preserves_avatar_actions_when_omitted(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                **autonomy._default_profile(),
                "allowed_avatar_actions": ["wave"],
                "avatar_action_profiles": {
                    "wave": [{"name": "Wave", "value": True, "reset_after_sec": 0.2}]
                },
            }
        ),
        encoding="utf-8",
    )

    result = vrchat_profile.prepare_autonomy_profile(profile_path=profile_path)

    assert result["success"] is True
    saved = json.loads(profile_path.read_text(encoding="utf-8"))
    assert saved["allowed_avatar_actions"] == ["wave"]
    assert saved["avatar_action_profiles"]["wave"][0]["name"] == "Wave"


def test_prepare_autonomy_profile_live_requires_exact_ack_and_does_not_write(tmp_path):
    profile_path = tmp_path / "profile.json"

    result = vrchat_profile.prepare_autonomy_profile(
        profile_path=profile_path,
        arm_live=True,
        live_ack="yes",
    )

    assert result["success"] is False
    assert result["written"] is False
    assert result["code"] == "LIVE_ACK_REQUIRED"
    assert result["required_live_ack"] == autonomy.LIVE_ACTUATION_ACK
    assert not profile_path.exists()


def test_prepare_autonomy_profile_live_writes_only_with_exact_ack(tmp_path):
    profile_path = tmp_path / "profile.json"

    result = vrchat_profile.prepare_autonomy_profile(
        profile_path=profile_path,
        arm_live=True,
        live_ack=autonomy.LIVE_ACTUATION_ACK,
    )

    assert result["success"] is True
    assert result["code"] == "LIVE_PROFILE_ARMED"
    saved = json.loads(profile_path.read_text(encoding="utf-8"))
    assert saved["dry_run"] is False
    assert saved["live_actuation_ack"] == autonomy.LIVE_ACTUATION_ACK


def test_prepare_autonomy_profile_rejects_public_movement_without_write(tmp_path):
    profile_path = tmp_path / "profile.json"

    result = vrchat_profile.prepare_autonomy_profile(
        profile_path=profile_path,
        mode="public",
        allow_movement=True,
    )

    assert result["success"] is False
    assert result["written"] is False
    assert "public_mode_movement_not_allowed" in result["validation"]["errors"]
    assert not profile_path.exists()
