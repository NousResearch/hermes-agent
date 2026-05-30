from __future__ import annotations

import json

from tools.openclaw import neuro_bridge
from tools.openclaw import vrchat_autonomy as autonomy


def _profile(**overrides):
    profile = {
        **autonomy._default_profile(),
        "enabled": True,
        "mode": "private_test",
        "dry_run": True,
        "allow_voice": True,
        "allow_chatbox": True,
        "allowed_avatar_actions": ["wave"],
        "avatar_action_profiles": {
            "wave": [{"name": "Wave", "value": True, "reset_after_sec": 0.1}]
        },
    }
    profile.update(overrides)
    return profile


def _write_profile(tmp_path, **overrides):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(**overrides)), encoding="utf-8")
    return profile_path


def test_vendor_status_sees_cloned_sdk(monkeypatch, tmp_path):
    sdk_path = tmp_path / "neuro-sdk"
    api_path = sdk_path / "API"
    api_path.mkdir(parents=True)
    (api_path / "SPECIFICATION.md").write_text("# Spec\n", encoding="utf-8")
    (api_path / "README.md").write_text("# API\n", encoding="utf-8")
    (sdk_path / "LICENSE.md").write_text("MIT License\n", encoding="utf-8")
    monkeypatch.setattr(neuro_bridge, "NEURO_SDK_PATH", sdk_path)

    status = neuro_bridge.neuro_sdk_vendor_status()

    assert status["success"] is True
    assert status["license"] == "MIT"
    assert status["specification_exists"] is True
    assert status["api_readme_exists"] is True


def test_build_bootstrap_messages_include_startup_context_and_actions(tmp_path):
    profile_path = _write_profile(tmp_path)

    result = neuro_bridge.build_neuro_bridge_bootstrap(
        profile_path=profile_path,
        context="Hermes bridge connected.",
    )

    assert result["success"] is True
    assert [message["command"] for message in result["messages"]] == [
        "startup",
        "context",
        "actions/register",
    ]
    action_names = [action["name"] for action in result["messages"][2]["data"]["actions"]]
    assert "vrchat_autonomy_turn" in action_names
    assert "vrchat_avatar_action" in action_names


def test_action_catalog_uses_profile_avatar_actions(tmp_path):
    profile_path = _write_profile(tmp_path, allowed_avatar_actions=["wave", "nod"])

    actions = neuro_bridge.build_vrchat_neuro_actions(profile_path=profile_path)
    avatar_action = next(action for action in actions if action["name"] == "vrchat_avatar_action")

    assert avatar_action["schema"]["properties"]["avatar_action"]["enum"] == ["wave", "nod"]


def test_handle_action_blocks_disabled_profile(tmp_path):
    profile_path = _write_profile(tmp_path, enabled=False)
    message = {
        "command": "action",
        "data": {"id": "a1", "name": "vrchat_chatbox", "data": json.dumps({"text": "hello"})},
    }

    result = neuro_bridge.handle_neuro_action_message(message, profile_path=profile_path)

    assert result["success"] is False
    assert result["turn"] is None
    assert result["action_result"]["command"] == "action/result"
    assert result["action_result"]["data"]["success"] is True
    assert "profile_not_enabled_or_invalid" in result["action_result"]["data"]["message"]


def test_handle_action_plans_dry_run_chatbox(tmp_path):
    profile_path = _write_profile(tmp_path)
    message = {
        "command": "action",
        "data": {"id": "a2", "name": "vrchat_chatbox", "data": json.dumps({"text": "hello"})},
    }

    result = neuro_bridge.handle_neuro_action_message(message, profile_path=profile_path)

    assert result["success"] is True
    assert result["turn"]["dry_run"] is True
    assert result["turn"]["planned_actions"][0]["kind"] == "chatbox"
    assert result["action_result"]["data"]["success"] is True
    assert "Dry-run planned safely" in result["action_result"]["data"]["message"]


def test_handle_action_force_dry_run_prevents_live_profile_execution(tmp_path):
    profile_path = _write_profile(
        tmp_path,
        dry_run=False,
        live_actuation_ack=autonomy.LIVE_ACTUATION_ACK,
    )
    message = {
        "command": "action",
        "data": {
            "id": "a-live-audit",
            "name": "vrchat_autonomy_turn",
            "data": json.dumps(
                {
                    "speak_text": "hello",
                    "chatbox_text": "hello",
                    "emotion": "neutral",
                    "avatar_action": "",
                    "urgency": "low",
                }
            ),
        },
    }

    result = neuro_bridge.handle_neuro_action_message(
        message,
        profile_path=profile_path,
        force_dry_run=True,
    )

    assert result["success"] is True
    assert result["turn"]["dry_run"] is True
    assert [action["kind"] for action in result["turn"]["planned_actions"]] == ["chatbox", "voice"]
    assert result["turn"]["execution_results"] == []
    assert result["turn"]["safety"]["actuation_performed"] is False


def test_handle_action_rejects_invalid_json_without_retry(tmp_path):
    profile_path = _write_profile(tmp_path)
    message = {
        "command": "action",
        "data": {"id": "a3", "name": "vrchat_chatbox", "data": "{not-json"},
    }

    result = neuro_bridge.handle_neuro_action_message(message, profile_path=profile_path)

    assert result["success"] is False
    assert result["action_result"]["data"]["success"] is True
    assert "action_payload_invalid_json" in result["action_result"]["data"]["message"]


def test_handle_action_rejects_invalid_json_with_retry(tmp_path):
    profile_path = _write_profile(tmp_path)
    message = {
        "command": "action",
        "data": {"id": "a4", "name": "vrchat_chatbox", "data": "{not-json"},
    }

    result = neuro_bridge.handle_neuro_action_message(
        message,
        profile_path=profile_path,
        retry_on_failure=True,
    )

    assert result["success"] is False
    assert result["action_result"]["data"]["success"] is False
