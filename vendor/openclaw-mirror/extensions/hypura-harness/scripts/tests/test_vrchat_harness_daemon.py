from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from hypura.vrchat_avatar_registry import (
    AvatarOscEndpoint,
    AvatarOscParameter,
    AvatarParameterCatalog,
)
from hypura.vrchat_safety_gate import VrchatSafetyGate


def make_catalog() -> AvatarParameterCatalog:
    return AvatarParameterCatalog(
        avatarId="avtr_existing",
        avatarName="Existing Avatar",
        sourceFile="avatar.json",
        loadedAt=1,
        parameters=[
            AvatarOscParameter(
                name="Smile",
                input=AvatarOscEndpoint(address="/avatar/parameters/Smile", type="Bool"),
                output=None,
                writable=True,
                readable=False,
                inferredRole="expression",
                safety="safe",
            )
        ],
    )


def test_vrc_action_rejects_unapproved_profile(monkeypatch) -> None:
    import harness_daemon as hd

    catalog = make_catalog()
    monkeypatch.setattr(hd.vrchat_registry, "current_avatar_id", "avtr_existing")
    monkeypatch.setattr(hd.vrchat_registry, "catalog", catalog)
    monkeypatch.setattr(hd, "is_vrchat_active", Mock(return_value=True))
    monkeypatch.setattr(
        hd.vrchat_profiles,
        "load_profile",
        Mock(return_value={"avatarId": "avtr_existing", "approved": False, "actions": {}}),
    )
    client = TestClient(hd.app)

    resp = client.post("/vrc/action", json={"action": "smile"})

    assert resp.status_code == 200
    assert resp.json()["error"] == "profile_not_approved"


def test_vrc_action_sends_only_approved_profile_steps(monkeypatch) -> None:
    import harness_daemon as hd

    catalog = make_catalog()
    profile = {
        "avatarId": "avtr_existing",
        "approved": True,
        "actions": {
            "smile": {
                "label": "Smile",
                "cooldownMs": 0,
                "steps": [
                    {
                        "address": "/avatar/parameters/Smile",
                        "name": "Smile",
                        "type": "Bool",
                        "value": True,
                    }
                ],
            }
        },
    }
    bridge = Mock()
    monkeypatch.setattr(hd.vrchat_registry, "current_avatar_id", "avtr_existing")
    monkeypatch.setattr(hd.vrchat_registry, "catalog", catalog)
    monkeypatch.setattr(hd.vrchat_profiles, "load_profile", Mock(return_value=profile))
    monkeypatch.setattr(hd, "vrchat_bridge", bridge)
    monkeypatch.setattr(hd, "vrchat_safety", VrchatSafetyGate(global_rate_limit_per_second=100))
    monkeypatch.setattr(hd, "is_vrchat_active", Mock(return_value=True))
    client = TestClient(hd.app)

    resp = client.post("/vrc/action", json={"action": "smile", "reason": "test"})

    assert resp.status_code == 200
    assert resp.json()["success"] is True
    bridge.send_parameter.assert_called_once_with("/avatar/parameters/Smile", True)


def test_direct_vrc_parameter_write_is_disabled_by_default() -> None:
    from harness_daemon import app

    client = TestClient(app)
    resp = client.post(
        "/vrc/parameter",
        json={"address": "/avatar/parameters/Smile", "value": True},
    )

    assert resp.status_code == 200
    assert resp.json()["error"] == "direct_parameter_write_disabled"


def test_vrchat_bridge_callbacks_update_legacy_telemetry(monkeypatch) -> None:
    import harness_daemon as hd

    event_bus = Mock()
    registry = Mock()
    registry.set_current_avatar.return_value = None
    profiles = Mock()
    profiles.load_profile.return_value = None
    monkeypatch.setattr(hd, "companion3d_events", event_bus)
    monkeypatch.setattr(hd, "vrchat_registry", registry)
    monkeypatch.setattr(hd, "vrchat_profiles", profiles)
    hd.osc_listen.telemetry.clear()

    hd._handle_vrchat_avatar_change("avtr_test")
    hd._handle_vrchat_parameter("/avatar/parameters/Smile", True)

    assert hd.osc_listen.telemetry["avatar_id"] == "avtr_test"
    assert hd.osc_listen.telemetry["/avatar/parameters/Smile"] is True


def test_companion3d_rejects_remote_model_url() -> None:
    from harness_daemon import app

    client = TestClient(app)
    resp = client.post(
        "/companion3d/load-model",
        json={"model_path": "https://example.com/avatar.vrm"},
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is False
    assert resp.json()["error"] == "companion3d_model_rejected"


def test_companion3d_emotion_event_forwards_to_companion(monkeypatch) -> None:
    import harness_daemon as hd

    mock_bridge = Mock()
    mock_bridge.forward_emotion = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(hd, "companion_bridge", mock_bridge)
    client = TestClient(hd.app)

    resp = client.post(
        "/companion3d/event",
        json={"type": "emotion", "payload": {"emotion": "happy", "intensity": 0.75}},
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is True
    mock_bridge.forward_emotion.assert_awaited_once_with("happy")
