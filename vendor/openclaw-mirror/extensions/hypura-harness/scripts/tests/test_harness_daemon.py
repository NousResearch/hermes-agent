# scripts/hypura/tests/test_harness_daemon.py
import importlib
import json
import sys
import time
import types
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient


def test_health_returns_200_without_dependency_probe() -> None:
    from harness_daemon import app

    with patch("harness_daemon._probe_http_ok", side_effect=AssertionError):
        client = TestClient(app)
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "daemon_version": "0.1.0"}


def test_status_returns_200() -> None:
    from harness_daemon import app

    client = TestClient(app)
    resp = client.get("/status")
    assert resp.status_code == 200


def test_status_has_required_keys() -> None:
    from harness_daemon import app

    client = TestClient(app)
    data = client.get("/status").json()
    assert "daemon_version" in data
    assert "osc_connected" in data
    assert "voicevox_alive" in data
    assert "ollama_alive" in data
    assert "lora" in data
    assert "base_model_configured" in data["lora"]


def test_channel_readiness_endpoint_redacts_config_values(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "channels": {
                    "line": {
                        "channelAccessToken": "line-secret-token",
                        "channelSecret": "line-channel-secret",
                    },
                    "telegram": {
                        "botToken": "telegram-secret-token",
                        "groups": {"-1001234567890": {}},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(config_path))
    from harness_daemon import app

    client = TestClient(app)
    resp = client.get("/channels/readiness")

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["channels"]["line"]["credentialPresence"]["channelAccessToken"] is True
    assert data["channels"]["line"]["liveRoundtripReady"] is False
    assert data["channels"]["telegram"]["liveRoundtripReady"] is True
    serialized = json.dumps(data)
    assert "line-secret-token" not in serialized
    assert "line-channel-secret" not in serialized
    assert "telegram-secret-token" not in serialized
    assert "-1001234567890" not in serialized


def test_status_times_out_optional_redis_loop(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_HYPURA_STATUS_DEP_TIMEOUT_SEC", "0.05")
    from harness_daemon import app

    def slow_loop_stats() -> dict[str, str]:
        time.sleep(0.5)
        return {"redis": "connected"}

    with patch("harness_daemon.redis_loop.get_loop_stats", side_effect=slow_loop_stats):
        client = TestClient(app)
        resp = client.get("/status")

    assert resp.status_code == 200
    assert resp.json()["loop"] == {"redis": "timeout"}


def test_redis_loop_unavailable_connection_is_cached(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_HYPURA_REDIS_TIMEOUT_SEC", "0.01")
    monkeypatch.setenv("OPENCLAW_HYPURA_REDIS_RETRY_INTERVAL_SEC", "30")
    import redis_loop

    redis_loop = importlib.reload(redis_loop)

    class FakeRedis:
        calls = 0

        def __init__(self, **_kwargs) -> None:
            FakeRedis.calls += 1

        def ping(self) -> None:
            raise TimeoutError("redis unavailable")

    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=FakeRedis))

    assert redis_loop.get_loop_stats() == {"redis": "unavailable"}
    assert redis_loop.get_loop_stats() == {"redis": "unavailable"}
    assert FakeRedis.calls == 1


def test_osc_endpoint_chatbox() -> None:
    from harness_daemon import app

    with patch("harness_daemon.is_vrchat_active", return_value=True), patch(
        "harness_daemon.osc_ctrl"
    ) as mock_osc:
        client = TestClient(app)
        resp = client.post(
            "/osc", json={"action": "chatbox", "payload": {"text": "hi"}}
        )
        assert resp.status_code == 200
        mock_osc.send_chatbox.assert_called_once_with(
            "hi", immediate=True, sfx=True
        )


def test_run_endpoint_returns_result() -> None:
    from harness_daemon import app

    with patch("harness_daemon.code_runner_instance") as mock_runner:
        mock_runner.run_task.return_value = {"success": True, "output": "done"}
        client = TestClient(app)
        resp = client.post("/run", json={"task": "print hello"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True


def test_speak_forwards_to_companion_bridge() -> None:
    from harness_daemon import app

    with patch("harness_daemon.is_vrchat_active", return_value=True), patch(
        "harness_daemon.voicevox_seq"
    ) as mock_vx, patch(
        "harness_daemon.companion_bridge"
    ) as mock_bridge:
        mock_vx.speak = AsyncMock()
        mock_bridge.forward_speak = AsyncMock()
        client = TestClient(app)
        resp = client.post(
            "/speak", json={"text": "hello", "emotion": "happy"}
        )
        assert resp.status_code == 200
        mock_bridge.forward_speak.assert_awaited_once_with("hello", "happy")


def test_voice_devices_returns_sounddevice_inventory() -> None:
    from harness_daemon import app

    with patch(
        "harness_daemon.list_audio_devices",
        return_value={"devices": [{"index": 4, "name": "Speaker"}], "default": [1, 4]},
    ):
        client = TestClient(app)
        resp = client.get("/voice/devices")
        assert resp.status_code == 200
        assert resp.json()["devices"][0]["index"] == 4


def test_voice_test_say_synthesizes_and_plays_selected_device() -> None:
    from harness_daemon import app

    with patch("harness_daemon.voicevox_seq") as mock_vx:
        mock_vx.synthesize = AsyncMock(return_value=b"RIFF")
        client = TestClient(app)
        resp = client.post(
            "/voice/test-say",
            json={"text": "hello", "speaker": 3, "output_device": 4},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_vx.synthesize.assert_awaited_once_with("hello", emotion="neutral", speaker=3)
        mock_vx.play_wav_bytes.assert_called_once_with(b"RIFF", output_devices=[4])


def test_voice_test_say_accepts_dual_output_devices() -> None:
    from harness_daemon import app

    with patch("harness_daemon.voicevox_seq") as mock_vx:
        mock_vx.synthesize = AsyncMock(return_value=b"RIFF")
        client = TestClient(app)
        resp = client.post(
            "/voice/test-say",
            json={"text": "hello", "speaker": 3, "output_devices": [5, 4]},
        )
        assert resp.status_code == 200
        assert resp.json()["output_devices"] == [5, 4]
        mock_vx.play_wav_bytes.assert_called_once_with(b"RIFF", output_devices=[5, 4])


def test_voice_transcribe_uses_whisper_bridge() -> None:
    from harness_daemon import app

    with patch("harness_daemon.transcribe_wav", return_value="transcript") as mock_transcribe:
        client = TestClient(app)
        resp = client.post(
            "/voice/transcribe",
            json={"wav_path": "C:/tmp/input.wav"},
        )
        assert resp.status_code == 200
        assert resp.json()["transcript"] == "transcript"
        assert mock_transcribe.called


def test_voice_turn_speaks_openclaw_reply() -> None:
    from harness_daemon import app

    with patch(
        "harness_daemon.run_voice_turn",
        return_value={"success": True, "transcript": "hi", "reply": "reply"},
    ) as mock_turn, patch("harness_daemon.voicevox_seq") as mock_vx:
        mock_vx.synthesize = AsyncMock(return_value=b"RIFF")
        client = TestClient(app)
        resp = client.post(
            "/voice/turn",
            json={
                "record_seconds": 1,
                "input_device": 1,
                "output_device": 4,
                "speaker": 3,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["reply"] == "reply"
        assert mock_turn.called
        mock_vx.synthesize.assert_awaited_once_with("reply", emotion="neutral", speaker=3)
        mock_vx.play_wav_bytes.assert_called_once_with(b"RIFF", output_devices=[4])


def test_osc_emotion_forwards_companion_bridge() -> None:
    from harness_daemon import app

    with patch("harness_daemon.is_vrchat_active", return_value=True), patch(
        "harness_daemon.osc_ctrl"
    ) as mock_osc, patch(
        "harness_daemon.companion_bridge"
    ) as mock_bridge:
        mock_bridge.forward_emotion = AsyncMock()
        client = TestClient(app)
        resp = client.post(
            "/osc",
            json={"action": "emotion", "payload": {"emotion": "happy"}},
        )
        assert resp.status_code == 200
        mock_osc.apply_emotion.assert_called_once_with("happy")
        mock_bridge.forward_emotion.assert_awaited_once_with("happy")


def test_companion_control_motion_endpoint() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge:
        mock_bridge.forward_motion = AsyncMock()
        client = TestClient(app)
        resp = client.post(
            "/companion/control",
            json={"action": "motion", "value": "Idle", "motion_index": 1},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_bridge.forward_motion.assert_awaited_once_with("Idle", 1)


def test_companion_control_speak_endpoint_forwards_emotion_and_tts_provider() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge:
        mock_bridge.forward_speak = AsyncMock(return_value={"ok": True})
        client = TestClient(app)
        resp = client.post(
            "/companion/control",
            json={
                "action": "speak",
                "value": "hello",
                "emotion": "happy",
                "tts_provider": "web-speech",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_bridge.forward_speak.assert_awaited_once_with(
            "hello", "happy", "web-speech"
        )


def test_companion_control_state_endpoints() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge:
        mock_bridge.get_state = AsyncMock(return_value={"state": {"voice": {"micActive": False}}})
        mock_bridge.set_permission = AsyncMock(return_value={"permissionState": {"ok": True}})
        mock_bridge.set_mic_enabled = AsyncMock(return_value={"micResult": {"ok": True}})
        mock_bridge.input_snapshot = AsyncMock(
            return_value={"inputSnapshot": {"transcript": "hello"}}
        )
        mock_bridge.window_capture = AsyncMock(
            return_value={"windowCapture": {"mimeType": "image/png"}}
        )
        client = TestClient(app)

        status_resp = client.post("/companion/control", json={"action": "status"})
        permission_resp = client.post(
            "/companion/control",
            json={"action": "permission", "capability": "mic", "decision": "granted"},
        )
        mic_resp = client.post(
            "/companion/control", json={"action": "mic", "enabled": True}
        )
        snapshot_resp = client.post(
            "/companion/control",
            json={
                "action": "input_snapshot",
                "include_camera": True,
                "capture_camera": False,
            },
        )
        capture_resp = client.post(
            "/companion/control", json={"action": "window_capture"}
        )

        assert status_resp.status_code == 200
        assert permission_resp.status_code == 200
        assert mic_resp.status_code == 200
        assert snapshot_resp.status_code == 200
        assert capture_resp.status_code == 200
        assert status_resp.json()["state"]["voice"]["micActive"] is False
        assert snapshot_resp.json()["inputSnapshot"]["transcript"] == "hello"
        assert capture_resp.json()["windowCapture"]["mimeType"] == "image/png"
        mock_bridge.set_permission.assert_awaited_once_with("mic", "granted")
        mock_bridge.set_mic_enabled.assert_awaited_once_with(True)
        mock_bridge.input_snapshot.assert_awaited_once_with(
            include_camera=True,
            capture_camera=False,
        )
        mock_bridge.window_capture.assert_awaited_once_with()


def test_companion_control_mic_reports_nested_bridge_denial() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge:
        mock_bridge.set_mic_enabled = AsyncMock(
            return_value={
                "ok": True,
                "micResult": {
                    "ok": False,
                    "reason": "Microphone permission is denied",
                },
            }
        )
        client = TestClient(app)
        resp = client.post(
            "/companion/control", json={"action": "mic", "enabled": True}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["micResult"]["reason"] == "Microphone permission is denied"
        mock_bridge.set_mic_enabled.assert_awaited_once_with(True)


def test_companion_control_speak_reports_bridge_failure() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge, patch(
        "harness_daemon.voicevox_seq"
    ) as mock_voicevox:
        mock_bridge.forward_speak = AsyncMock(
            return_value={"ok": False, "error": "Desktop companion IPC unavailable"}
        )
        mock_voicevox.synthesize = AsyncMock(return_value=b"wav")
        mock_voicevox.play_wav_bytes = Mock()
        client = TestClient(app)
        resp = client.post(
            "/companion/control",
            json={"action": "speak", "value": "hello", "emotion": "happy"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["error"] == "Desktop companion IPC unavailable"
        mock_bridge.forward_speak.assert_awaited_once_with("hello", "happy", None)


def test_voice_companion_mic_reports_nested_bridge_denial() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge:
        mock_bridge.set_mic_enabled = AsyncMock(
            return_value={
                "ok": True,
                "micResult": {
                    "ok": False,
                    "reason": "Microphone permission is denied",
                },
            }
        )
        client = TestClient(app)
        resp = client.post("/voice/companion-mic", json={"enabled": True})

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["enabled"] is True
        assert data["micResult"]["reason"] == "Microphone permission is denied"


def test_voice_companion_turn_reports_companion_speech_failure() -> None:
    import harness_daemon as hd

    with patch(
        "harness_daemon.run_companion_transcript_turn",
        return_value={"success": True, "reply": "reply", "emotion": "happy"},
    ), patch("harness_daemon.companion_bridge") as mock_bridge, patch(
        "harness_daemon.voicevox_seq"
    ) as mock_voicevox:
        mock_bridge.forward_emotion = AsyncMock(return_value={"ok": True})
        mock_bridge.forward_speak = AsyncMock(
            return_value={"ok": False, "error": "Desktop companion IPC unavailable"}
        )
        mock_voicevox.synthesize = AsyncMock(return_value=b"wav")
        mock_voicevox.play_wav_bytes = Mock()
        old_config = hd.config
        hd.config = {**old_config, "voice": {"output_devices": [4]}}
        hd.companion3d_events.events.clear()
        hd.companion3d_events.last_state.clear()
        try:
            client = TestClient(hd.app)
            resp = client.post(
                "/voice/companion-turn",
                json={"transcript": "hi", "speak": True, "animate": True},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert data["companion_animation"]["ok"] is True
            assert data["companion_speech"]["error"] == "Desktop companion IPC unavailable"
            assert data["companion_monitor_speech"]["output_devices"] == [4]
            assert [event["type"] for event in hd.companion3d_events.events[-5:]] == [
                "emotion",
                "speak_start",
                "state",
                "speak_end",
                "state",
            ]
            assert hd.companion3d_events.last_state["state"]["speaking"] is False
            assert hd.companion3d_events.last_state["state"]["lipSync"] is False
            mock_voicevox.synthesize.assert_awaited_once_with("reply", emotion="happy", speaker=8)
            mock_voicevox.play_wav_bytes.assert_called_once_with(b"wav", output_devices=[4])
        finally:
            hd.config = old_config


def test_companion_control_look_at_endpoint() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge:
        mock_bridge.forward_look = AsyncMock()
        client = TestClient(app)
        resp = client.post(
            "/companion/control",
            json={"action": "look_at", "x": 0.25, "y": -0.5},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_bridge.forward_look.assert_awaited_once_with(0.25, -0.5)


def test_companion_control_load_model_resolves_repo_relative_path() -> None:
    from harness_daemon import app

    with patch("harness_daemon.companion_bridge") as mock_bridge:
        mock_bridge.forward_load_model = AsyncMock()
        client = TestClient(app)
        resp = client.post(
            "/companion/control",
            json={
                "action": "load_model",
                "model_path": "assets/NFD/SampleAvatar/FBX/SampleAvatar.fbx",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        forwarded_path = mock_bridge.forward_load_model.await_args.args[0]
        assert forwarded_path.endswith("assets\\NFD\\SampleAvatar\\FBX\\SampleAvatar.fbx")


def test_submodule_run_proxies_to_gateway_tool_surface() -> None:
    import harness_daemon as hd

    mock_response = Mock()
    mock_response.is_success = True
    mock_response.json.return_value = {
        "ok": True,
        "result": {
            "ok": True,
            "status": "completed",
            "repoId": "vrchat-mcp-osc",
            "preset": "status",
        },
    }

    with patch("harness_daemon._resolve_gateway_auth_token", return_value="gateway-token"), patch(
        "harness_daemon._resolve_gateway_base_url", return_value="http://127.0.0.1:18789"
    ), patch("harness_daemon.httpx.AsyncClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__aenter__.return_value
        mock_client.post = AsyncMock(return_value=mock_response)
        client = TestClient(hd.app)
        resp = client.post(
            "/submodule/run",
            json={
                "repoId": "vrchat-mcp-osc",
                "preset": "status",
                "extraArgs": ["--branch"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        mock_client.post.assert_awaited_once_with(
            "http://127.0.0.1:18789/tools/invoke",
            json={
                "tool": "submodule_run",
                "args": {
                    "repoId": "vrchat-mcp-osc",
                    "preset": "status",
                    "extraArgs": ["--branch"],
                },
            },
            headers={
                "Authorization": "Bearer gateway-token",
                "x-openclaw-message-channel": "node",
            },
        )


def test_reload_returns_reloaded_true(tmp_path, monkeypatch) -> None:
    import harness_daemon as hd

    cfg_path = tmp_path / "harness.config.json"
    cfg_path.write_text(json.dumps({"daemon_port": 18794}))
    monkeypatch.setattr(hd, "CONFIG_PATH", cfg_path)
    client = TestClient(hd.app)
    resp = client.post("/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reloaded"] is True
    assert "config" in body


def test_reload_reflects_updated_config(tmp_path, monkeypatch) -> None:
    import harness_daemon as hd

    cfg_path = tmp_path / "harness.config.json"
    cfg_path.write_text(
        json.dumps({"daemon_port": 18794, "test_key": "before"})
    )
    monkeypatch.setattr(hd, "CONFIG_PATH", cfg_path)
    client = TestClient(hd.app)
    assert client.post("/reload").json()["config"]["test_key"] == "before"
    cfg_path.write_text(json.dumps({"daemon_port": 18794, "test_key": "after"}))
    assert client.post("/reload").json()["config"]["test_key"] == "after"
