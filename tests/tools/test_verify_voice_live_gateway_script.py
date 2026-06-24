import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "verify_voice_live_gateway.py"
)
SCRIPTS_DIR = SCRIPT_PATH.parent


def _load_script_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("verify_voice_live_gateway", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_voice_native_root(root: Path) -> None:
    tools_dir = root / "tools"
    gateway_dir = root / "gateway" / "platforms"
    bridge_bin = root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"
    tools_dir.mkdir(parents=True)
    gateway_dir.mkdir(parents=True)
    bridge_bin.mkdir(parents=True)
    (tools_dir / "tts_tool.py").write_text(
        "\n".join(["voice_compatible", "libopus", "-application", "voip"]),
        encoding="utf-8",
    )
    (tools_dir / "transcription_tools.py").write_text(
        "\n".join(["stt.providers", "_transcribe_command_stt", "transcribe_audio"]),
        encoding="utf-8",
    )
    (gateway_dir / "whatsapp_cloud.py").write_text(
        "\n".join(
            [
                "calling_sidecar_url",
                "voice.webrtc_sidecar",
                "_send_calling_sidecar_tts_stream_command",
                "_clear_calling_sidecar_audio",
                "-application",
                "voip",
            ]
        ),
        encoding="utf-8",
    )


def _voice_sidecar_contract() -> dict:
    audio = {
        "sample_rate": 48_000,
        "channels": 1,
        "frame_ms": 20,
        "encoding": "pcm_s16le",
        "bytes_per_sample": 2,
        "samples_per_frame": 960,
        "frame_bytes": 1_920,
    }
    return {
        "contract": "voice.webrtc_sidecar",
        "version": 1,
        "status": "experimental",
        "summary": "Local HTTP/WebRTC bridge contract.",
        "audio": audio,
        "voice_surfaces": {
            "completed_voice_note": {
                "command": 'voice say --format ogg-opus --output reply.ogg "hello"',
                "output": "audio/ogg; codecs=opus",
                "transport": "completed_file",
            },
            "streamed_voice_note": {
                "command": 'voice stream --output reply.ogg --format ogg-opus "hello"',
                "output": "audio/ogg; codecs=opus",
                "transport": "daemon_stream_encoded_file",
            },
            "raw_outbound_pcm": {
                "command": 'voice stream --sample-rate 48000 --frame-ms 20 --raw-output - "hello"',
                "output": "pcm_s16le",
                "transport": "stdout_pcm_frames",
                "frame_bytes": audio["frame_bytes"],
            },
            "raw_inbound_pcm": {
                "command": "voice stream-transcribe --raw-input - --sample-rate 48000 --frame-ms 20",
                "input": "pcm_s16le",
                "transport": "stdin_pcm_frames",
                "frame_bytes": audio["frame_bytes"],
            },
            "file_transcription_smoke": {
                "command": "voice stream-transcribe recording.ogg",
                "input": "audio_file",
                "transport": "decoded_file_to_daemon_frames",
            },
        },
        "endpoints": {
            "contract": {"method": "GET", "path": "/contract"},
            "health": {"method": "GET", "path": "/health"},
            "offer": {"method": "POST", "path": "/offer"},
            "call_status": {"method": "GET", "path": "/calls/{call_id}"},
            "receive_audio": {"method": "GET", "path": "/calls/{call_id}/audio"},
            "send_audio": {"method": "POST", "path": "/calls/{call_id}/audio"},
            "clear_audio": {
                "method": "POST",
                "path": "/calls/{call_id}/audio/clear",
            },
            "close_call": {"method": "POST", "path": "/calls/{call_id}/close"},
        },
        "payloads": {
            "offer_request": {"sdp": "required"},
            "offer_response": {"sdp": "answer"},
            "call_state": {
                "ready_for_accept": "ready",
                "readiness": "checks",
                "queued_tx_bytes": "bytes",
                "queued_tx_ms": "ms",
                "max_tx_queue_bytes": "bytes",
                "max_tx_queue_ms": "ms",
                "queued_rx_bytes": "bytes",
                "queued_rx_ms": "ms",
                "max_rx_queue_bytes": "bytes",
                "max_rx_queue_ms": "ms",
            },
            "call_status_response": "call_state",
            "close_call_response": {"closed": True},
            "send_audio_request": {"pcm_s16le_base64": "required"},
            "send_audio_response": {
                "accepted_bytes": "bytes",
                "accepted_ms": "ms",
                "queued_tx_bytes": "bytes",
                "queued_tx_ms": "ms",
                "max_tx_queue_bytes": "bytes",
                "max_tx_queue_ms": "ms",
            },
            "clear_audio_response": {
                "dropped_tx_bytes": "bytes",
                "dropped_tx_ms": "ms",
                "queued_tx_bytes": "bytes",
                "queued_tx_ms": "ms",
                "max_tx_queue_bytes": "bytes",
                "max_tx_queue_ms": "ms",
            },
            "receive_audio_response": {
                "returned_bytes": "bytes",
                "returned_ms": "ms",
                "queued_rx_bytes": "bytes",
                "queued_rx_ms": "ms",
                "max_rx_queue_bytes": "bytes",
                "max_rx_queue_ms": "ms",
                "pcm_s16le_base64": "data",
            },
            "audio_shape": {"sample_rate": "audio.sample_rate"},
            "error_response": {"error": "message"},
        },
    }


def test_parse_systemctl_show_keeps_key_values():
    script = _load_script_module()

    parsed = script.parse_systemctl_show(
        "ActiveState=active\nMainPID=123\nignored line\nEnvironment=A=B\n"
    )

    assert parsed == {
        "ActiveState": "active",
        "MainPID": "123",
        "Environment": "A=B",
    }


def test_parse_systemd_environment_handles_quoted_values():
    script = _load_script_module()

    parsed = script.parse_systemd_environment(
        'PYTHONPATH=/live PATH=/bin "EXTRA=value with spaces"'
    )

    assert parsed == {
        "PYTHONPATH": "/live",
        "PATH": "/bin",
        "EXTRA": "value with spaces",
    }


def test_parse_proc_environ_reads_nul_separated_environment():
    script = _load_script_module()

    parsed = script.parse_proc_environ(b"PYTHONPATH=/live\0PATH=/bin\0bad\0")

    assert parsed == {"PYTHONPATH": "/live", "PATH": "/bin"}


def test_parse_dotenv_text_handles_quotes_exports_and_comments():
    script = _load_script_module()

    parsed = script.parse_dotenv_text(
        "\n".join(
            [
                "# ignored",
                "export WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                'WHATSAPP_CLOUD_VERIFY_TOKEN="verify-token-value"',
                "WHATSAPP_CLOUD_ALLOWED_USERS=15551234567 # operator note",
            ]
        )
    )

    assert parsed == {
        "WHATSAPP_CLOUD_PHONE_NUMBER_ID": "7794189252778687",
        "WHATSAPP_CLOUD_VERIFY_TOKEN": "verify-token-value",
        "WHATSAPP_CLOUD_ALLOWED_USERS": "15551234567",
    }


def test_validate_whatsapp_cloud_readiness_reports_sources_without_secrets(
    tmp_path: Path,
):
    script = _load_script_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    access_token = "EAA" + ("x" * 120)
    app_secret = "0123456789abcdef0123456789abcdef"
    verify_token = "verify-token-value-long-enough"
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                f"WHATSAPP_CLOUD_ACCESS_TOKEN={access_token}",
                f"WHATSAPP_CLOUD_APP_SECRET={app_secret}",
                f"WHATSAPP_CLOUD_VERIFY_TOKEN={verify_token}",
                "WHATSAPP_CLOUD_ALLOWED_USERS=15551234567,15557654321",
            ]
        ),
        encoding="utf-8",
    )

    raw_result = script.validate_whatsapp_cloud_readiness(
        hermes_home=hermes_home,
        process_env={},
    )
    result = script.remove_private_readiness_data(raw_result)

    assert result["required_fields"]["WHATSAPP_CLOUD_ACCESS_TOKEN"] == {
        "present": True,
        "source_shape": "meta_access_token",
        "source": "env_file",
    }
    assert result["authorization"]["allowed_users_count"] == 2
    serialized = json.dumps(result)
    assert access_token not in serialized
    assert app_secret not in serialized
    assert verify_token not in serialized


def test_whatsapp_cloud_health_url_uses_loopback_for_wildcard_host():
    script = _load_script_module()

    result = script.whatsapp_cloud_health_url_from_env(
        file_env={
            "WHATSAPP_CLOUD_WEBHOOK_HOST": "0.0.0.0",
            "WHATSAPP_CLOUD_WEBHOOK_PORT": "8091",
        },
        process_env={},
    )

    assert result == "http://127.0.0.1:8091/health"


def test_whatsapp_cloud_webhook_url_uses_configured_path():
    script = _load_script_module()

    result = script.whatsapp_cloud_webhook_url_from_env(
        file_env={
            "WHATSAPP_CLOUD_WEBHOOK_HOST": "0.0.0.0",
            "WHATSAPP_CLOUD_WEBHOOK_PORT": "8091",
            "WHATSAPP_CLOUD_WEBHOOK_PATH": "whatsapp/webhook",
        },
        process_env={},
    )

    assert result == "http://127.0.0.1:8091/whatsapp/webhook"


def test_validate_whatsapp_cloud_health_returns_redacted_summary():
    script = _load_script_module()

    result = script.validate_whatsapp_cloud_health(
        {
            "status": "ok",
            "platform": "whatsapp_cloud",
            "phone_number_id": "7794189252778687",
            "webhook_path": "/whatsapp/webhook",
            "verify_token_configured": True,
            "app_secret_configured": True,
            "calling_sidecar_configured": True,
            "calling_sidecar_contract_loaded": True,
            "calling_sidecar_tts_stream_configured": True,
        },
        expected_phone_number_id="7794189252778687",
        expect_calling_sidecar=True,
    )

    assert result == {
        "status": "ok",
        "platform": "whatsapp_cloud",
        "webhook_path": "/whatsapp/webhook",
        "verify_token_configured": True,
        "app_secret_configured": True,
        "calling_sidecar_configured": True,
        "calling_sidecar_contract_loaded": True,
        "calling_sidecar_tts_stream_configured": True,
    }
    assert "phone_number_id" not in result


def test_validate_whatsapp_cloud_health_rejects_unloaded_calling_sidecar():
    script = _load_script_module()

    with pytest.raises(SystemExit, match="calling sidecar is not configured"):
        script.validate_whatsapp_cloud_health(
            {
                "status": "ok",
                "platform": "whatsapp_cloud",
                "phone_number_id": "7794189252778687",
                "verify_token_configured": True,
                "app_secret_configured": True,
                "calling_sidecar_configured": False,
                "calling_sidecar_tts_stream_configured": True,
            },
            expected_phone_number_id="7794189252778687",
            expect_calling_sidecar=True,
        )


def test_check_whatsapp_cloud_verify_handshake_uses_token_without_reporting_it(
    tmp_path: Path,
    monkeypatch,
):
    script = _load_script_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    verify_token = "verify-token-value-long-enough"
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                "WHATSAPP_CLOUD_ACCESS_TOKEN=EAA" + ("x" * 120),
                "WHATSAPP_CLOUD_APP_SECRET=0123456789abcdef0123456789abcdef",
                f"WHATSAPP_CLOUD_VERIFY_TOKEN={verify_token}",
                "WHATSAPP_CLOUD_ALLOWED_USERS=15551234567",
                "WHATSAPP_CLOUD_WEBHOOK_PORT=8091",
            ]
        ),
        encoding="utf-8",
    )
    readiness = script.validate_whatsapp_cloud_readiness(
        hermes_home=hermes_home,
        process_env={},
    )

    def fake_get_text_url(target, *, timeout, label):
        assert verify_token in target
        assert verify_token not in label
        assert label == "http://127.0.0.1:8091/whatsapp/webhook"
        return 200, "challenge-ok"

    monkeypatch.setattr(script, "get_text_url", fake_get_text_url)

    result = script.check_whatsapp_cloud_verify_handshake(
        readiness=readiness,
        webhook_url=None,
        challenge="challenge-ok",
        timeout=1.0,
    )

    assert result == {
        "url": "http://127.0.0.1:8091/whatsapp/webhook",
        "status": 200,
        "challenge_echoed": True,
    }
    assert verify_token not in json.dumps(result)


def test_check_whatsapp_cloud_signed_post_uses_hmac_without_reporting_secret(
    tmp_path: Path,
    monkeypatch,
):
    script = _load_script_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    app_secret = "0123456789abcdef0123456789abcdef"
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                "WHATSAPP_CLOUD_ACCESS_TOKEN=EAA" + ("x" * 120),
                f"WHATSAPP_CLOUD_APP_SECRET={app_secret}",
                "WHATSAPP_CLOUD_VERIFY_TOKEN=verify-token-value-long-enough",
                "WHATSAPP_CLOUD_ALLOWED_USERS=15551234567",
                "WHATSAPP_CLOUD_WEBHOOK_PORT=8091",
            ]
        ),
        encoding="utf-8",
    )
    readiness = script.validate_whatsapp_cloud_readiness(
        hermes_home=hermes_home,
        process_env={},
    )

    def fake_post_json_url(target, *, body, headers, timeout, label):
        assert target == "http://127.0.0.1:8091/whatsapp/webhook"
        assert label == target
        assert app_secret.encode("utf-8") not in body
        assert headers["Content-Type"] == "application/json"
        signature = headers["X-Hub-Signature-256"]
        assert signature.startswith("sha256=")
        assert app_secret not in signature
        payload = json.loads(body.decode("utf-8"))
        assert payload["entry"][0]["changes"][0]["value"]["statuses"][0][
            "status"
        ] == "delivered"
        return 200, ""

    monkeypatch.setattr(script, "post_json_url", fake_post_json_url)

    result = script.check_whatsapp_cloud_signed_post(
        readiness=readiness,
        webhook_url=None,
        timeout=1.0,
    )

    assert result == {
        "url": "http://127.0.0.1:8091/whatsapp/webhook",
        "status": 200,
        "payload": "status_delivery_receipt",
        "signature_accepted": True,
    }
    assert app_secret not in json.dumps(result)


def test_validate_whatsapp_cloud_readiness_rejects_missing_authorization(
    tmp_path: Path,
):
    script = _load_script_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                "WHATSAPP_CLOUD_ACCESS_TOKEN=EAA" + ("x" * 120),
                "WHATSAPP_CLOUD_APP_SECRET=0123456789abcdef0123456789abcdef",
                "WHATSAPP_CLOUD_VERIFY_TOKEN=verify-token-value-long-enough",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="recipient authorization"):
        script.validate_whatsapp_cloud_readiness(
            hermes_home=hermes_home,
            process_env={},
        )


def test_validate_whatsapp_cloud_readiness_reports_all_missing_fields(
    tmp_path: Path,
):
    script = _load_script_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID=15551234567",
                "WHATSAPP_CLOUD_ACCESS_TOKEN=sk-not-meta",
                "WHATSAPP_CLOUD_APP_SECRET=too-short",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        script.validate_whatsapp_cloud_readiness(
            hermes_home=hermes_home,
            process_env={},
        )

    message = str(exc_info.value)
    assert "WhatsApp Cloud readiness failed:" in message
    assert "WHATSAPP_CLOUD_PHONE_NUMBER_ID looks like a phone number" in message
    assert "WHATSAPP_CLOUD_ACCESS_TOKEN should start with EAA" in message
    assert "WHATSAPP_CLOUD_APP_SECRET should be exactly 32 hex characters" in message
    assert "WHATSAPP_CLOUD_VERIFY_TOKEN is not configured" in message
    assert "recipient authorization is not configured" in message
    assert "15551234567" not in message
    assert "sk-not-meta" not in message
    assert "too-short" not in message


def test_path_is_under_accepts_children_and_rejects_siblings(tmp_path: Path):
    script = _load_script_module()
    root = tmp_path / "root"
    child = root / "pkg" / "module.py"
    sibling = tmp_path / "root-other" / "module.py"
    child.parent.mkdir(parents=True)
    sibling.parent.mkdir(parents=True)
    child.write_text("", encoding="utf-8")
    sibling.write_text("", encoding="utf-8")

    assert script.path_is_under(child, root) is True
    assert script.path_is_under(sibling, root) is False


def test_validate_service_state_requires_active_service_and_pid():
    script = _load_script_module()

    assert script.validate_service_state({"ActiveState": "active", "MainPID": "42"}) == 42

    with pytest.raises(SystemExit, match="not active"):
        script.validate_service_state({"ActiveState": "inactive", "MainPID": "42"})
    with pytest.raises(SystemExit, match="MainPID is invalid"):
        script.validate_service_state({"ActiveState": "active", "MainPID": "nope"})
    with pytest.raises(SystemExit, match="no running MainPID"):
        script.validate_service_state({"ActiveState": "active", "MainPID": "0"})


def test_validate_env_points_at_root_requires_pythonpath_and_bridge_path(tmp_path: Path):
    script = _load_script_module()
    root = tmp_path / "hermes"
    bridge_bin = root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"
    bridge_bin.mkdir(parents=True)
    env = {
        "PYTHONPATH": str(root),
        "PATH": f"/usr/bin:{bridge_bin}",
    }

    result = script.validate_env_points_at_root(
        env,
        root,
        label="unit",
        bridge_bin_dir=bridge_bin,
    )

    assert result["PYTHONPATH"] == str(root)
    assert result["bridge_bin_on_path"] is True

    with pytest.raises(SystemExit, match="PYTHONPATH"):
        script.validate_env_points_at_root({}, root, label="unit")
    with pytest.raises(SystemExit, match="PATH"):
        script.validate_env_points_at_root(
            {"PYTHONPATH": str(root), "PATH": "/usr/bin"},
            root,
            label="unit",
            bridge_bin_dir=bridge_bin,
        )


def test_validate_bridge_health_requires_connected_when_requested():
    script = _load_script_module()

    script.validate_bridge_health({"status": "connected"}, require_connected=True)
    script.validate_bridge_health({"status": "connecting"}, require_connected=False)
    with pytest.raises(SystemExit, match="not connected"):
        script.validate_bridge_health({"status": "connecting"}, require_connected=True)


def test_validate_calling_sidecar_env_requires_url_and_stream_command():
    script = _load_script_module()
    env = {
        "WHATSAPP_CLOUD_CALLING_SIDECAR_URL": "http://127.0.0.1:8787/",
        "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND": (
            "voice stream --raw-output - --input-file {input_path} "
            "--sample-rate {sample_rate} --frame-ms {frame_ms}"
        ),
        "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT": "180",
    }

    result = script.validate_calling_sidecar_env(env, "http://127.0.0.1:8787")

    assert result["url"] == "http://127.0.0.1:8787"
    assert result["tts_stream_timeout"] == "180"

    with pytest.raises(SystemExit, match="does not match"):
        script.validate_calling_sidecar_env(env, "http://127.0.0.1:9999")
    with pytest.raises(SystemExit, match="missing"):
        script.validate_calling_sidecar_env(
            {
                "WHATSAPP_CLOUD_CALLING_SIDECAR_URL": "http://127.0.0.1:8787",
                "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND": "voice stream",
            },
            "http://127.0.0.1:8787",
        )


def test_validate_sidecar_service_state_accepts_expected_unit(tmp_path: Path):
    script = _load_script_module()
    voice_repo = tmp_path / "voice"
    sidecar_path = voice_repo / "examples" / "webrtc-sidecar" / "sidecar.py"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text("", encoding="utf-8")
    voice_bin = tmp_path / "bin" / "voice"
    voice_bin.parent.mkdir()
    voice_bin.write_text("", encoding="utf-8")

    result = script.validate_sidecar_service_state(
        {
            "ActiveState": "active",
            "MainPID": "42",
            "After": "network.target voiced.service",
            "Environment": f"VOICE_BIN={voice_bin}",
            "WorkingDirectory": str(voice_repo),
            "ExecStart": (
                f"{{ argv[]={sys.executable} {sidecar_path} "
                "--host 127.0.0.1 --port 8787 }}"
            ),
        },
        service="voice-webrtc-sidecar.service",
        voice_bin=str(voice_bin),
        voice_repo=voice_repo,
        sidecar_url="http://127.0.0.1:8787",
        voice_daemon_service="voiced.service",
    )

    assert result["service"] == "voice-webrtc-sidecar.service"
    assert result["pid"] == 42
    assert result["voice_bin"] == str(voice_bin)
    assert result["working_directory"] == str(voice_repo)
    assert result["sidecar_path"] == str(sidecar_path)
    assert result["sidecar_url"] == "http://127.0.0.1:8787"
    assert result["bind"] == {"host": "127.0.0.1", "port": 8787}
    assert "voiced.service" in result["after"]


def test_validate_sidecar_service_state_rejects_deprecated_daemon_unit():
    script = _load_script_module()

    with pytest.raises(SystemExit, match="deprecated voice-daemon.service"):
        script.validate_sidecar_service_state(
            {
                "ActiveState": "active",
                "MainPID": "42",
                "After": "network.target voice-daemon.service",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=None,
            voice_repo=None,
            sidecar_url=None,
            voice_daemon_service="voiced.service",
        )


def test_validate_sidecar_service_state_requires_voiced_ordering():
    script = _load_script_module()

    with pytest.raises(SystemExit, match="does not order after voiced.service"):
        script.validate_sidecar_service_state(
            {
                "ActiveState": "active",
                "MainPID": "42",
                "After": "network.target",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=None,
            voice_repo=None,
            sidecar_url=None,
            voice_daemon_service="voiced.service",
        )


def test_validate_sidecar_service_state_rejects_stale_voice_bin(tmp_path: Path):
    script = _load_script_module()
    expected_voice = tmp_path / "expected" / "voice"
    stale_voice = tmp_path / "stale" / "voice"
    expected_voice.parent.mkdir()
    stale_voice.parent.mkdir()
    expected_voice.write_text("", encoding="utf-8")
    stale_voice.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="VOICE_BIN"):
        script.validate_sidecar_service_state(
            {
                "ActiveState": "active",
                "MainPID": "42",
                "After": "network.target voiced.service",
                "Environment": f"VOICE_BIN={stale_voice}",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=str(expected_voice),
            voice_repo=None,
            sidecar_url=None,
            voice_daemon_service="voiced.service",
        )


def test_validate_sidecar_service_state_rejects_wrong_voice_repo(tmp_path: Path):
    script = _load_script_module()
    voice_repo = tmp_path / "voice"
    stale_repo = tmp_path / "old-voice"
    sidecar_path = voice_repo / "examples" / "webrtc-sidecar" / "sidecar.py"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text("", encoding="utf-8")
    stale_repo.mkdir()

    with pytest.raises(SystemExit, match="WorkingDirectory"):
        script.validate_sidecar_service_state(
            {
                "ActiveState": "active",
                "MainPID": "42",
                "After": "network.target voiced.service",
                "Environment": "",
                "WorkingDirectory": str(stale_repo),
                "ExecStart": f"{{ argv[]={sys.executable} {sidecar_path} }}",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=None,
            voice_repo=voice_repo,
            sidecar_url=None,
            voice_daemon_service="voiced.service",
        )


def test_validate_sidecar_service_state_rejects_wrong_bind_port(tmp_path: Path):
    script = _load_script_module()

    with pytest.raises(SystemExit, match="expected sidecar URL"):
        script.validate_sidecar_service_state(
            {
                "ActiveState": "active",
                "MainPID": "42",
                "After": "network.target voiced.service",
                "Environment": "",
                "ExecStart": "{ argv[]=/python sidecar.py --host 127.0.0.1 --port 87870 }",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=None,
            voice_repo=None,
            sidecar_url="http://127.0.0.1:8787",
            voice_daemon_service="voiced.service",
        )


def test_validate_sidecar_service_state_accepts_custom_daemon_ordering():
    script = _load_script_module()

    result = script.validate_sidecar_service_state(
        {
            "ActiveState": "active",
            "MainPID": "42",
            "After": "network.target custom-voiced.service",
        },
        service="voice-webrtc-sidecar.service",
        voice_bin=None,
        voice_repo=None,
        sidecar_url=None,
        voice_daemon_service="custom-voiced.service",
    )

    assert "custom-voiced.service" in result["after"]


def test_validate_voice_daemon_service_state_accepts_expected_unit(tmp_path: Path):
    script = _load_script_module()
    voice_bin = tmp_path / "bin" / "voice"
    voice_bin.parent.mkdir()
    voice_bin.write_text("", encoding="utf-8")

    result = script.validate_voice_daemon_service_state(
        {
            "ActiveState": "active",
            "MainPID": "24",
            "ExecStart": f"{{ argv[]={voice_bin} daemon start --tts-only }}",
        },
        service="voiced.service",
        voice_bin=str(voice_bin),
    )

    assert result["service"] == "voiced.service"
    assert result["pid"] == 24
    assert "daemon start" in result["exec_start"]


def test_validate_voice_daemon_service_state_rejects_wrong_voice_bin(tmp_path: Path):
    script = _load_script_module()
    expected_voice = tmp_path / "expected" / "voice"
    stale_voice = tmp_path / "stale" / "voice"
    expected_voice.parent.mkdir()
    stale_voice.parent.mkdir()
    expected_voice.write_text("", encoding="utf-8")
    stale_voice.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="expected voice binary"):
        script.validate_voice_daemon_service_state(
            {
                "ActiveState": "active",
                "MainPID": "24",
                "ExecStart": f"{{ argv[]={stale_voice} daemon start --tts-only }}",
            },
            service="voiced.service",
            voice_bin=str(expected_voice),
        )


def test_validate_voice_daemon_service_state_rejects_non_daemon_command():
    script = _load_script_module()

    with pytest.raises(SystemExit, match="not a voice daemon start command"):
        script.validate_voice_daemon_service_state(
            {
                "ActiveState": "active",
                "MainPID": "24",
                "ExecStart": "{ argv[]=/usr/bin/voice say hello }",
            },
            service="voiced.service",
            voice_bin="/usr/bin/voice",
        )


def test_validate_calling_sidecar_contract_requires_voice_pcm_shape():
    script = _load_script_module()
    contract = _voice_sidecar_contract()

    result = script.validate_calling_sidecar_contract(contract)

    assert result["contract"] == "voice.webrtc_sidecar"
    assert result["audio"]["frame_bytes"] == 1_920
    assert result["required_sections"]["endpoints"] == list(script.REQUIRED_ENDPOINTS)

    bad = {**contract, "contract": "other"}
    with pytest.raises(SystemExit, match="contract id"):
        script.validate_calling_sidecar_contract(bad)
    drifted = {
        **contract,
        "audio": {**contract["audio"], "sample_rate": 16_000},
    }
    with pytest.raises(SystemExit, match="audio contract"):
        script.validate_calling_sidecar_contract(drifted)


def test_validate_calling_sidecar_contract_requires_named_surfaces_and_endpoints():
    script = _load_script_module()
    contract = _voice_sidecar_contract()

    missing_surface = {
        **contract,
        "voice_surfaces": {
            key: value
            for key, value in contract["voice_surfaces"].items()
            if key != "raw_inbound_pcm"
        },
    }
    with pytest.raises(SystemExit, match="voice_surfaces missing keys"):
        script.validate_calling_sidecar_contract(missing_surface)

    missing_clear = {
        **contract,
        "endpoints": {
            key: value
            for key, value in contract["endpoints"].items()
            if key != "clear_audio"
        },
    }
    with pytest.raises(SystemExit, match="endpoints missing keys"):
        script.validate_calling_sidecar_contract(missing_clear)

    missing_clear_payload = {
        **contract,
        "payloads": {
            key: value
            for key, value in contract["payloads"].items()
            if key != "clear_audio_response"
        },
    }
    with pytest.raises(SystemExit, match="payloads missing keys"):
        script.validate_calling_sidecar_contract(missing_clear_payload)

    missing_queue_ms = {
        **contract,
        "payloads": {
            **contract["payloads"],
            "call_state": {
                key: value
                for key, value in contract["payloads"]["call_state"].items()
                if key != "queued_tx_ms"
            },
        },
    }
    with pytest.raises(SystemExit, match="payloads.call_state missing fields"):
        script.validate_calling_sidecar_contract(missing_queue_ms)

    missing_readiness = {
        **contract,
        "payloads": {
            **contract["payloads"],
            "call_state": {
                key: value
                for key, value in contract["payloads"]["call_state"].items()
                if key != "ready_for_accept"
            },
        },
    }
    with pytest.raises(SystemExit, match="payloads.call_state missing fields"):
        script.validate_calling_sidecar_contract(missing_readiness)


def test_load_voice_stream_contract_runs_voice_binary(monkeypatch):
    script = _load_script_module()
    contract = _voice_sidecar_contract()
    calls = []

    def fake_run(command, *, timeout):
        calls.append((command, timeout))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(contract),
            stderr="",
        )

    monkeypatch.setattr(script, "run_command", fake_run)

    result = script.load_voice_stream_contract("/usr/bin/voice", timeout=3)

    assert result == contract
    assert calls == [(["/usr/bin/voice", "stream-contract"], 3)]


def test_compare_voice_and_sidecar_contracts_requires_exact_machine_contract():
    script = _load_script_module()
    contract = _voice_sidecar_contract()

    result = script.compare_voice_and_sidecar_contracts(
        voice_contract=contract,
        sidecar_contract=dict(contract),
    )

    assert result["success"] is True
    assert result["matched_keys"] == list(script.CONTRACT_COMPARE_KEYS)
    assert result["required_sections"]["payloads"] == list(script.REQUIRED_PAYLOADS)

    drifted = {
        **contract,
        "voice_surfaces": {"raw_outbound_pcm": {"frame_bytes": 960}},
    }
    with pytest.raises(SystemExit, match="voice_surfaces"):
        script.compare_voice_and_sidecar_contracts(
            voice_contract=contract,
            sidecar_contract=drifted,
        )


def test_get_calling_sidecar_status_fetches_contract_and_health(monkeypatch):
    script = _load_script_module()
    calls = []

    def fake_json_url(target, *, timeout):
        calls.append((target, timeout))
        if target.endswith("/contract"):
            return _voice_sidecar_contract()
        return {"ok": True, "sessions": 0, "call_ids": []}

    monkeypatch.setattr(script, "get_json_url", fake_json_url)

    result = script.get_calling_sidecar_status("http://127.0.0.1:8787/", timeout=3)

    assert result["contract"]["contract"] == "voice.webrtc_sidecar"
    assert result["health"] == {"ok": True, "sessions": 0, "call_ids": []}
    assert calls == [
        ("http://127.0.0.1:8787/contract", 3),
        ("http://127.0.0.1:8787/health", 3),
    ]


def test_run_calling_sidecar_offer_smoke_requires_ready_answer(monkeypatch):
    script = _load_script_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "status": 200,
                    "body": {
                        "state": {
                            "ready_for_accept": True,
                            "readiness": {
                                "not_closed": True,
                                "local_sdp_answer": True,
                                "signaling_stable": True,
                                "ice_gathering_complete": True,
                                "outbound_audio_track": True,
                            },
                        }
                    },
                    "close": {"call_id": "call-1", "closed": True},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    result = script.run_calling_sidecar_offer_smoke(
        python_bin="/tmp/webrtc-python",
        sidecar_url="http://127.0.0.1:8787/",
        call_id="call-1",
        timeout=3,
    )

    assert result["success"] is True
    assert result["ready_for_accept"] is True
    assert result["readiness"]["outbound_audio_track"] is True
    assert calls[0][0][0] == "/tmp/webrtc-python"
    assert calls[0][0][1] == "-c"
    assert calls[0][0][-3:] == ["http://127.0.0.1:8787/", "call-1", "3"]


def test_run_calling_sidecar_offer_smoke_rejects_failed_readiness(monkeypatch):
    script = _load_script_module()

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "status": 200,
                    "body": {
                        "state": {
                            "ready_for_accept": False,
                            "readiness": {
                                "not_closed": True,
                                "local_sdp_answer": True,
                                "signaling_stable": True,
                                "ice_gathering_complete": False,
                                "outbound_audio_track": True,
                            },
                        }
                    },
                    "close": {"call_id": "call-1", "closed": True},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="ice_gathering_complete"):
        script.run_calling_sidecar_offer_smoke(
            python_bin="/tmp/webrtc-python",
            sidecar_url="http://127.0.0.1:8787",
            call_id="call-1",
            timeout=3,
        )


def _successful_live_sidecar_result(**overrides):
    result = {
        "success": True,
        "graph_actions": ["pre_accept", "accept"],
        "sidecar_ready_for_accept": True,
        "sidecar_readiness": {
            "not_closed": True,
            "local_sdp_answer": True,
            "signaling_stable": True,
            "ice_gathering_complete": True,
            "outbound_audio_track": True,
        },
        "webhook_statuses": {"connect": 200, "terminate": 200},
        "outbound_webrtc_bytes": 23040,
        "inbound_drain_bytes": 1920,
        "clear_audio": {
            "dropped_tx_bytes": 1920,
            "queued_tx_bytes": 0,
        },
        "sidecar_close": {"closed": True},
    }
    result.update(overrides)
    return result


def test_run_calling_live_sidecar_smoke_uses_live_root_imports(
    tmp_path: Path,
    monkeypatch,
):
    script = _load_script_module()
    live_root = tmp_path / "live"
    hermes_home = tmp_path / ".hermes"
    voice_repo = tmp_path / "voice"
    live_root.mkdir()
    hermes_home.mkdir()
    voice_repo.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_successful_live_sidecar_result()),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    result = script.run_calling_live_sidecar_smoke(
        python_bin="/tmp/voice-webrtc-venv/bin/python",
        live_root=live_root,
        hermes_home=hermes_home,
        voice_repo=voice_repo,
        sidecar_url="http://127.0.0.1:8787",
        timeout=12,
    )

    assert result["graph_actions"] == ["pre_accept", "accept"]
    command, kwargs = calls[0]
    assert command[:2] == [
        "/tmp/voice-webrtc-venv/bin/python",
        str(SCRIPT_PATH.parent / "verify_voice_whatsapp_calling_live_sidecar.py"),
    ]
    assert command[-6:] == [
        "--voice-repo",
        str(voice_repo),
        "--timeout",
        "12",
        "--sidecar-url",
        "http://127.0.0.1:8787",
    ]
    assert kwargs["env"]["PYTHONPATH"] == str(live_root)
    assert kwargs["env"]["HERMES_HOME"] == str(hermes_home)
    assert kwargs["cwd"] == str(live_root)


def test_run_calling_live_sidecar_smoke_rejects_missing_audio(monkeypatch, tmp_path):
    script = _load_script_module()

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                _successful_live_sidecar_result(outbound_webrtc_bytes=0)
            ),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="did not move audio"):
        script.run_calling_live_sidecar_smoke(
            python_bin="/tmp/voice-webrtc-venv/bin/python",
            live_root=tmp_path,
            hermes_home=tmp_path,
            voice_repo=tmp_path,
            sidecar_url=None,
            timeout=12,
        )


def test_run_calling_live_sidecar_smoke_rejects_missing_readiness(
    monkeypatch,
    tmp_path,
):
    script = _load_script_module()

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                _successful_live_sidecar_result(sidecar_readiness=None)
            ),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="did not report readiness"):
        script.run_calling_live_sidecar_smoke(
            python_bin="/tmp/voice-webrtc-venv/bin/python",
            live_root=tmp_path,
            hermes_home=tmp_path,
            voice_repo=tmp_path,
            sidecar_url=None,
            timeout=12,
        )


def test_run_calling_live_sidecar_smoke_rejects_uncleared_audio(
    monkeypatch,
    tmp_path,
):
    script = _load_script_module()

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                _successful_live_sidecar_result(
                    clear_audio={
                        "dropped_tx_bytes": 0,
                        "queued_tx_bytes": 1920,
                    }
                )
            ),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="did not clear outbound audio"):
        script.run_calling_live_sidecar_smoke(
            python_bin="/tmp/voice-webrtc-venv/bin/python",
            live_root=tmp_path,
            hermes_home=tmp_path,
            voice_repo=tmp_path,
            sidecar_url=None,
            timeout=12,
        )


def test_run_calling_live_sidecar_smoke_rejects_missing_close(
    monkeypatch,
    tmp_path,
):
    script = _load_script_module()

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_successful_live_sidecar_result(sidecar_close=None)),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="did not close sidecar cleanly"):
        script.run_calling_live_sidecar_smoke(
            python_bin="/tmp/voice-webrtc-venv/bin/python",
            live_root=tmp_path,
            hermes_home=tmp_path,
            voice_repo=tmp_path,
            sidecar_url=None,
            timeout=12,
        )


def test_parse_ffprobe_json_extracts_audio_shape():
    script = _load_script_module()

    parsed = script.parse_ffprobe_json(
        json.dumps(
            {
                "streams": [
                    {
                        "codec_name": "opus",
                        "sample_rate": "48000",
                        "channels": 1,
                    }
                ]
            }
        )
    )

    assert parsed == {"codec_name": "opus", "sample_rate": "48000", "channels": "1"}

    with pytest.raises(ValueError, match="no streams"):
        script.parse_ffprobe_json(json.dumps({"streams": []}))


def test_import_smoke_runs_from_live_root(tmp_path: Path, monkeypatch):
    script = _load_script_module()
    live_root = tmp_path / "live"
    module_path = live_root / "tools" / "tts_tool.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")

    def fake_run(_command, **kwargs):
        assert kwargs["cwd"] == str(live_root)
        return subprocess.CompletedProcess(
            _command,
            0,
            stdout=json.dumps({"tools.tts_tool": str(module_path)}),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    result = script.import_smoke(
        python_bin="/python",
        live_root=live_root,
        hermes_home=tmp_path / "home",
        timeout=1.0,
    )

    assert result == {"tools.tts_tool": str(module_path)}


def test_run_tts_smoke_runs_from_live_root(tmp_path: Path, monkeypatch):
    script = _load_script_module()
    live_root = tmp_path / "live"
    audio_path = tmp_path / "speech.ogg"
    live_root.mkdir()
    audio_path.write_bytes(b"OggS")

    def fake_run(_command, **kwargs):
        assert kwargs["cwd"] == str(live_root)
        return subprocess.CompletedProcess(
            _command,
            0,
            stdout=json.dumps(
                {
                    "success": True,
                    "voice_compatible": True,
                    "media_tag": f"[[audio_as_voice]]\nMEDIA:{audio_path}",
                    "file_path": str(audio_path),
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        script,
        "probe_audio",
        lambda *_args, **_kwargs: {
            "codec_name": "opus",
            "sample_rate": "48000",
            "channels": "1",
        },
    )

    result = script.run_tts_smoke(
        python_bin="/python",
        live_root=live_root,
        hermes_home=tmp_path / "home",
        platform="whatsapp",
        text="hello",
        ffprobe_bin="/ffprobe",
        timeout=1.0,
    )

    assert result["probe"]["codec_name"] == "opus"


def test_run_stt_smoke_runs_existing_config_verifier(tmp_path: Path, monkeypatch):
    script = _load_script_module()
    live_root = tmp_path / "live"
    verifier = live_root / "scripts" / "verify_voice_command_stt.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("", encoding="utf-8")

    def fake_run(command, *, timeout):
        assert command[:2] == ["/python", str(verifier)]
        assert "--use-existing-config" in command
        assert "--hermes-home" in command
        assert str(tmp_path / "home") in command
        assert "--voice-bin" in command
        assert "/voice" in command
        assert "--provider" in command
        assert "voice" in command
        assert "--expect-word" in command
        assert timeout == 7.0
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "success": True,
                    "provider": "voice",
                    "transcript": "Hello world.",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(script, "run_command", fake_run)

    result = script.run_stt_smoke(
        python_bin="/python",
        live_root=live_root,
        hermes_home=tmp_path / "home",
        voice_bin="/voice",
        provider="voice",
        text="hello world",
        expect_words=["hello"],
        stt_timeout=3.0,
        generate_timeout=4.0,
        process_timeout=7.0,
    )

    assert result["transcript"] == "Hello world."


def test_main_skips_ffprobe_when_tts_smoke_is_disabled(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    script = _load_script_module()
    live_root = tmp_path / "hermes"
    _write_voice_native_root(live_root)
    bridge_bin = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"

    labels = []

    def fake_resolve(value, *, label):
        labels.append(label)
        if label == "ffprobe":
            raise AssertionError("ffprobe should only be needed for TTS smoke")
        return value

    monkeypatch.setattr(script, "resolve_executable", fake_resolve)
    monkeypatch.setattr(
        script,
        "get_service_state",
        lambda *_args, **_kwargs: {
            "ActiveState": "active",
            "MainPID": "123",
            "Environment": (
                f"PYTHONPATH={live_root} PATH=/usr/bin:{bridge_bin}"
            ),
            "DropInPaths": "/drop-in.conf",
        },
    )
    monkeypatch.setattr(
        script,
        "read_process_env",
        lambda _pid: {"PYTHONPATH": str(live_root), "PATH": f"/usr/bin:{bridge_bin}"},
    )
    monkeypatch.setattr(
        script,
        "import_smoke",
        lambda **_kwargs: {"tools.tts_tool": str(live_root / "tools" / "tts_tool.py")},
    )
    monkeypatch.setattr(
        script,
        "get_bridge_health",
        lambda *_args, **_kwargs: {"status": "connected", "queueLength": 0},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_live_gateway.py",
            "--live-hermes-root",
            str(live_root),
            "--python-bin",
            "/python",
        ],
    )

    assert script.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["success"] is True
    assert labels == ["Hermes Python"]


def test_main_can_skip_bridge_health_for_cloud_only_gateway(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    script = _load_script_module()
    live_root = tmp_path / "hermes"
    _write_voice_native_root(live_root)
    bridge_bin = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"

    monkeypatch.setattr(script, "resolve_executable", lambda value, *, label: value)
    monkeypatch.setattr(
        script,
        "get_service_state",
        lambda *_args, **_kwargs: {
            "ActiveState": "active",
            "MainPID": "123",
            "Environment": (
                f"PYTHONPATH={live_root} PATH=/usr/bin:{bridge_bin}"
            ),
            "DropInPaths": "/drop-in.conf",
        },
    )
    monkeypatch.setattr(
        script,
        "read_process_env",
        lambda _pid: {"PYTHONPATH": str(live_root), "PATH": f"/usr/bin:{bridge_bin}"},
    )
    monkeypatch.setattr(
        script,
        "import_smoke",
        lambda **_kwargs: {"tools.tts_tool": str(live_root / "tools" / "tts_tool.py")},
    )

    def fail_bridge_health(*_args, **_kwargs):
        raise AssertionError("bridge health should be skipped")

    monkeypatch.setattr(script, "get_bridge_health", fail_bridge_health)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_live_gateway.py",
            "--live-hermes-root",
            str(live_root),
            "--python-bin",
            "/python",
            "--skip-bridge-health",
        ],
    )

    assert script.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["checks"]["bridge_health"] == {
        "success": True,
        "skipped": True,
        "reason": "--skip-bridge-health was provided",
    }


def test_main_can_require_whatsapp_cloud_readiness_without_printing_secrets(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    script = _load_script_module()
    live_root = tmp_path / "hermes"
    hermes_home = tmp_path / ".hermes"
    _write_voice_native_root(live_root)
    hermes_home.mkdir()
    bridge_bin = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"
    access_token = "EAA" + ("x" * 120)
    app_secret = "0123456789abcdef0123456789abcdef"
    verify_token = "verify-token-value-long-enough"
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                f"WHATSAPP_CLOUD_ACCESS_TOKEN={access_token}",
                f"WHATSAPP_CLOUD_APP_SECRET={app_secret}",
                f"WHATSAPP_CLOUD_VERIFY_TOKEN={verify_token}",
                "WHATSAPP_CLOUD_ALLOWED_USERS=15551234567",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(script, "resolve_executable", lambda value, *, label: value)
    monkeypatch.setattr(
        script,
        "get_service_state",
        lambda *_args, **_kwargs: {
            "ActiveState": "active",
            "MainPID": "123",
            "Environment": (
                f"PYTHONPATH={live_root} PATH=/usr/bin:{bridge_bin}"
            ),
            "DropInPaths": "/drop-in.conf",
        },
    )
    monkeypatch.setattr(
        script,
        "read_process_env",
        lambda _pid: {"PYTHONPATH": str(live_root), "PATH": f"/usr/bin:{bridge_bin}"},
    )
    monkeypatch.setattr(
        script,
        "import_smoke",
        lambda **_kwargs: {"tools.tts_tool": str(live_root / "tools" / "tts_tool.py")},
    )
    monkeypatch.setattr(
        script,
        "get_bridge_health",
        lambda *_args, **_kwargs: {"status": "connected"},
    )
    monkeypatch.setattr(
        script,
        "get_json_url",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "platform": "whatsapp_cloud",
            "phone_number_id": "7794189252778687",
            "webhook_path": "/whatsapp/webhook",
            "verify_token_configured": True,
            "app_secret_configured": True,
            "calling_sidecar_configured": False,
            "calling_sidecar_contract_loaded": False,
            "calling_sidecar_tts_stream_configured": False,
        },
    )

    def fake_get_text_url(target, *, timeout, label):
        assert verify_token in target
        assert verify_token not in label
        return 200, "hermes-local-cloud-verify-smoke"

    monkeypatch.setattr(script, "get_text_url", fake_get_text_url)
    monkeypatch.setattr(
        script,
        "post_json_url",
        lambda *_args, **_kwargs: (200, ""),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_live_gateway.py",
            "--live-hermes-root",
            str(live_root),
            "--hermes-home",
            str(hermes_home),
            "--python-bin",
            "/python",
            "--require-whatsapp-cloud-readiness",
        ],
    )

    assert script.main() == 0

    output_text = capsys.readouterr().out
    output = json.loads(output_text)
    readiness = output["checks"]["whatsapp_cloud_readiness"]
    assert readiness["required_fields"]["WHATSAPP_CLOUD_ACCESS_TOKEN"] == {
        "present": True,
        "source_shape": "meta_access_token",
        "source": "env_file",
    }
    assert readiness["authorization"]["allowed_users_count"] == 1
    assert output["checks"]["whatsapp_cloud_health"] == {
        "url": "http://127.0.0.1:8090/health",
        "status": "ok",
        "platform": "whatsapp_cloud",
        "webhook_path": "/whatsapp/webhook",
        "verify_token_configured": True,
        "app_secret_configured": True,
        "calling_sidecar_configured": False,
        "calling_sidecar_contract_loaded": False,
        "calling_sidecar_tts_stream_configured": False,
    }
    assert output["checks"]["whatsapp_cloud_verify_handshake"] == {
        "url": "http://127.0.0.1:8090/whatsapp/webhook",
        "status": 200,
        "challenge_echoed": True,
    }
    assert output["checks"]["whatsapp_cloud_signed_post"] == {
        "url": "http://127.0.0.1:8090/whatsapp/webhook",
        "status": 200,
        "payload": "status_delivery_receipt",
        "signature_accepted": True,
    }
    assert access_token not in output_text
    assert app_secret not in output_text
    assert verify_token not in output_text


def test_main_validates_calling_sidecar_when_requested(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    script = _load_script_module()
    live_root = tmp_path / "hermes"
    _write_voice_native_root(live_root)
    bridge_bin = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"
    sidecar_url = "http://127.0.0.1:8787"
    stream_command = (
        "voice stream --raw-output - --input-file {input_path} "
        "--sample-rate {sample_rate} --frame-ms {frame_ms}"
    )

    monkeypatch.setattr(script, "resolve_executable", lambda value, *, label: value)
    monkeypatch.setattr(
        script,
        "get_service_state",
        lambda *_args, **_kwargs: {
            "ActiveState": "active",
            "MainPID": "123",
            "Environment": (
                f"PYTHONPATH={live_root} PATH=/usr/bin:{bridge_bin}"
            ),
            "DropInPaths": "/drop-in.conf",
        },
    )
    monkeypatch.setattr(
        script,
        "read_process_env",
        lambda _pid: {
            "PYTHONPATH": str(live_root),
            "PATH": f"/usr/bin:{bridge_bin}",
            "WHATSAPP_CLOUD_CALLING_SIDECAR_URL": sidecar_url,
            "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND": stream_command,
            "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT": "180",
        },
    )
    monkeypatch.setattr(
        script,
        "import_smoke",
        lambda **_kwargs: {"tools.tts_tool": str(live_root / "tools" / "tts_tool.py")},
    )
    monkeypatch.setattr(
        script,
        "get_bridge_health",
        lambda *_args, **_kwargs: {"status": "connected"},
    )
    monkeypatch.setattr(
        script,
        "get_calling_sidecar_status",
        lambda *_args, **_kwargs: {
            "contract": {"contract": "voice.webrtc_sidecar"},
            "health": {"ok": True, "sessions": 0, "call_ids": []},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_live_gateway.py",
            "--live-hermes-root",
            str(live_root),
            "--python-bin",
            "/python",
            "--calling-sidecar-url",
            sidecar_url,
        ],
    )

    assert script.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["checks"]["calling_sidecar"]["env"]["url"] == sidecar_url
    assert output["checks"]["calling_sidecar"]["sidecar"]["health"]["ok"] is True


def test_main_runs_calling_live_sidecar_smoke_when_requested(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    script = _load_script_module()
    live_root = tmp_path / "hermes"
    voice_repo = tmp_path / "voice"
    _write_voice_native_root(live_root)
    voice_repo.mkdir()
    bridge_bin = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"

    monkeypatch.setattr(script, "resolve_executable", lambda value, *, label: value)
    monkeypatch.setattr(
        script,
        "get_service_state",
        lambda *_args, **_kwargs: {
            "ActiveState": "active",
            "MainPID": "123",
            "Environment": f"PYTHONPATH={live_root} PATH=/usr/bin:{bridge_bin}",
            "DropInPaths": "/drop-in.conf",
        },
    )
    monkeypatch.setattr(
        script,
        "read_process_env",
        lambda _pid: {"PYTHONPATH": str(live_root), "PATH": f"/usr/bin:{bridge_bin}"},
    )
    monkeypatch.setattr(
        script,
        "import_smoke",
        lambda **_kwargs: {"tools.tts_tool": str(live_root / "tools" / "tts_tool.py")},
    )
    monkeypatch.setattr(
        script,
        "get_bridge_health",
        lambda *_args, **_kwargs: {"status": "connected"},
    )

    def fake_live_sidecar_smoke(**kwargs):
        assert kwargs["python_bin"] == "/tmp/voice-webrtc-venv/bin/python"
        assert kwargs["live_root"] == live_root.resolve()
        assert kwargs["voice_repo"] == voice_repo.resolve()
        assert kwargs["sidecar_url"] is None
        return _successful_live_sidecar_result()

    monkeypatch.setattr(
        script,
        "run_calling_live_sidecar_smoke",
        fake_live_sidecar_smoke,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_live_gateway.py",
            "--live-hermes-root",
            str(live_root),
            "--python-bin",
            "/python",
            "--voice-repo",
            str(voice_repo),
            "--webrtc-python-bin",
            "/tmp/voice-webrtc-venv/bin/python",
            "--run-calling-live-sidecar-smoke",
        ],
    )

    assert script.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["checks"]["calling_live_sidecar_smoke"]["graph_actions"] == [
        "pre_accept",
        "accept",
    ]


def test_main_passes_calling_sidecar_url_to_live_sidecar_smoke(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    script = _load_script_module()
    live_root = tmp_path / "hermes"
    voice_repo = tmp_path / "voice"
    _write_voice_native_root(live_root)
    voice_repo.mkdir()
    bridge_bin = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"
    sidecar_url = "http://127.0.0.1:8787"

    monkeypatch.setattr(script, "resolve_executable", lambda value, *, label: value)
    monkeypatch.setattr(
        script,
        "get_service_state",
        lambda *_args, **_kwargs: {
            "ActiveState": "active",
            "MainPID": "123",
            "Environment": f"PYTHONPATH={live_root} PATH=/usr/bin:{bridge_bin}",
            "DropInPaths": "/drop-in.conf",
        },
    )
    monkeypatch.setattr(
        script,
        "read_process_env",
        lambda _pid: {
            "PYTHONPATH": str(live_root),
            "PATH": f"/usr/bin:{bridge_bin}",
            "WHATSAPP_CLOUD_CALLING_SIDECAR_URL": sidecar_url,
            "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND": (
                "voice stream --raw-output - --input-file {input_path} "
                "--sample-rate {sample_rate} --frame-ms {frame_ms}"
            ),
        },
    )
    monkeypatch.setattr(
        script,
        "import_smoke",
        lambda **_kwargs: {"tools.tts_tool": str(live_root / "tools" / "tts_tool.py")},
    )
    monkeypatch.setattr(
        script,
        "get_calling_sidecar_status",
        lambda *_args, **_kwargs: {
            "contract": {"contract": "voice.webrtc_sidecar"},
            "health": {"ok": True, "sessions": 0, "call_ids": []},
        },
    )
    monkeypatch.setattr(
        script,
        "get_bridge_health",
        lambda *_args, **_kwargs: {"status": "connected"},
    )

    def fake_live_sidecar_smoke(**kwargs):
        assert kwargs["sidecar_url"] == sidecar_url
        return _successful_live_sidecar_result()

    monkeypatch.setattr(
        script,
        "run_calling_live_sidecar_smoke",
        fake_live_sidecar_smoke,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_live_gateway.py",
            "--live-hermes-root",
            str(live_root),
            "--python-bin",
            "/python",
            "--voice-repo",
            str(voice_repo),
            "--webrtc-python-bin",
            "/tmp/voice-webrtc-venv/bin/python",
            "--calling-sidecar-url",
            sidecar_url,
            "--run-calling-live-sidecar-smoke",
        ],
    )

    assert script.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["checks"]["calling_sidecar"]["env"]["url"] == sidecar_url


def test_main_validates_sidecar_service_when_requested(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    script = _load_script_module()
    live_root = tmp_path / "hermes"
    voice_repo = tmp_path / "voice"
    sidecar_path = voice_repo / "examples" / "webrtc-sidecar" / "sidecar.py"
    voice_bin = tmp_path / "bin" / "voice"
    _write_voice_native_root(live_root)
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text("", encoding="utf-8")
    voice_bin.parent.mkdir()
    voice_bin.write_text("", encoding="utf-8")
    bridge_bin = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"

    monkeypatch.setattr(script, "resolve_executable", lambda value, *, label: value)

    def fake_service_state(service, *, timeout):
        if service == "voiced.service":
            return {
                "ActiveState": "active",
                "MainPID": "789",
                "ExecStart": f"{{ argv[]={voice_bin} daemon start --tts-only }}",
            }
        if service == "voice-webrtc-sidecar.service":
            return {
                "ActiveState": "active",
                "MainPID": "456",
                "After": "network.target voiced.service",
                "Environment": f"VOICE_BIN={voice_bin}",
                "WorkingDirectory": str(voice_repo),
                "ExecStart": f"{{ argv[]=/python {sidecar_path} --port 8787 }}",
            }
        return {
            "ActiveState": "active",
            "MainPID": "123",
            "Environment": f"PYTHONPATH={live_root} PATH=/usr/bin:{bridge_bin}",
            "DropInPaths": "/drop-in.conf",
        }

    monkeypatch.setattr(script, "get_service_state", fake_service_state)
    monkeypatch.setattr(
        script,
        "read_process_env",
        lambda _pid: {"PYTHONPATH": str(live_root), "PATH": f"/usr/bin:{bridge_bin}"},
    )
    monkeypatch.setattr(
        script,
        "import_smoke",
        lambda **_kwargs: {"tools.tts_tool": str(live_root / "tools" / "tts_tool.py")},
    )
    monkeypatch.setattr(
        script,
        "get_bridge_health",
        lambda *_args, **_kwargs: {"status": "connected"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_live_gateway.py",
            "--live-hermes-root",
            str(live_root),
            "--python-bin",
            "/python",
            "--voice-bin",
            str(voice_bin),
            "--voice-repo",
            str(voice_repo),
            "--sidecar-service",
            "voice-webrtc-sidecar.service",
        ],
    )

    assert script.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["checks"]["sidecar_service"]["pid"] == 456
    assert output["checks"]["voice_daemon_service"]["pid"] == 789
    assert output["checks"]["sidecar_service"]["voice_bin"] == str(voice_bin)
    assert output["checks"]["sidecar_service"]["working_directory"] == str(voice_repo)
