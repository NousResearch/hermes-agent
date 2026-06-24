import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "verify_voice_local_stack.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_voice_local_stack", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "provider": "kokoro",
        "stt_provider": "voice",
        "voice": "af_heart",
        "speed": "1.0",
        "tts_timeout": 180.0,
        "stt_timeout": 300.0,
        "stream_timeout": 180.0,
        "calling_control_plane_timeout": 10.0,
        "calling_live_sidecar_timeout": 12.0,
        "full_duplex_timeout": 90.0,
        "full_duplex_max_queued_tx_ms": 1000,
        "command_text": "command smoke",
        "command_stt_text": "hello world",
        "voice_contract_text": "contract smoke",
        "voice_contract_timeout": 240.0,
        "whatsapp_bridge_media_timeout": 15.0,
        "whatsapp_cloud_webhook_timeout": 15.0,
        "whatsapp_cloud_voice_timeout": 15.0,
        "node_bin": "node",
        "stream_text": "stream smoke",
        "stream_command_template": None,
        "skip_voice_contract": False,
        "skip_whatsapp_bridge_media": False,
        "skip_whatsapp_cloud_webhook": False,
        "skip_whatsapp_cloud_voice": False,
        "skip_command_stt": False,
        "skip_calling_control_plane": False,
        "skip_calling_live_sidecar": False,
        "skip_full_duplex": False,
        "voice_repo": Path("/voice"),
        "webrtc_python_bin": None,
        "run_live_gateway": False,
        "live_gateway_python_bin": "/home/user/.hermes/hermes-agent/venv/bin/python",
        "live_gateway_hermes_home": Path("/home/user/.hermes"),
        "calling_sidecar_url": None,
        "live_gateway_sidecar_service": "voice-webrtc-sidecar.service",
        "live_gateway_voice_daemon_service": "voiced.service",
        "skip_live_gateway_bridge_health": False,
        "require_live_gateway_whatsapp_cloud_readiness": False,
        "run_live_gateway_stt_smoke": False,
        "run_live_gateway_calling_live_sidecar_smoke": False,
        "live_gateway_timeout": 360.0,
        "full_duplex_inbound_text": "hello world",
        "full_duplex_outbound_text": "hello back",
        "stt_expect_word": ["hello", "world"],
        "expect_word": ["hello", "world"],
        "keep_home": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_command_tts_command_uses_isolated_hermes_home():
    script = _load_script_module()

    command = script.command_tts_command(
        _args(keep_home=True),
        voice_bin="/tmp/voice",
        hermes_home=Path("/tmp/hermes-home"),
    )

    assert command[:2] == [sys.executable, str(script.script_path("verify_voice_command_tts.py"))]
    assert "--voice-bin" in command
    assert "/tmp/voice" in command
    assert "--hermes-home" in command
    assert "/tmp/hermes-home" in command
    assert "--force" in command
    assert "--keep-home" in command
    assert command[-2:] == ["--text", "command smoke"]


def test_command_stt_command_uses_isolated_hermes_home():
    script = _load_script_module()

    command = script.command_stt_command(
        _args(keep_home=True, stt_expect_word=["hello"]),
        voice_bin="/tmp/voice",
        hermes_home=Path("/tmp/hermes-home"),
    )

    assert command[:2] == [sys.executable, str(script.script_path("verify_voice_command_stt.py"))]
    assert "--voice-bin" in command
    assert "/tmp/voice" in command
    assert "--hermes-home" in command
    assert "/tmp/hermes-home" in command
    assert "--force" in command
    assert "--keep-home" in command
    assert "--provider" in command
    assert "voice" in command
    assert "--generate-timeout" in command
    assert command[-4:] == ["--text", "hello world", "--expect-word", "hello"]


def test_stream_tts_command_can_use_custom_template():
    script = _load_script_module()

    command = script.stream_tts_command(
        _args(stream_command_template="voice stream --raw-output -"),
        voice_bin="/tmp/voice",
    )

    assert command[:2] == [sys.executable, str(script.script_path("verify_voice_stream_tts.py"))]
    assert "--voice-bin" in command
    assert "/tmp/voice" in command
    assert "--command-template" in command
    assert "voice stream --raw-output -" in command


def test_voice_contract_command_requires_daemon_and_passes_text():
    script = _load_script_module()

    command = script.voice_contract_command(
        _args(voice_contract_text="contract text"),
        voice_bin="/tmp/voice",
        script=Path("/voice/scripts/verify_whatsapp_voice_contract.sh"),
    )

    assert command == [
        "/voice/scripts/verify_whatsapp_voice_contract.sh",
        "--voice-bin",
        "/tmp/voice",
        "--text",
        "contract text",
        "--require-daemon",
        "--run-stt-smoke",
    ]


def test_whatsapp_bridge_media_payload_command_runs_node_test():
    script = _load_script_module()

    command = script.whatsapp_bridge_media_payload_command(node_bin="/usr/bin/node")

    assert command == [
        "/usr/bin/node",
        "--test",
        str(
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "whatsapp-bridge"
            / "media-payload.test.mjs"
        ),
    ]


def test_whatsapp_cloud_voice_note_command_runs_json_verifier():
    script = _load_script_module()

    command = script.whatsapp_cloud_voice_note_command()

    assert command == [
        sys.executable,
        str(script.script_path("verify_voice_whatsapp_cloud_voice_note.py")),
    ]


def test_whatsapp_cloud_webhook_command_runs_json_verifier():
    script = _load_script_module()

    command = script.whatsapp_cloud_webhook_command()

    assert command == [
        sys.executable,
        str(script.script_path("verify_voice_whatsapp_cloud_webhook.py")),
    ]


def test_resolve_voice_contract_script_uses_voice_checkout(tmp_path: Path):
    script = _load_script_module()
    verifier = tmp_path / "scripts" / "verify_whatsapp_voice_contract.sh"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    verifier.chmod(0o755)

    resolved = script.resolve_voice_contract_script(_args(voice_repo=tmp_path))

    assert resolved == verifier


def test_resolve_voice_contract_script_can_skip_or_fail(tmp_path: Path):
    script = _load_script_module()

    assert (
        script.resolve_voice_contract_script(
            _args(skip_voice_contract=True, voice_repo=None)
        )
        is None
    )
    assert script.resolve_voice_contract_script(_args(voice_repo=None)) is None
    with pytest.raises(SystemExit, match="contract verifier not found"):
        script.resolve_voice_contract_script(_args(voice_repo=tmp_path))

    verifier = tmp_path / "scripts" / "verify_whatsapp_voice_contract.sh"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    verifier.chmod(0o644)
    with pytest.raises(SystemExit, match="not executable"):
        script.resolve_voice_contract_script(_args(voice_repo=tmp_path))


def test_full_duplex_command_passes_voice_repo_and_python():
    script = _load_script_module()

    command = script.full_duplex_command(
        _args(
            webrtc_python_bin="/tmp/venv/bin/python",
            expect_word=["testing"],
            full_duplex_max_queued_tx_ms=250,
        ),
        voice_bin="/tmp/voice",
        voice_repo=Path("/voice"),
    )

    assert command[:2] == [
        sys.executable,
        str(script.script_path("verify_voice_full_duplex_sidecar.py")),
    ]
    assert "--voice-repo" in command
    assert "/voice" in command
    assert "--python-bin" in command
    assert "/tmp/venv/bin/python" in command
    assert "--max-queued-tx-ms" in command
    assert "250" in command
    assert command[-2:] == ["--expect-word", "testing"]


def test_calling_control_plane_command_uses_synthetic_verifier():
    script = _load_script_module()

    command = script.calling_control_plane_command(
        _args(calling_control_plane_timeout=12.5)
    )

    assert command[:2] == [
        sys.executable,
        str(script.script_path("verify_voice_whatsapp_calling_control_plane.py")),
    ]
    assert command[-2:] == ["--timeout", "12.5"]


def test_calling_live_sidecar_command_uses_webrtc_python_and_voice_repo():
    script = _load_script_module()

    command = script.calling_live_sidecar_command(
        _args(
            webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
            calling_live_sidecar_timeout=9.5,
        ),
        voice_repo=Path("/voice"),
    )

    assert command == [
        "/tmp/voice-webrtc-venv/bin/python",
        str(script.script_path("verify_voice_whatsapp_calling_live_sidecar.py")),
        "--voice-repo",
        "/voice",
        "--timeout",
        "9.5",
    ]


def test_live_gateway_command_runs_offer_readiness_smoke():
    script = _load_script_module()

    command = script.live_gateway_command(
        _args(
            live_hermes_root=Path("/checkout/hermes-agent"),
            live_gateway_python_bin="/home/user/.hermes/hermes-agent/venv/bin/python",
            live_gateway_hermes_home=Path("/home/user/.hermes"),
            calling_sidecar_url="http://127.0.0.1:8787",
            webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
            run_live_gateway_stt_smoke=True,
            run_live_gateway_calling_live_sidecar_smoke=True,
            skip_live_gateway_bridge_health=True,
            require_live_gateway_whatsapp_cloud_readiness=True,
        ),
        voice_bin="/home/user/.local/bin/voice",
        voice_repo=Path("/voice"),
    )

    assert command[:2] == [
        sys.executable,
        str(script.script_path("verify_voice_live_gateway.py")),
    ]
    assert command[command.index("--live-hermes-root") + 1] == (
        str(Path("/checkout/hermes-agent").resolve())
    )
    assert command[command.index("--python-bin") + 1] == (
        "/home/user/.hermes/hermes-agent/venv/bin/python"
    )
    assert command[command.index("--hermes-home") + 1] == "/home/user/.hermes"
    assert command[command.index("--calling-sidecar-url") + 1] == (
        "http://127.0.0.1:8787"
    )
    assert command[command.index("--voice-bin") + 1] == (
        "/home/user/.local/bin/voice"
    )
    assert "--run-tts-smoke" in command
    assert "--run-sidecar-offer-smoke" in command
    assert command[command.index("--webrtc-python-bin") + 1] == (
        "/tmp/voice-webrtc-venv/bin/python"
    )
    assert "--run-stt-smoke" in command
    assert command[command.index("--stt-provider") + 1] == "voice"
    assert "--run-calling-live-sidecar-smoke" in command
    assert command[command.index("--sidecar-service") + 1] == (
        "voice-webrtc-sidecar.service"
    )
    assert command[command.index("--voice-daemon-service") + 1] == "voiced.service"
    assert command[command.index("--voice-repo") + 1] == "/voice"
    assert "--skip-bridge-health" in command
    assert "--require-whatsapp-cloud-readiness" in command


def test_live_gateway_command_passes_custom_voice_daemon_service():
    script = _load_script_module()

    command = script.live_gateway_command(
        _args(
            live_hermes_root=Path("/checkout/hermes-agent"),
            calling_sidecar_url="http://127.0.0.1:8787",
            webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
            live_gateway_voice_daemon_service="custom-voiced.service",
        ),
        voice_bin="/home/user/.local/bin/voice",
        voice_repo=Path("/voice"),
    )

    assert command[command.index("--voice-daemon-service") + 1] == (
        "custom-voiced.service"
    )


def test_live_gateway_command_requires_voice_repo_for_calling_sidecar_smoke():
    script = _load_script_module()

    with pytest.raises(SystemExit, match="requires --voice-repo"):
        script.live_gateway_command(
            _args(
                live_hermes_root=Path("/checkout/hermes-agent"),
                calling_sidecar_url="http://127.0.0.1:8787",
                webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
                run_live_gateway_calling_live_sidecar_smoke=True,
            ),
            voice_bin="/home/user/.local/bin/voice",
            voice_repo=None,
        )


def test_live_gateway_command_skips_daemon_service_without_sidecar_service():
    script = _load_script_module()

    command = script.live_gateway_command(
        _args(
            live_hermes_root=Path("/checkout/hermes-agent"),
            calling_sidecar_url="http://127.0.0.1:8787",
            webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
            live_gateway_sidecar_service="",
        ),
        voice_bin="/home/user/.local/bin/voice",
        voice_repo=Path("/voice"),
    )

    assert "--sidecar-service" not in command
    assert "--voice-daemon-service" not in command


def test_live_gateway_command_requires_live_inputs():
    script = _load_script_module()

    with pytest.raises(SystemExit, match="requires --live-hermes-root"):
        script.live_gateway_command(
            _args(
                live_hermes_root=None,
                calling_sidecar_url="http://127.0.0.1:8787",
                webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
            ),
            voice_bin="/tmp/voice",
            voice_repo=Path("/voice"),
        )
    with pytest.raises(SystemExit, match="requires --calling-sidecar-url"):
        script.live_gateway_command(
            _args(
                live_hermes_root=Path("/checkout/hermes-agent"),
                calling_sidecar_url=None,
                webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
            ),
            voice_bin="/tmp/voice",
            voice_repo=Path("/voice"),
        )
    with pytest.raises(SystemExit, match="requires --webrtc-python-bin"):
        script.live_gateway_command(
            _args(
                live_hermes_root=Path("/checkout/hermes-agent"),
                calling_sidecar_url="http://127.0.0.1:8787",
                webrtc_python_bin=None,
            ),
            voice_bin="/tmp/voice",
            voice_repo=Path("/voice"),
        )


def test_resolve_voice_repo_requires_full_duplex_checkout(tmp_path: Path):
    script = _load_script_module()

    with pytest.raises(SystemExit, match="full-duplex validation needs"):
        script.resolve_voice_repo_for_full_duplex(_args(voice_repo=None))

    assert script.resolve_voice_repo_for_full_duplex(
        _args(skip_full_duplex=True, voice_repo=None)
    ) is None

    with pytest.raises(SystemExit, match="voice full-duplex smoke not found"):
        script.resolve_voice_repo_for_full_duplex(_args(voice_repo=tmp_path))


def test_parse_json_object_accepts_progress_prefix():
    script = _load_script_module()

    parsed = script.parse_json_object('progress\n{"success": true, "value": 1}\n')

    assert parsed == {"success": True, "value": 1}


def test_run_json_step_rejects_unsuccessful_result(monkeypatch):
    script = _load_script_module()

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"success": False, "error": "bad"}),
            stderr="",
        )

    monkeypatch.setattr(script, "run_process", fake_run)

    with pytest.raises(SystemExit, match="reported failure"):
        script.run_json_step("demo", ["demo"], timeout=1.0, env={})


def test_child_env_sets_hermes_home_and_checkout_pythonpath(
    monkeypatch, tmp_path: Path
):
    script = _load_script_module()
    monkeypatch.setenv("PYTHONPATH", "/existing")

    env = script.child_env(hermes_home=tmp_path)

    assert env["HERMES_HOME"] == str(tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(Path(__file__).resolve().parents[2]),
        "/existing",
    ]


def _write_live_root(root: Path, *, include_cloud: bool = True) -> None:
    tools_dir = root / "tools"
    gateway_dir = root / "gateway" / "platforms"
    tools_dir.mkdir(parents=True)
    gateway_dir.mkdir(parents=True)
    (tools_dir / "tts_tool.py").write_text(
        "\n".join(["voice_compatible", "libopus", "-application", "voip"]),
        encoding="utf-8",
    )
    (tools_dir / "transcription_tools.py").write_text(
        "\n".join(["stt.providers", "_transcribe_command_stt", "transcribe_audio"]),
        encoding="utf-8",
    )
    if include_cloud:
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


def test_audit_live_hermes_root_accepts_voice_native_checkout(tmp_path: Path):
    script = _load_script_module()
    _write_live_root(tmp_path)

    result = script.audit_live_hermes_root(tmp_path)

    assert result["success"] is True
    assert result["failures"] == []
    assert [check["path"] for check in result["checked"]] == [
        "tools/tts_tool.py",
        "tools/transcription_tools.py",
        "gateway/platforms/whatsapp_cloud.py",
    ]


def test_audit_live_hermes_root_reports_stale_checkout(tmp_path: Path):
    script = _load_script_module()
    _write_live_root(tmp_path, include_cloud=False)

    result = script.audit_live_hermes_root(tmp_path)

    assert result["success"] is False
    assert "gateway/platforms/whatsapp_cloud.py is missing" in result["failures"]


def test_require_live_hermes_root_fails_on_missing_surfaces(tmp_path: Path):
    script = _load_script_module()

    with pytest.raises(SystemExit, match="voice-native integration surfaces"):
        script.require_live_hermes_root(tmp_path / "missing")
