import builtins
import importlib.util
import json
from pathlib import Path
import sys

from ruamel.yaml import YAML
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "install_voice_local_stack.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "install_voice_local_stack",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_sidecar_unit_points_at_voice_repo_and_pcm_sink(tmp_path: Path):
    script = _load_script_module()
    voice_repo = tmp_path / "voice"
    rx_pcm = tmp_path / ".hermes" / "voice-webrtc-sidecar" / "inbound.s16le"

    unit = script.render_sidecar_unit(
        voice_repo=voice_repo,
        voice_bin="/home/user/.local/bin/voice",
        webrtc_python_bin="/tmp/voice-webrtc-venv/bin/python",
        voice_daemon_service="voiced.service",
        host="127.0.0.1",
        port=8787,
        rx_pcm_path=rx_pcm,
        log_level="INFO",
    )

    assert "Description=Voice WebRTC Sidecar" in unit
    assert "After=network.target voiced.service" in unit
    assert "Wants=voiced.service" in unit
    assert "voice-daemon.service" not in unit
    assert f"WorkingDirectory={voice_repo}" in unit
    assert 'Environment="VOICE_BIN=/home/user/.local/bin/voice"' in unit
    assert "/tmp/voice-webrtc-venv/bin/python" in unit
    assert str(voice_repo / "examples" / "webrtc-sidecar" / "sidecar.py") in unit
    assert "--host 127.0.0.1 --port 8787" in unit
    assert f"--rx-pcm {rx_pcm}" in unit


def test_render_hermes_dropin_exposes_voice_stream_contract(tmp_path: Path):
    script = _load_script_module()
    live_root = tmp_path / "hermes-agent"

    dropin = script.render_hermes_dropin(
        live_hermes_root=live_root,
        gateway_path="/venv/bin:/bridge/bin:/usr/bin",
        sidecar_url="http://127.0.0.1:8787",
        voice_bin="/home/user/.local/bin/voice",
        voice="af_heart",
        speed="1.0",
        stream_timeout=180.0,
    )

    assert f'Environment="PYTHONPATH={live_root}"' in dropin
    assert 'Environment="PATH=/venv/bin:/bridge/bin:/usr/bin"' in dropin
    assert (
        'Environment="WHATSAPP_CLOUD_CALLING_SIDECAR_URL=http://127.0.0.1:8787"'
        in dropin
    )
    assert "voice stream --quiet --sample-rate {sample_rate}" in dropin
    assert "--frame-ms {frame_ms} --raw-output - --input-file {input_path}" in dropin
    assert "--voice af_heart --speed 1.0" in dropin


def test_build_plan_generates_sidecar_unit_and_gateway_dropin(tmp_path: Path):
    script = _load_script_module()
    systemd_dir = tmp_path / "systemd"
    hermes_home = tmp_path / ".hermes"
    live_root = tmp_path / "hermes-agent"
    voice_repo = tmp_path / "voice"

    args = script.parse_args(
        [
            "--systemd-user-dir",
            str(systemd_dir),
            "--hermes-home",
            str(hermes_home),
            "--config-path",
            str(hermes_home / "config.yaml"),
            "--live-hermes-root",
            str(live_root),
            "--voice-repo",
            str(voice_repo),
            "--voice-bin",
            sys.executable,
            "--webrtc-python-bin",
            sys.executable,
        ]
    )

    plan = script.build_plan(args)

    assert plan["apply"] is False
    assert plan["sidecar_url"] == "http://127.0.0.1:8787"
    assert plan["voice_bin"] == sys.executable
    assert plan["webrtc_python_bin"] == sys.executable
    assert plan["voice_daemon_service"] == "voiced.service"
    assert plan["paths"]["hermes_dropin"] == str(
        systemd_dir / "hermes-gateway.service.d" / "voice-stack.conf"
    )
    assert plan["paths"]["sidecar_unit"] == str(
        systemd_dir / "voice-webrtc-sidecar.service"
    )
    assert [item["kind"] for item in plan["files"]] == [
        "systemd_user_service",
        "systemd_user_dropin",
    ]
    assert plan["tts_provider"]["output_format"] == "ogg"
    assert plan["tts_provider"]["voice_compatible"] is True
    assert plan["stt_provider"]["command"] == (
        f"{sys.executable} stream-transcribe --quiet {{input_path}}"
    )
    assert plan["stt_provider"]["format"] == "txt"
    local_stack = plan["verify_commands"]["local_stack"]
    assert local_stack[:2] == [
        sys.executable,
        str(script.repo_root() / "scripts" / "verify_voice_local_stack.py"),
    ]
    assert "--voice-bin" in local_stack
    assert sys.executable in local_stack
    assert "--voice-repo" in local_stack
    assert str(voice_repo.resolve()) in local_stack
    assert "--webrtc-python-bin" in local_stack
    assert "--live-hermes-root" in local_stack
    assert str(live_root.resolve()) in local_stack
    local_stack_live = plan["verify_commands"]["local_stack_live_gateway"]
    assert local_stack_live[: len(local_stack)] == local_stack
    assert "--run-live-gateway" in local_stack_live
    assert local_stack_live[local_stack_live.index("--calling-sidecar-url") + 1] == (
        "http://127.0.0.1:8787"
    )
    assert local_stack_live[local_stack_live.index("--live-gateway-python-bin") + 1] == (
        str(hermes_home.resolve() / "hermes-agent" / "venv" / "bin" / "python")
    )
    assert local_stack_live[
        local_stack_live.index("--live-gateway-hermes-home") + 1
    ] == str(hermes_home.resolve())
    assert "--run-live-gateway-calling-live-sidecar-smoke" in local_stack_live
    assert local_stack_live[
        local_stack_live.index("--live-gateway-sidecar-service") + 1
    ] == "voice-webrtc-sidecar.service"
    assert local_stack_live[
        local_stack_live.index("--live-gateway-voice-daemon-service") + 1
    ] == "voiced.service"
    assert "--run-live-gateway-stt-smoke" not in local_stack_live
    assert plan["verify_commands"]["local_stack_live_gateway_cloud_only"][-1] == (
        "--skip-live-gateway-bridge-health"
    )
    assert plan["verify_commands"]["local_stack_live_gateway_cloud_ready"][-2:] == [
        "--skip-live-gateway-bridge-health",
        "--require-live-gateway-whatsapp-cloud-readiness",
    ]
    live_gateway = plan["verify_commands"]["live_gateway"]
    assert live_gateway[:2] == [
        sys.executable,
        str(script.repo_root() / "scripts" / "verify_voice_live_gateway.py"),
    ]
    assert "--calling-sidecar-url" in live_gateway
    assert "http://127.0.0.1:8787" in live_gateway
    assert "--voice-bin" in live_gateway
    assert "--run-tts-smoke" in live_gateway
    assert "--run-sidecar-offer-smoke" in live_gateway
    assert "--run-calling-live-sidecar-smoke" in live_gateway
    assert (
        live_gateway[live_gateway.index("--webrtc-python-bin") + 1]
        == sys.executable
    )
    assert "--run-stt-smoke" not in live_gateway
    assert "--sidecar-service" in live_gateway
    assert "voice-webrtc-sidecar.service" in live_gateway
    assert live_gateway[live_gateway.index("--voice-daemon-service") + 1] == (
        "voiced.service"
    )
    assert "--voice-repo" in live_gateway
    assert str(voice_repo.resolve()) in live_gateway
    assert "--skip-bridge-health" not in live_gateway
    assert plan["verify_commands"]["live_gateway_cloud_only"][-1] == (
        "--skip-bridge-health"
    )
    assert plan["verify_commands"]["live_gateway_cloud_ready"][-2:] == [
        "--skip-bridge-health",
        "--require-whatsapp-cloud-readiness",
    ]


def test_build_plan_adds_live_stt_smoke_when_configuring_stt(tmp_path: Path):
    script = _load_script_module()
    args = script.parse_args(
        [
            "--systemd-user-dir",
            str(tmp_path / "systemd"),
            "--hermes-home",
            str(tmp_path / ".hermes"),
            "--live-hermes-root",
            str(tmp_path / "hermes-agent"),
            "--voice-repo",
            str(tmp_path / "voice"),
            "--voice-bin",
            sys.executable,
            "--webrtc-python-bin",
            sys.executable,
            "--configure-stt",
            "--stt-provider-name",
            "voice",
            "--stt-timeout",
            "123",
        ]
    )

    plan = script.build_plan(args)
    live_gateway = plan["verify_commands"]["live_gateway"]
    local_stack_live = plan["verify_commands"]["local_stack_live_gateway"]

    assert "--run-stt-smoke" in live_gateway
    assert live_gateway[live_gateway.index("--stt-provider") + 1] == "voice"
    assert live_gateway[live_gateway.index("--stt-timeout") + 1] == "123"
    assert "--run-live-gateway-stt-smoke" in local_stack_live


def test_build_plan_threads_custom_voice_daemon_service(tmp_path: Path):
    script = _load_script_module()
    args = script.parse_args(
        [
            "--systemd-user-dir",
            str(tmp_path / "systemd"),
            "--hermes-home",
            str(tmp_path / ".hermes"),
            "--live-hermes-root",
            str(tmp_path / "hermes-agent"),
            "--voice-repo",
            str(tmp_path / "voice"),
            "--voice-bin",
            sys.executable,
            "--webrtc-python-bin",
            sys.executable,
            "--voice-daemon-service",
            "custom-voiced.service",
        ]
    )

    plan = script.build_plan(args)
    sidecar_unit = next(
        item["content"]
        for item in plan["files"]
        if item["kind"] == "systemd_user_service"
    )

    assert plan["voice_daemon_service"] == "custom-voiced.service"
    assert "After=network.target custom-voiced.service" in sidecar_unit
    assert "Wants=custom-voiced.service" in sidecar_unit
    local_stack_live = plan["verify_commands"]["local_stack_live_gateway"]
    assert local_stack_live[
        local_stack_live.index("--live-gateway-voice-daemon-service") + 1
    ] == "custom-voiced.service"
    live_gateway = plan["verify_commands"]["live_gateway"]
    assert live_gateway[live_gateway.index("--voice-daemon-service") + 1] == (
        "custom-voiced.service"
    )


def test_configure_tts_provider_writes_voice_compatible_ogg_provider(tmp_path: Path):
    script = _load_script_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("gateway:\n  enabled: true\n", encoding="utf-8")
    provider = script.build_tts_provider(
        voice_bin="/home/user/.local/bin/voice",
        voice="af_heart",
        speed="1.0",
        timeout=180,
        max_text_length=2000,
    )

    result = script.configure_tts_provider(
        config_path=config_path,
        provider_name="kokoro",
        provider=provider,
    )

    parsed = YAML().load(config_path.read_text(encoding="utf-8"))
    configured = parsed["tts"]["providers"]["kokoro"]
    assert result == {
        "path": str(config_path),
        "provider": "kokoro",
        "output_format": "ogg",
        "voice_compatible": True,
    }
    assert parsed["gateway"]["enabled"] is True
    assert parsed["tts"]["provider"] == "kokoro"
    assert configured["command"].startswith(
        "/home/user/.local/bin/voice say --format ogg-opus"
    )
    assert configured["output_format"] == "ogg"
    assert configured["voice_compatible"] is True


def test_configure_stt_provider_writes_voice_stream_transcribe_provider(
    tmp_path: Path,
):
    script = _load_script_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "gateway:\n  enabled: true\nstt:\n  enabled: true\n  provider: local\n",
        encoding="utf-8",
    )
    provider = script.build_stt_provider(
        voice_bin="/home/user/.local/bin/voice",
        timeout=300,
    )

    result = script.configure_stt_provider(
        config_path=config_path,
        provider_name="voice",
        provider=provider,
    )

    parsed = YAML().load(config_path.read_text(encoding="utf-8"))
    configured = parsed["stt"]["providers"]["voice"]
    assert result == {
        "path": str(config_path),
        "provider": "voice",
        "format": "txt",
    }
    assert parsed["gateway"]["enabled"] is True
    assert parsed["stt"]["enabled"] is True
    assert parsed["stt"]["provider"] == "voice"
    assert configured["type"] == "command"
    assert configured["command"] == (
        "/home/user/.local/bin/voice stream-transcribe --quiet {input_path}"
    )
    assert configured["format"] == "txt"
    assert configured["timeout"] == 300


def test_configure_tts_provider_falls_back_to_pyyaml(
    tmp_path: Path,
    monkeypatch,
):
    script = _load_script_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("gateway:\n  enabled: true\n", encoding="utf-8")
    provider = script.build_tts_provider(
        voice_bin="/home/user/.local/bin/voice",
        voice="af_heart",
        speed="1.0",
        timeout=180,
        max_text_length=2000,
    )
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ruamel.yaml":
            raise ModuleNotFoundError("No module named 'ruamel'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = script.configure_tts_provider(
        config_path=config_path,
        provider_name="kokoro",
        provider=provider,
    )

    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert result["provider"] == "kokoro"
    assert parsed["gateway"]["enabled"] is True
    assert parsed["tts"]["providers"]["kokoro"]["output_format"] == "ogg"
    assert parsed["tts"]["providers"]["kokoro"]["voice_compatible"] is True


def test_apply_without_systemctl_writes_files_and_config(
    tmp_path: Path,
    capsys,
):
    script = _load_script_module()
    systemd_dir = tmp_path / "systemd"
    hermes_home = tmp_path / ".hermes"
    live_root = tmp_path / "hermes-agent"
    voice_repo = tmp_path / "voice"
    config_path = hermes_home / "config.yaml"

    exit_code = script.main(
        [
            "--apply",
            "--no-systemctl",
            "--no-start",
            "--configure-tts",
            "--configure-stt",
            "--systemd-user-dir",
            str(systemd_dir),
            "--hermes-home",
            str(hermes_home),
            "--config-path",
            str(config_path),
            "--live-hermes-root",
            str(live_root),
            "--voice-repo",
            str(voice_repo),
            "--voice-bin",
            sys.executable,
            "--webrtc-python-bin",
            sys.executable,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["success"] is True
    assert output["applied"] is True
    assert output["apply_result"]["systemctl"] == []
    assert (
        systemd_dir / "voice-webrtc-sidecar.service"
    ).read_text(encoding="utf-8").startswith("[Unit]")
    assert (
        systemd_dir / "hermes-gateway.service.d" / "voice-stack.conf"
    ).read_text(encoding="utf-8").startswith("[Service]")
    parsed = YAML().load(config_path.read_text(encoding="utf-8"))
    assert parsed["tts"]["providers"]["kokoro"]["output_format"] == "ogg"
    assert parsed["tts"]["providers"]["kokoro"]["voice_compatible"] is True
    assert parsed["stt"]["provider"] == "voice"
    assert parsed["stt"]["providers"]["voice"]["command"] == (
        f"{sys.executable} stream-transcribe --quiet {{input_path}}"
    )
    assert parsed["stt"]["providers"]["voice"]["format"] == "txt"
