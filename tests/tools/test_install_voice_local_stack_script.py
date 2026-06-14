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
        host="127.0.0.1",
        port=8787,
        rx_pcm_path=rx_pcm,
        log_level="INFO",
    )

    assert "Description=Voice WebRTC Sidecar" in unit
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
