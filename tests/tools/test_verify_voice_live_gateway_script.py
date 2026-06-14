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
            "call_state": {"queued_tx_bytes": "bytes"},
            "call_status_response": "call_state",
            "close_call_response": {"closed": True},
            "send_audio_request": {"pcm_s16le_base64": "required"},
            "send_audio_response": {"accepted_bytes": "bytes"},
            "clear_audio_response": {"dropped_tx_bytes": "bytes"},
            "receive_audio_response": {"pcm_s16le_base64": "data"},
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
    )

    assert result["service"] == "voice-webrtc-sidecar.service"
    assert result["pid"] == 42
    assert result["voice_bin"] == str(voice_bin)
    assert result["working_directory"] == str(voice_repo)
    assert result["sidecar_path"] == str(sidecar_path)
    assert result["sidecar_url"] == "http://127.0.0.1:8787"
    assert result["bind"] == {"host": "127.0.0.1", "port": 8787}


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
                "Environment": f"VOICE_BIN={stale_voice}",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=str(expected_voice),
            voice_repo=None,
            sidecar_url=None,
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
                "Environment": "",
                "WorkingDirectory": str(stale_repo),
                "ExecStart": f"{{ argv[]={sys.executable} {sidecar_path} }}",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=None,
            voice_repo=voice_repo,
            sidecar_url=None,
        )


def test_validate_sidecar_service_state_rejects_wrong_bind_port(tmp_path: Path):
    script = _load_script_module()

    with pytest.raises(SystemExit, match="expected sidecar URL"):
        script.validate_sidecar_service_state(
            {
                "ActiveState": "active",
                "MainPID": "42",
                "Environment": "",
                "ExecStart": "{ argv[]=/python sidecar.py --host 127.0.0.1 --port 87870 }",
            },
            service="voice-webrtc-sidecar.service",
            voice_bin=None,
            voice_repo=None,
            sidecar_url="http://127.0.0.1:8787",
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
        if service == "voice-webrtc-sidecar.service":
            return {
                "ActiveState": "active",
                "MainPID": "456",
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
    assert output["checks"]["sidecar_service"]["voice_bin"] == str(voice_bin)
    assert output["checks"]["sidecar_service"]["working_directory"] == str(voice_repo)
