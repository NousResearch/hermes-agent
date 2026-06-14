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
                "-application",
                "voip",
            ]
        ),
        encoding="utf-8",
    )


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
