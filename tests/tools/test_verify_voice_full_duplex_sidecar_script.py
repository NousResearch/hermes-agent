import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_voice_full_duplex_sidecar.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_voice_full_duplex_sidecar",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_result():
    return {
        "success": True,
        "transcript": "Hello World",
        "outbound_webrtc_bytes": 23_040,
        "decoded_pcm_bytes": 124_992,
        "audio": {
            "sample_rate": 48_000,
            "channels": 1,
            "frame_ms": 20,
            "encoding": "pcm_s16le",
            "bytes_per_sample": 2,
        },
        "queued_tx_bytes": 1_920,
        "queued_tx_ms": 20,
        "stt": {"text": "Hello World"},
    }


def test_build_smoke_command_points_at_voice_full_duplex_smoke():
    script = _load_script_module()
    smoke_path = Path("/repo/examples/webrtc-sidecar/full_duplex_loopback_smoke.py")

    command = script.build_smoke_command(
        python_bin="/tmp/venv/bin/python",
        smoke_path=smoke_path,
        voice_bin="/usr/local/bin/voice",
        inbound_text="hello world",
        outbound_text="hello back",
        voice="af_heart",
        speed="1.0",
        timeout=90.0,
        expect_words=["hello", "world"],
        max_queued_tx_ms=1_000,
    )

    assert command[:4] == [
        "/tmp/venv/bin/python",
        str(smoke_path),
        "--voice-bin",
        "/usr/local/bin/voice",
    ]
    assert "--inbound-text" in command
    assert "--outbound-text" in command
    assert "--max-queued-tx-ms" in command
    assert "1000" in command
    assert command[-4:] == ["--expect-word", "hello", "--expect-word", "world"]


def test_full_duplex_smoke_path_uses_voice_repo_layout():
    script = _load_script_module()

    assert script.full_duplex_smoke_path(Path("/voice")) == Path(
        "/voice/examples/webrtc-sidecar/full_duplex_loopback_smoke.py"
    )


def test_resolve_executable_preserves_virtualenv_symlink(tmp_path: Path):
    script = _load_script_module()
    target = tmp_path / "python3"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    symlink = tmp_path / "python"
    symlink.symlink_to(target.name)

    resolved = script.resolve_executable(str(symlink), label="python")

    assert resolved == os.path.abspath(str(symlink))
    assert Path(resolved).is_symlink()


def test_parse_smoke_json_accepts_progress_prefix():
    script = _load_script_module()
    result = _valid_result()

    parsed = script.parse_smoke_json("progress line\n" + json.dumps(result))

    assert parsed == result


def test_parse_smoke_json_rejects_missing_json():
    script = _load_script_module()

    with pytest.raises(ValueError, match="did not print"):
        script.parse_smoke_json("not json")


def test_format_child_error_preserves_tail():
    script = _load_script_module()
    stderr = "Traceback header\n" + ("x" * 1200) + "\nRuntimeError: useful tail"

    formatted = script.format_child_error(stderr, max_chars=80)

    assert "omitted" in formatted
    assert "Traceback header" not in formatted
    assert "RuntimeError: useful tail" in formatted


def test_validate_smoke_result_accepts_voice_contract_shape():
    script = _load_script_module()

    script.validate_smoke_result(_valid_result(), max_queued_tx_ms=1_000)


def test_validate_smoke_result_rejects_silent_outbound():
    script = _load_script_module()
    result = _valid_result()
    result["outbound_webrtc_bytes"] = 0

    with pytest.raises(SystemExit, match="outbound_webrtc_bytes"):
        script.validate_smoke_result(result, max_queued_tx_ms=1_000)


def test_validate_smoke_result_rejects_excessive_queued_tx():
    script = _load_script_module()
    result = _valid_result()
    result["queued_tx_ms"] = 2_000

    with pytest.raises(SystemExit, match="queued_tx_ms"):
        script.validate_smoke_result(result, max_queued_tx_ms=1_000)


def test_validate_smoke_result_calculates_queued_tx_ms_when_missing():
    script = _load_script_module()
    result = _valid_result()
    result.pop("queued_tx_ms")
    result["queued_tx_bytes"] = 192_000

    with pytest.raises(SystemExit, match="queued_tx_ms"):
        script.validate_smoke_result(result, max_queued_tx_ms=1_000)


def test_parse_args_uses_default_expected_words(monkeypatch):
    script = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_voice_full_duplex_sidecar.py", "--voice-repo", "/voice"],
    )

    args = script.parse_args()

    assert args.voice_repo == Path("/voice")
    assert args.expect_word == ["hello", "world"]
    assert args.max_queued_tx_ms == script.DEFAULT_MAX_QUEUED_TX_MS


def test_parse_args_replaces_default_expected_words(monkeypatch):
    script = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_full_duplex_sidecar.py",
            "--voice-repo",
            "/voice",
            "--expect-word",
            "testing",
        ],
    )

    args = script.parse_args()

    assert args.expect_word == ["testing"]


def test_parse_args_rejects_negative_queue_budget(monkeypatch):
    script = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_full_duplex_sidecar.py",
            "--voice-repo",
            "/voice",
            "--max-queued-tx-ms",
            "-1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        script.parse_args()
    assert exc.value.code == 2
