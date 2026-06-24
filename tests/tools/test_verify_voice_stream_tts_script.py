import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_voice_stream_tts.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_voice_stream_tts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_default_command_template_streams_raw_pcm_stdout():
    script = _load_script_module()

    template = script.build_default_command_template("/tmp/bin/voice with spaces")

    assert template.startswith("'/tmp/bin/voice with spaces' stream --quiet")
    assert "--sample-rate {sample_rate}" in template
    assert "--frame-ms {frame_ms}" in template
    assert "--raw-output -" in template
    assert "--input-file {input_path}" in template
    assert "--voice {voice}" in template
    assert "--speed {speed}" in template


def test_audio_contract_matches_voice_sidecar_pcm_shape():
    script = _load_script_module()

    contract = script.AudioContract()

    assert contract.samples_per_frame == 960
    assert contract.frame_bytes == 1920
    assert contract.bytes_per_second == 96_000
    assert contract.as_dict() == {
        "sample_rate": 48_000,
        "channels": 1,
        "frame_ms": 20,
        "encoding": "pcm_s16le",
        "bytes_per_sample": 2,
        "samples_per_frame": 960,
        "frame_bytes": 1920,
    }


def stream_contract_json(**surface_overrides):
    raw_outbound = {
        "command": 'voice stream --sample-rate 48000 --frame-ms 20 --raw-output - "hello"',
        "output": "pcm_s16le",
        "transport": "stdout_pcm_frames",
        "frame_bytes": 1_920,
    }
    raw_outbound.update(surface_overrides)
    return {
        "audio": {
            "sample_rate": 48_000,
            "channels": 1,
            "frame_ms": 20,
            "encoding": "pcm_s16le",
            "bytes_per_sample": 2,
            "frame_bytes": 1_920,
        },
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
            "raw_outbound_pcm": raw_outbound,
            "raw_inbound_pcm": {
                "command": "voice stream-transcribe --raw-input - --sample-rate 48000 --frame-ms 20",
                "input": "pcm_s16le",
                "transport": "stdin_pcm_frames",
                "frame_bytes": 1_920,
            },
        },
    }


def test_audio_contract_from_voice_reads_stream_contract(monkeypatch):
    script = _load_script_module()

    def fake_load(_voice_bin):
        return stream_contract_json()

    monkeypatch.setattr(script, "load_voice_stream_contract", fake_load)

    contract = script.audio_contract_from_voice(
        "/tmp/voice",
        fallback=script.AudioContract(sample_rate=16_000),
    )

    assert contract.sample_rate == 48_000
    assert contract.frame_bytes == 1_920
    assert "voice stream" in contract.raw_outbound_pcm_command
    assert "--raw-input" in contract.raw_inbound_pcm_command
    assert "--format ogg-opus" in contract.completed_voice_note_command
    assert "voice stream" in contract.streamed_voice_note_command


def test_audio_contract_from_voice_rejects_surface_frame_drift(monkeypatch):
    script = _load_script_module()

    def fake_load(_voice_bin):
        return stream_contract_json(frame_bytes=960)

    monkeypatch.setattr(script, "load_voice_stream_contract", fake_load)

    with pytest.raises(SystemExit, match="raw_outbound_pcm frame_bytes"):
        script.audio_contract_from_voice("/tmp/voice", fallback=script.AudioContract())


def test_validate_pcm_accepts_non_silent_whole_frames():
    script = _load_script_module()
    contract = script.AudioContract()
    frame = (12_000).to_bytes(2, byteorder="little", signed=True) * 960

    stats = script.validate_pcm(frame * 10, contract=contract)

    assert stats == {
        "bytes": 19_200,
        "frames": 10,
        "duration_ms": 200,
        "peak": 12_000,
    }


def test_validate_pcm_rejects_silence():
    script = _load_script_module()
    contract = script.AudioContract()

    with pytest.raises(SystemExit, match="PCM peak 0 is below minimum"):
        script.validate_pcm(b"\x00" * contract.frame_bytes, contract=contract)


def test_validate_pcm_rejects_partial_sample():
    script = _load_script_module()
    contract = script.AudioContract()

    with pytest.raises(SystemExit, match="whole s16le samples"):
        script.validate_pcm(b"\x00", contract=contract)


def test_validate_pcm_rejects_partial_frame():
    script = _load_script_module()
    contract = script.AudioContract()

    with pytest.raises(SystemExit, match="not aligned to 1920-byte frames"):
        script.validate_pcm(b"\x01\x00" * 100, contract=contract)


def test_build_streamed_ogg_command_uses_daemon_stream_output(tmp_path: Path):
    script = _load_script_module()
    contract = script.AudioContract()

    command = script.build_streamed_ogg_command(
        voice_bin="/tmp/voice",
        input_path=tmp_path / "input.txt",
        output_path=tmp_path / "reply.ogg",
        contract=contract,
        voice="af_heart",
        speed="1.0",
    )

    assert command[:3] == ["/tmp/voice", "stream", "--quiet"]
    assert "--output" in command
    assert str(tmp_path / "reply.ogg") in command
    assert "--format" in command
    assert "ogg-opus" in command
    assert "--sample-rate" in command
    assert "48000" in command


def test_validate_streamed_ogg_requires_real_opus_file(tmp_path: Path):
    script = _load_script_module()
    output = tmp_path / "reply.ogg"
    output.write_bytes(b"OggS")

    stats = script.validate_streamed_ogg(
        output,
        probe={"codec_name": "opus", "sample_rate": "48000", "channels": "1"},
    )

    assert stats["bytes"] == 4
    assert stats["codec_name"] == "opus"

    with pytest.raises(SystemExit, match="codec_name=opus"):
        script.validate_streamed_ogg(
            output,
            probe={"codec_name": "vorbis", "sample_rate": "48000", "channels": "1"},
        )
