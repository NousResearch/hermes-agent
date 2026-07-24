"""Regressions for lossless long-form TTS and final-artifact delivery limits."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.tts_tool import (
    AudioDeliveryProfile,
    _build_audio_delivery_files,
    _concat_audio_files,
    _split_text_for_tts,
    text_to_speech_tool,
)


def test_split_text_for_tts_preserves_normalized_text_under_request_cap():
    text = (
        "First sentence has useful shape. Second sentence stays intact.\n\n"
        "A final sentence closes the thought cleanly."
    )

    chunks = _split_text_for_tts(text, max_chars=48)

    assert len(chunks) > 1
    assert all(len(chunk) <= 48 for chunk in chunks)
    assert " ".join(chunks) == " ".join(text.split())


def test_voice_chunk_combine_reencodes_opus_without_stream_copy(
    tmp_path, monkeypatch,
):
    first = tmp_path / "first.ogg"
    second = tmp_path / "second.ogg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    commands = []

    monkeypatch.setattr("tools.tts_tool.shutil.which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"combined opus")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("tools.tts_tool.subprocess.run", fake_run)
    output = tmp_path / "combined.ogg"

    result = _concat_audio_files(
        [str(first), str(second)], str(output), voice_compatible=True,
    )

    assert result == str(output)
    assert output.read_bytes() == b"combined opus"
    command = commands[0]
    assert "libopus" in command
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-b:a") + 1] == "64k"
    assert command[command.index("-vbr") + 1] == "off"
    assert "copy" not in command


def test_non_voice_chunk_combine_preserves_provider_encoded_frames(
    tmp_path, monkeypatch,
):
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    commands = []

    monkeypatch.setattr("tools.tts_tool.shutil.which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"combined mp3")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("tools.tts_tool.subprocess.run", fake_run)
    output = tmp_path / "combined.mp3"

    result = _concat_audio_files([str(first), str(second)], str(output))

    assert result == str(output)
    assert output.read_bytes() == b"combined mp3"
    command = commands[0]
    assert command[command.index("-c:a") + 1] == "copy"
    assert "libopus" not in command


def test_ogg_combine_reencodes_even_without_voice_presentation_opt_in(
    tmp_path, monkeypatch,
):
    first = tmp_path / "first.ogg"
    second = tmp_path / "second.ogg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    commands = []

    monkeypatch.setattr("tools.tts_tool.shutil.which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"combined opus")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("tools.tts_tool.subprocess.run", fake_run)
    output = tmp_path / "combined.ogg"

    result = _concat_audio_files([str(first), str(second)], str(output))

    assert result == str(output)
    command = commands[0]
    assert command[command.index("-c:a") + 1] == "libopus"
    assert "copy" not in command


def test_failed_combine_preserves_ordered_separate_deliverables(
    tmp_path, monkeypatch,
):
    sources = []
    for index, payload in enumerate((b"one", b"two"), start=1):
        path = tmp_path / f"chunk{index}.mp3"
        path.write_bytes(payload)
        sources.append(str(path))

    monkeypatch.setattr("tools.tts_tool._concat_audio_files", lambda *a, **k: None)
    profile = AudioDeliveryProfile("test", max_file_bytes=1000, safety_ratio=1.0)

    paths, combined = _build_audio_delivery_files(
        sources, str(tmp_path / "reply.mp3"), profile,
    )

    assert combined is False
    assert [Path(path).read_bytes() for path in paths] == [b"one", b"two"]
    assert [Path(path).name for path in paths] == [
        "reply.part01.mp3", "reply.part02.mp3",
    ]


def test_post_encode_growth_is_repacked_into_compliant_outputs(
    tmp_path, monkeypatch,
):
    sources = []
    for index in range(3):
        path = tmp_path / f"chunk{index}.ogg"
        path.write_bytes(b"x" * 300)
        sources.append(str(path))

    def fake_combine(paths, output_path, *, voice_compatible=False):
        # The first three-way final encoding grows beyond the hard cap. Smaller
        # recursive groups remain compliant.
        size = 1200 if len(paths) == 3 else sum(Path(path).stat().st_size for path in paths)
        Path(output_path).write_bytes(b"z" * size)
        return output_path

    monkeypatch.setattr("tools.tts_tool._concat_audio_files", fake_combine)
    profile = AudioDeliveryProfile("test", max_file_bytes=1000, safety_ratio=1.0)

    paths, combined = _build_audio_delivery_files(
        sources,
        str(tmp_path / "reply.ogg"),
        profile,
        voice_compatible=True,
    )

    assert combined is True
    assert len(paths) == 2
    assert all(Path(path).stat().st_size <= profile.max_file_bytes for path in paths)


def test_single_final_encoded_chunk_over_hard_limit_fails_closed(tmp_path):
    source = tmp_path / "oversize.ogg"
    source.write_bytes(b"x" * 1001)
    profile = AudioDeliveryProfile("test", max_file_bytes=1000, safety_ratio=1.0)

    with pytest.raises(ValueError, match="Final-encoded TTS chunk exceeds"):
        _build_audio_delivery_files(
            [str(source)], str(tmp_path / "reply.ogg"), profile,
        )


def test_text_to_speech_tool_chunks_all_input_in_order(tmp_path, monkeypatch):
    calls = []

    def fake_openai(text, output_path, config):
        calls.append(text)
        Path(output_path).write_bytes(("audio:" + text).encode())
        return output_path

    def fake_combine(paths, output_path, *, voice_compatible=False):
        Path(output_path).write_bytes(b"".join(Path(path).read_bytes() for path in paths))
        return output_path

    monkeypatch.setattr("tools.tts_tool._import_openai_client", lambda: object)
    monkeypatch.setattr("tools.tts_tool._generate_openai_tts", fake_openai)
    monkeypatch.setattr("tools.tts_tool._concat_audio_files", fake_combine)
    monkeypatch.setattr(
        "tools.tts_tool._load_tts_config",
        lambda: {
            "provider": "openai",
            "openai": {"max_text_length": 60},
            "delivery_profiles": {
                "default": {"max_file_bytes": 1_000_000, "safety_ratio": 1.0},
            },
        },
    )

    text = " ".join(
        f"Sentence {index} contains enough words for natural chunking."
        for index in range(12)
    )
    result = json.loads(
        text_to_speech_tool(text=text, output_path=str(tmp_path / "reply.mp3"))
    )

    assert result["success"] is True
    assert result["chunk_count"] == len(calls) > 1
    assert all(len(call) <= 60 for call in calls)
    assert " ".join(calls) == " ".join(text.split())
    assert result["delivery_file_count"] == 1
    assert result["combined_chunks"] is True
    assert os.path.isfile(result["file_path"])


def test_long_form_generation_uses_one_config_snapshot(tmp_path, monkeypatch):
    calls = []
    config_loads = 0

    def load_config_once():
        nonlocal config_loads
        config_loads += 1
        if config_loads > 1:
            raise AssertionError("TTS config was reloaded between chunks")
        return {
            "provider": "openai",
            "openai": {"max_text_length": 40},
            "delivery_profiles": {
                "default": {
                    "max_file_bytes": 1_000_000,
                    "safety_ratio": 1.0,
                },
            },
        }

    def fake_openai(text, output_path, config):
        calls.append(text)
        Path(output_path).write_bytes(("audio:" + text).encode())
        return output_path

    def fake_combine(paths, output_path, *, voice_compatible=False):
        Path(output_path).write_bytes(
            b"".join(Path(path).read_bytes() for path in paths)
        )
        return output_path

    monkeypatch.setattr("tools.tts_tool._load_tts_config", load_config_once)
    monkeypatch.setattr("tools.tts_tool._import_openai_client", lambda: object)
    monkeypatch.setattr("tools.tts_tool._generate_openai_tts", fake_openai)
    monkeypatch.setattr("tools.tts_tool._concat_audio_files", fake_combine)

    text = "One stable provider must render every long-form chunk without loss. " * 4
    result = json.loads(
        text_to_speech_tool(text=text, output_path=str(tmp_path / "reply.mp3"))
    )

    assert result["success"] is True
    assert config_loads == 1
    assert len(calls) > 1
    assert " ".join(calls) == " ".join(text.split())
