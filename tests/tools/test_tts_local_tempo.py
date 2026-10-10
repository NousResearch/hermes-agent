"""Local tempo stays opt-in through real config loading and preserves audio pitch."""

import json
import math
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_platform.resolver import locate_command
from tools import tts_tool
from tools.tts_tool_delivery import _apply_local_tempo


@pytest.mark.parametrize("mode", [None, "forward", "local"])
def test_config_selects_local_tempo_without_changing_default(tmp_path, monkeypatch, mode):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    setting = f"    speed_mode: {mode}\n" if mode else ""
    (home / "config.yaml").write_text(
        "tts:\n  provider: openai\n  speed: 1.5\n  openai:\n"
        "    model: gpt-4o-mini-tts\n" + setting, encoding="utf-8")
    output = tmp_path / "speech.mp3"
    client = MagicMock()
    client.audio.speech.create.return_value.stream_to_file.side_effect = (
        lambda path: Path(path).write_bytes(b"ID3" + b"\0" * 64))

    with patch.object(tts_tool, "_import_openai_client", return_value=MagicMock(return_value=client)), \
         patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
               return_value=("synthetic-test-key", None, False)), \
         patch.object(tts_tool, "_apply_local_tempo", return_value=str(output)) as tempo:
        result = json.loads(tts_tool.text_to_speech_tool("Synthetic test", output_path=str(output)))

    assert result["success"], result
    request = client.audio.speech.create.call_args.kwargs
    if mode == "local":
        assert "speed" not in request
        tempo.assert_called_once_with(str(output), 1.5)
    else:
        assert request["speed"] == 1.5
        tempo.assert_not_called()


def test_normal_speed_needs_no_ffmpeg_and_leaves_file_unchanged(tmp_path):
    output = tmp_path / "speech.wav"
    output.write_bytes(b"original")
    with patch("tools.tts_tool_delivery.locate_command") as lookup:
        assert _apply_local_tempo(str(output), 1.0) == str(output)
    lookup.assert_not_called()
    assert output.read_bytes() == b"original"


@pytest.mark.parametrize("speed", [0.25, 1.5, 4.0])
def test_real_ffmpeg_changes_duration_without_shifting_pitch(tmp_path, speed):
    if not locate_command("ffmpeg").command:
        pytest.skip("ffmpeg not installed")
    sample_rate, seconds, frequency = 24000, 2, 440
    output = tmp_path / "tone.wav"
    samples = [int(12000 * math.sin(2 * math.pi * frequency * i / sample_rate))
               for i in range(sample_rate * seconds)]
    with wave.open(str(output), "wb") as audio:
        audio.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    assert _apply_local_tempo(str(output), speed) == str(output)
    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == sample_rate
        frames = audio.getnframes()
        processed = struct.unpack(f"<{frames}h", audio.readframes(frames))
    assert frames / sample_rate == pytest.approx(seconds / speed, abs=0.12)
    middle = processed[frames // 4:3 * frames // 4]
    crossings = sum(a <= 0 < b for a, b in zip(middle, middle[1:]))
    assert crossings * sample_rate / len(middle) == pytest.approx(frequency, abs=10)
    assert not list(tmp_path.glob(".*.atempo.wav"))
