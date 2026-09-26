"""Real-ffmpeg regressions: low amplitude is not evidence of silence.

These generated harmonic envelopes are speech-like signals, not human speech
or a VAD benchmark. No model, credentials, downloads, or private audio needed.
"""

import math
import random
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from tools.transcription_audio import _probe_audio_duration, _trim_silence_for_cloud_stt


@pytest.fixture(params=[1, 2], ids=["mono", "right-channel-only"])
def signal_wav(tmp_path, request):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not installed")

    def write(segments):
        rng = random.Random(42)
        samples = []
        for kind, seconds, gain in segments:
            for i in range(round(seconds * 16000)):
                t = i / 16000
                if kind == "noise":
                    value = rng.uniform(-1, 1)
                elif kind == "voice":
                    envelope = 0.25 + 0.75 * math.sin(math.pi * 3 * t) ** 2
                    value = envelope * (
                        0.6 * math.sin(2 * math.pi * 173 * t)
                        + 0.3 * math.sin(2 * math.pi * 346 * t)
                        + 0.1 * math.sin(2 * math.pi * 519 * t)
                    )
                else:
                    value = 0
                sample = round(32767 * gain * value)
                samples.extend([sample] if request.param == 1 else [0, sample])
        path = tmp_path / "signal.wav"
        with wave.open(str(path), "wb") as output:
            output.setparams((request.param, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return str(path)

    return write


@pytest.mark.parametrize("gain", [0.5, 0.01, 0.001, 0.0001])
@pytest.mark.parametrize("background", ["silence", "noise"])
@pytest.mark.parametrize("opening", [0, 0.05, 2])
def test_quiet_content_survives_after_loud_content(
    signal_wav, gain, background, opening
):
    # A loud opening defeats global peak normalization. Quiet content before
    # AND after a long pause must survive, even when most of the file is quiet.
    wav = signal_wav([
        ("voice", opening, 0.5),
        ("voice", 5, gain),
        (background, 6, 0.0001),
        ("voice", 5, gain),
        ("silence", 4, 0),
    ])
    original = Path(wav).read_bytes()
    trimmed = _trim_silence_for_cloud_stt(wav, {})
    try:
        selected = trimmed or wav
        duration = _probe_audio_duration(selected)
        # Every non-silent segment survives, not just a non-empty loud prefix.
        expected_content = opening + (10 if background == "silence" else 16)
        assert duration is not None and duration >= expected_content - 0.05
        assert trimmed is not None  # digital trailing silence still saves >10%
        original_duration = _probe_audio_duration(wav)
        assert original_duration is not None and duration < original_duration * 0.9
        assert Path(wav).read_bytes() == original
        if trimmed:
            # Decode the returned AAC, not merely its container duration. Both
            # quiet spans must still contain energy at the expected locations.
            ffmpeg = shutil.which("ffmpeg")
            assert ffmpeg is not None
            raw = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    selected,
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
            values = struct.unpack(f"<{len(raw) // 2}h", raw)
            for second in (4, int(expected_content) - 1):
                block = values[second * 16000 : (second + 1) * 16000]
                assert len(block) == 16000
                assert sum(x * x for x in block) > 0
    finally:
        if trimmed:
            shutil.rmtree(Path(trimmed).parent)


@pytest.mark.parametrize(
    "kind,gain", [("voice", 0.001), ("noise", 0.001), ("silence", 0)]
)
def test_low_level_dense_audio_and_normal_pauses_fail_open(signal_wav, kind, gain):
    wav = signal_wav([
        (kind, 6, gain),
        ("silence", 0.2, 0),
        (kind, 6, gain),
        ("silence", 0.2, 0),
        (kind, 2, gain),
    ])
    assert _trim_silence_for_cloud_stt(wav, {}) is None
