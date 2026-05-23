from __future__ import annotations

import io
import struct
import wave

from tools import voicevox_tts_tool as voicevox


def _sample_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<160h", *([0] * 160)))
    return buf.getvalue()


def test_resolve_output_device_by_name():
    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return [
                {"name": "Microphone", "max_output_channels": 0},
                {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2},
            ]

    index, name = voicevox._resolve_output_device(FakeSoundDevice, "vb-audio")

    assert index == 1
    assert name == "CABLE Input (VB-Audio Virtual Cable)"


def test_play_wav_to_named_output_device(monkeypatch):
    calls = []

    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return [
                {"name": "Default Speakers", "max_output_channels": 2},
                {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2},
            ]

        @staticmethod
        def play(data, samplerate, *, device=None, blocking=False):
            calls.append(
                {
                    "frames": len(data),
                    "samplerate": samplerate,
                    "device": device,
                    "blocking": blocking,
                }
            )

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSoundDevice)

    result = voicevox._play_wav(_sample_wav(), output_device="CABLE Input", blocking=True)

    assert result["success"] is True
    assert result["output_device_index"] == 1
    assert result["output_device"] == "CABLE Input (VB-Audio Virtual Cable)"
    assert calls == [{"frames": 160, "samplerate": 16000, "device": 1, "blocking": True}]
