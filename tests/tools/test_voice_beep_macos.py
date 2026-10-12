import wave
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.platforms("macos")
def test_beep_uses_temp_wav_without_sounddevice(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    from tools import voice_mode as vm

    monkeypatch.setattr(vm.tempfile, "tempdir", str(tmp_path))
    captured = []

    def capture_playback(path):
        with wave.open(path, "rb") as wav:
            assert wav.getframerate() == vm.SAMPLE_RATE
            assert wav.getsampwidth() == 2
            assert wav.getnchannels() == 1
            assert wav.getnframes() > 0
        captured.append(Path(path))
        return True

    with patch.object(vm, "_import_audio") as import_audio, \
         patch.object(vm, "play_audio_file", side_effect=capture_playback):
        vm.play_beep(frequency=880, duration=0.1, count=1)

    import_audio.assert_not_called()
    assert len(captured) == 1
    assert not captured[0].exists()
