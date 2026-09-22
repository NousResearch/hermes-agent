"""Cloud CAF conversion owns its output without modifying neighboring files."""

from pathlib import Path
import subprocess

import pytest

from tools import transcription_audio as audio
from tools import transcription_tools as stt


@pytest.mark.parametrize("outcome", ["success", "provider-error", "conversion-error"])
def test_caf_conversion_preserves_source_neighbors_and_removes_owned_output(tmp_path, monkeypatch, outcome):
    source = tmp_path / "voice.caf"
    source.write_bytes(b"caff fixture")
    neighbor = source.with_suffix(".wav")
    neighbor.write_bytes(b"existing recording")
    outputs = []
    monkeypatch.setattr(stt, "_load_stt_config", lambda: {
        "provider": "groq", "cloud_trim_silence": False,
    })
    monkeypatch.setattr(audio, "_find_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)

    def encode(command, **_kwargs):
        output = Path(command[-1])
        outputs.append(output)
        output.write_bytes(b"converted recording")
        if outcome == "conversion-error":
            raise subprocess.CalledProcessError(1, command)

    def transcribe(file_path, *_args):
        assert Path(file_path).read_bytes() == b"converted recording"
        if outcome == "provider-error":
            raise RuntimeError("transcription failed")
        return {"success": True, "transcript": "hello"}

    monkeypatch.setattr(audio, "_run_quiet", encode)
    monkeypatch.setattr(stt, "_dispatch_stt_provider", transcribe)
    if outcome == "provider-error":
        with pytest.raises(RuntimeError, match="transcription failed"):
            stt.transcribe_audio(str(source))
    else:
        result = stt.transcribe_audio(str(source))
        assert result["success"] is (outcome == "success")
    assert source.read_bytes() == b"caff fixture"
    assert neighbor.read_bytes() == b"existing recording"
    assert outputs and all(not path.exists() and not path.parent.exists() for path in outputs)
