"""Voice playback and real-file ownership regressions for PR #103892.

Based on the coverage contributed by crazyief. Upstream now owns artifact-path
resolution; exercise it through the running pipeline instead of mocking stat().
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.tts_tool_speaker import _SyncSentencePipeline
from tools.voice_mode import _system_player_candidates


@pytest.mark.platforms("macos")
def test_macos_players_only_offer_afplay_for_supported_containers():
    for extension in ("wav", "mp3", "m4a", "aac", "aif", "aiff", "caf"):
        path = f"clip.{extension.upper()}"
        assert _system_player_candidates(path)[0] == ["afplay", path]
    for extension in ("ogg", "opus", "flac", "xyz"):
        path = f"clip.{extension}"
        candidates = _system_player_candidates(path)
        assert candidates[0][0] == "ffplay"
        assert all(command[0] != "afplay" for command in candidates)


def test_sync_pipeline_plays_written_artifacts_and_removes_all_owned_files(tmp_path, monkeypatch):
    from tools import tts_tool_speaker, voice_mode

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    monkeypatch.setattr(tts_tool_speaker.tempfile, "tempdir", str(audio_dir))
    requested = []
    played = []

    def synthesize(*, text, output_path):
        requested.append(Path(output_path))
        if text == "failure":
            raise RuntimeError("provider failed")
        if text == "empty":
            return "not json"
        artifact = Path(output_path).with_suffix(".wav") if text == "renamed" else Path(output_path)
        artifact.write_bytes(text.encode())
        return json.dumps({"success": True, "file_path": str(artifact)})

    def play(path):
        played.append(Path(path).read_bytes())
        return True

    monkeypatch.setattr(tts_tool_speaker, "_origin", lambda: SimpleNamespace(text_to_speech_tool=synthesize))
    monkeypatch.setattr(voice_mode, "play_audio_file", play)
    pipeline = _SyncSentencePipeline(threading.Event())
    try:
        for sentence in ("renamed", "same", "empty", "failure"):
            pipeline.speak(sentence)
    finally:
        pipeline.close()

    assert played == [b"renamed", b"same"]
    assert len(requested) == 4
    assert all(not path.exists() for path in requested)
    assert not list(audio_dir.iterdir())
