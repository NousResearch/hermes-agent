import tools.transcription_tools as tt


def test_voice_stt_provider_alias_overrides_stt_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "stt": {"enabled": True, "provider": "local"},
            "voice": {"stt_provider": "elevenlabs_scribe_realtime"},
        },
    )
    cfg = tt._load_stt_config()
    assert cfg["provider"] == "elevenlabs_scribe_realtime"


def test_realtime_scribe_marks_committed_only(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFFxxxxWAVE")
    monkeypatch.setattr(tt, "_validate_audio_file", lambda path: None)
    monkeypatch.setattr(tt, "_load_stt_config", lambda: {"enabled": True, "provider": "elevenlabs_scribe_realtime", "elevenlabs": {"model_id": "scribe_v2"}})
    monkeypatch.setattr(tt, "_get_provider", lambda cfg: "elevenlabs_scribe_realtime")
    monkeypatch.setattr(tt, "_transcribe_elevenlabs", lambda path, model: {"success": True, "transcript": "hello", "provider": "elevenlabs"})

    result = tt.transcribe_audio(str(audio))

    assert result["success"] is True
    assert result["provider"] == "elevenlabs_scribe_realtime"
    assert result["partial_transcripts_executed"] is False
    assert result["committed_transcript_only"] is True
