"""OS media helpers must not inherit Hermes credentials.

Established by the TTS/STT command scrub (#56332 / #70342) and extended to
voice-mode playback: ``ffplay``/``afplay``/``aplay`` are spawned with
``hermes_subprocess_env(inherit_credentials=False)`` so provider API keys and
gateway tokens never reach an OS media helper.

``ffmpeg`` / ``ffprobe`` are the same kind of process — third-party binaries
Hermes shells out to for transcoding and duration probes — and they run on the
same voice/attachment paths.
"""

from unittest.mock import MagicMock, patch


_SECRETS = {
    "OPENAI_API_KEY": "sk-test",
    "TELEGRAM_BOT_TOKEN": "secret-token",
    "ANTHROPIC_API_KEY": "sk-ant-test",
}


def _seed_secrets(monkeypatch):
    for key, value in _SECRETS.items():
        monkeypatch.setenv(key, value)


def _assert_scrubbed(env):
    assert env is not None, "media helper inherited the full process environment"
    for key in _SECRETS:
        assert key not in env, f"{key} leaked into the media helper env"


def test_tts_ffmpeg_transcode_scrubs_credentials(tmp_path, monkeypatch):
    """The voice-note OGG transcode on the TTS path."""
    _seed_secrets(monkeypatch)
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFFfake")
    out = tmp_path / "out.ogg"

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        out.write_bytes(b"OggSfake")
        result = MagicMock()
        result.returncode = 0
        result.stderr = b""
        return result

    import tools.tts_tool as tts

    with patch.object(tts, "_has_ffmpeg", return_value=True), patch.object(
        tts.subprocess, "run", side_effect=fake_run
    ):
        tts._ffmpeg_transcode_to_opus(str(src), str(out))

    _assert_scrubbed(captured.get("env"))


def test_telegram_ffprobe_duration_scrubs_credentials(tmp_path, monkeypatch):
    """Inbound voice-note duration probe."""
    _seed_secrets(monkeypatch)
    media = tmp_path / "voice.ogg"
    media.write_bytes(b"OggSfake")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "3.5"
        return result

    from plugins.platforms.telegram import adapter as tg

    with patch("shutil.which", return_value="/usr/bin/ffprobe"), patch(
        "subprocess.run", side_effect=fake_run
    ):
        tg._probe_voice_duration_seconds(str(media))

    _assert_scrubbed(captured.get("env"))


def test_discord_ffprobe_duration_scrubs_credentials(tmp_path, monkeypatch):
    """Discord's audio duration probe — same helper, same exposure."""
    _seed_secrets(monkeypatch)
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"ID3fake")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "2.0"
        return result

    from plugins.platforms.discord import adapter as dc

    adapter = dc.DiscordAdapter.__new__(dc.DiscordAdapter)
    with patch.object(dc.subprocess, "run", side_effect=fake_run):
        adapter._probe_audio_duration_seconds(str(media))

    _assert_scrubbed(captured.get("env"))
