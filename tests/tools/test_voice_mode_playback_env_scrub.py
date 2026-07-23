"""Voice-mode system playback must scrub credential env (sibling of #70342)."""

from unittest.mock import MagicMock, patch


def test_play_audio_file_scrubbed_env(tmp_path, monkeypatch):
    audio = tmp_path / "t.mp3"
    audio.write_bytes(b"ID3fake")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        proc = MagicMock()
        proc.wait.return_value = 0
        return proc

    import tools.voice_mode as vm

    with patch.object(vm, "platform") as plat, patch.object(
        vm.shutil, "which", return_value="/usr/bin/ffplay"
    ), patch.object(vm.subprocess, "Popen", side_effect=fake_popen):
        plat.system.return_value = "Linux"
        ok = vm.play_audio_file(str(audio))

    assert ok is True
    assert captured.get("env") is not None
    assert "TELEGRAM_BOT_TOKEN" not in captured["env"]
    assert "OPENAI_API_KEY" not in captured["env"]
