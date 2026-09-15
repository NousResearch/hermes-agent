"""Configured STT failover is opt-in and never bypasses input guards."""
from unittest.mock import Mock

import pytest

from tools import transcription_tools as stt


@pytest.fixture
def audio(tmp_path, monkeypatch):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"RIFF" + b"\0" * 40)
    monkeypatch.setattr(stt, "_prepare_audio_for_transcription", lambda p: (p, None, None))
    monkeypatch.setattr(stt, "_get_provider", lambda cfg: cfg.get("provider", "primary"))
    monkeypatch.setattr(stt, "_trim_silence_for_cloud_stt", lambda *a: None)
    return str(path)


def configure(monkeypatch, **cfg):
    monkeypatch.setattr(stt, "_load_stt_config", lambda: {"provider": "primary", **cfg})


@pytest.mark.parametrize("failure", [{"success": False, "error": "unavailable"}, RuntimeError("backend failed")])
def test_failure_tries_fallback_once(audio, monkeypatch, failure):
    configure(monkeypatch, fallback_providers=["primary", "backup", "backup", "last"])
    dispatch = Mock(side_effect=[failure, {"success": True, "transcript": "hello", "provider": "backup"}])
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    assert stt.transcribe_audio(audio)["provider"] == "backup"
    assert [c.args[1] for c in dispatch.call_args_list] == ["primary", "backup"]


def test_exhaustion_returns_last_error(audio, monkeypatch):
    configure(monkeypatch, fallback_providers=["backup", "last", "backup"])
    dispatch = Mock(return_value={"success": False, "error": "failed"})
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    assert stt.transcribe_audio(audio)["error"] == "failed"
    assert [c.args[1] for c in dispatch.call_args_list] == ["primary", "backup", "last"]


@pytest.mark.parametrize("fallbacks", [None, [], "backup", [None, 3, "", " "]])
def test_no_implicit_fallback(audio, monkeypatch, fallbacks):
    configure(monkeypatch, fallback_providers=fallbacks)
    dispatch = Mock(return_value={"success": False, "error": "failed"})
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    assert not stt.transcribe_audio(audio)["success"]
    dispatch.assert_called_once()


def test_success_does_not_try_fallback(audio, monkeypatch):
    configure(monkeypatch, fallback_providers=["backup"])
    dispatch = Mock(return_value={"success": True, "transcript": "ok"})
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    assert stt.transcribe_audio(audio)["success"]
    dispatch.assert_called_once()


@pytest.mark.parametrize("guard", ["disabled", "missing", "unsupported", "blocked", "oversized"])
def test_input_guards_do_not_dispatch(audio, monkeypatch, guard):
    configure(monkeypatch, fallback_providers=["backup"], enabled=guard != "disabled")
    if guard == "missing":
        audio += ".missing.wav"
    elif guard == "unsupported":
        from pathlib import Path
        p = Path(audio).with_suffix(".txt")
        p.write_text("not audio")
        audio = str(p)
    elif guard == "blocked":
        monkeypatch.setattr(stt, "_read_block_error", lambda p: {"success": False, "error": "blocked"})
    elif guard == "oversized":
        monkeypatch.setattr(stt, "_validate_audio_file_size", lambda p: {"success": False, "error": "too large"})
    dispatch = Mock()
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    assert not stt.transcribe_audio(audio)["success"]
    dispatch.assert_not_called()


def test_model_override_is_not_forwarded_to_other_providers(audio, monkeypatch):
    configure(monkeypatch, fallback_providers=["backup"])
    dispatch = Mock(side_effect=[{"success": False}, {"success": True, "transcript": "ok"}])
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    assert stt.transcribe_audio(audio, model="primary-model", source="gateway")["success"]
    assert [(c.args[3], c.args[4]) for c in dispatch.call_args_list] == [("primary-model", "gateway"), (None, "gateway")]


def test_plugin_failure_reaches_builtin(audio, monkeypatch):
    configure(monkeypatch, fallback_providers=["local"])
    monkeypatch.setattr(stt, "_apply_pre_transcription_hook", lambda **kw: (kw["model"], kw["language"], kw["prompt"]))
    plugin = Mock(return_value={"success": False, "error": "plugin unavailable"})
    local = Mock(return_value={"success": True, "transcript": "local result"})
    monkeypatch.setattr(stt, "_dispatch_to_plugin_provider", plugin)
    monkeypatch.setattr(stt, "_transcribe_local", local)
    assert stt.transcribe_audio(audio)["transcript"] == "local result"
    plugin.assert_called_once()
    local.assert_called_once()


def test_local_failure_cannot_bypass_cloud_upload_limit(audio, monkeypatch):
    configure(monkeypatch, provider="local", fallback_providers=["openai", "backup"])
    dispatch = Mock(return_value={"success": False, "error": "local failed"})
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    size_check = Mock(return_value={"success": False, "error": "too large"})
    monkeypatch.setattr(stt, "_validate_audio_file_size", size_check)
    assert stt.transcribe_audio(audio)["error"] == "too large"
    assert [c.args[1] for c in dispatch.call_args_list] == ["local"]
    size_check.assert_called_once()


def test_trim_is_cleaned_and_not_reused_by_fallback(audio, monkeypatch, tmp_path):
    configure(monkeypatch, provider="openai", fallback_providers=["local"])
    trim_dir = tmp_path / "trim"
    trim_dir.mkdir()
    trimmed = trim_dir / "trim.wav"
    trimmed.write_bytes(b"RIFF")
    monkeypatch.setattr(stt, "_trim_silence_for_cloud_stt", lambda *a: str(trimmed))
    dispatch = Mock(side_effect=[RuntimeError("failed"), {"success": True, "transcript": "ok"}])
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    assert stt.transcribe_audio(audio)["success"]
    assert [c.args[0] for c in dispatch.call_args_list] == [str(trimmed), audio]
    assert not trim_dir.exists()


def test_all_exceptions_return_failure_envelope(audio, monkeypatch):
    configure(monkeypatch, fallback_providers=["backup"])
    dispatch = Mock(side_effect=RuntimeError("unavailable"))
    monkeypatch.setattr(stt, "_dispatch_stt_provider", dispatch)
    result = stt.transcribe_audio(audio)
    assert result["success"] is False
    assert result["provider"] == "backup"
    assert "unavailable" in result["error"]
    assert dispatch.call_count == 2
