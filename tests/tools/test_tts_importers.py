"""SDK import fallback uses the real import system and exact PM feature mapping."""
import importlib
import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

import pm
from tools import tts_tool


@pytest.mark.parametrize("importer,module,attribute,feature", [
    (tts_tool._import_edge_tts, "edge_tts", None, "edge-tts"),
    (tts_tool._import_elevenlabs, "elevenlabs.client", "ElevenLabs", "tts-premium"),
    (tts_tool._import_mistral_client, "mistralai.client", "Mistral", "mistral"),
])
@pytest.mark.parametrize("missing,install_fails", [(False, False), (False, True), (True, True)])
def test_sdk_import_fallback(tmp_path, monkeypatch, importer, module, attribute, feature, missing, install_fails):
    parts = module.split(".")
    for name in list(sys.modules):
        if name == parts[0] or name.startswith(parts[0] + "."):
            monkeypatch.delitem(sys.modules, name)
    if missing:
        monkeypatch.setitem(sys.modules, parts[0], None)
    else:
        package = tmp_path / parts[0]
        package.mkdir()
        (package / "__init__.py").write_text("marker = 'installed outside venv'\n", encoding="utf-8")
        if attribute:
            (package / "client.py").write_text(f"class {attribute}: pass\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
    ensure = MagicMock(side_effect=pm.InstallError(feature, "fixture refusal") if install_fails else None)
    monkeypatch.setattr(pm, "ensure_import", ensure)
    with pytest.raises(ImportError) if missing else nullcontext():
        result = importer()
        loaded = importlib.import_module(module)
        assert result is (getattr(loaded, attribute) if attribute else loaded)
        assert str(tmp_path) in loaded.__file__
    ensure.assert_called_once_with(feature)


def test_mistral_stt_fallback_reads_audio_and_requires_key(tmp_path, monkeypatch):
    from tools import transcription_tools

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fixture-audio")
    calls = []

    def complete(**kwargs):
        calls.append((kwargs["model"], kwargs["file"]["content"].read()))
        return {"text": "hello"}

    client = MagicMock()
    client.return_value.__enter__.return_value.audio.transcriptions.complete.side_effect = complete
    module = type(sys)("mistralai.client")
    module.Mistral = client
    monkeypatch.setitem(sys.modules, "mistralai.client", module)
    ensure = MagicMock(side_effect=pm.InstallError("mistral", "fixture refusal"))
    monkeypatch.setattr(pm, "ensure_import", ensure)
    monkeypatch.setattr(transcription_tools, "_resolve_provider_key", lambda *a: "test-key")
    result = transcription_tools._transcribe_mistral(str(audio), "fixture-model")
    assert result["success"] and result["transcript"] == "hello"
    assert calls == [("fixture-model", b"fixture-audio")]
    client.assert_called_once_with(api_key="test-key")
    ensure.assert_called_once_with("mistral")
    monkeypatch.setattr(transcription_tools, "_resolve_provider_key", lambda *a: "")
    assert "MISTRAL_API_KEY not set" in transcription_tools._transcribe_mistral(str(audio), "fixture-model")["error"]
    assert len(calls) == 1
