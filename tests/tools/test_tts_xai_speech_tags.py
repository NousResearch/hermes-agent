"""Tests for xAI TTS speech-tag handling."""

from unittest.mock import Mock

from tools.tts_tool import _apply_xai_auto_speech_tags, _generate_xai_tts


def test_apply_xai_auto_speech_tags_adds_light_pause_after_first_sentence():
    text = "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."

    assert _apply_xai_auto_speech_tags(text) == (
        "Bonjour Monsieur Talbot. [pause] Ceci est un test de réponse vocale."
    )


def test_apply_xai_auto_speech_tags_preserves_explicit_tags():
    text = "Bonjour. [pause] <whisper>Déjà balisé.</whisper>"

    assert _apply_xai_auto_speech_tags(text) == text


def test_apply_xai_auto_speech_tags_preserves_all_documented_xai_tags():
    text = "Bonjour Monsieur Talbot. [sigh] <slow>Je parle lentement.</slow> <emphasis>Important.</emphasis>"

    assert _apply_xai_auto_speech_tags(text) == text


def test_generate_xai_tts_sends_auto_speech_tags_when_enabled(tmp_path, monkeypatch):
    captured = {}

    class FakeResponse:
        content = b"mp3"

        def raise_for_status(self):
            pass

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setattr("requests.post", fake_post)

    out = tmp_path / "out.mp3"
    _generate_xai_tts(
        "Bonjour Monsieur Talbot. Ceci est un test.",
        str(out),
        {"xai": {"voice_id": "ara", "language": "fr", "auto_speech_tags": True}},
    )

    assert out.read_bytes() == b"mp3"
    assert captured["url"] == "https://api.x.ai/v1/tts"
    assert captured["json"]["voice_id"] == "ara"
    assert captured["json"]["language"] == "fr"
    assert captured["json"]["text"] == "Bonjour Monsieur Talbot. [pause] Ceci est un test."


def test_generate_xai_tts_leaves_text_plain_by_default(tmp_path, monkeypatch):
    captured = {}

    fake_response = Mock()
    fake_response.content = b"mp3"
    fake_response.raise_for_status.return_value = None

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return fake_response

    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setattr("requests.post", fake_post)

    _generate_xai_tts(
        "Bonjour Monsieur Talbot. Ceci est un test.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "fr"}},
    )

    assert captured["json"]["text"] == "Bonjour Monsieur Talbot. Ceci est un test."
