"""Tests for xAI TTS speech-tag handling."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tools.tts_tool import (
    _XAI_INLINE_SPEECH_TAGS,
    _XAI_WRAPPING_SPEECH_TAGS,
    _apply_xai_auto_speech_tags,
    _generate_xai_tts,
)


def test_apply_xai_auto_speech_tags_adds_light_pause_after_first_sentence():
    text = "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."

    assert _apply_xai_auto_speech_tags(text) == (
        "Bonjour Monsieur Talbot. [pause] Ceci est un test de réponse vocale."
    )


def test_apply_xai_auto_speech_tags_preserves_explicit_tags():
    text = "Bonjour. [pause] <whisper>Déjà balisé.</whisper>"

    assert _apply_xai_auto_speech_tags(text) == text


def test_apply_xai_auto_speech_tags_multi_paragraph_emits_single_pause():
    """Regression for #29417 — multi-paragraph input doubled the pause.

    Pre-fix the paragraph substitution injected ``[pause]`` between
    paragraphs, then the unconditional first-sentence substitution
    added another one right after, producing ``[pause] [pause]`` in
    the audio.  The fix re-checks the tag-detection guard after the
    paragraph pass.

    Requires a first sentence of 12+ chars to hit the
    ``_XAI_FIRST_SENTENCE_RE`` length floor — the trivial
    ``"Hello.\\n\\nWorld."`` case dodged the bug by accident.
    """
    text = "Welcome to the demo of our new product line.\n\nIt has many features."
    result = _apply_xai_auto_speech_tags(text)

    # Exactly one [pause] between the paragraphs, not two.
    assert result.count("[pause]") == 1, (
        f"expected single [pause], got {result.count('[pause]')} in {result!r}"
    )
    assert result == (
        "Welcome to the demo of our new product line. [pause] It has many features."
    )


def test_apply_xai_auto_speech_tags_single_paragraph_still_gets_first_sentence_pause():
    """Sanity guard — the fix only suppresses the first-sentence pass when
    a paragraph pass already injected ``[pause]``.  Single-paragraph input
    must still get its first-sentence pause.
    """
    text = "Welcome to the demo of our new product line. It has many features."
    assert _apply_xai_auto_speech_tags(text) == (
        "Welcome to the demo of our new product line. [pause] It has many features."
    )


def test_generate_xai_tts_sends_auxiliary_rewriter_output_to_api(
    tmp_path, monkeypatch
):
    """auto_speech_tags=True should send the auxiliary rewriter's tagged
    output (not the conservative local pause fallback) to the xAI TTS API.

    The previous version of this test asserted on the local pause-tagged
    text — which only happened to match because ``call_llm`` returns
    ``None`` in the test environment and the function silently fell
    back. With the new auxiliary-rewrite path the user-visible contract
    is "what the LLM said wins", so this test pins that down.
    """
    captured = {}
    rewriter_output = "Bonjour Monsieur Talbot. [warmly] Ceci est un test. [soft laugh]"

    class FakeResponse:
        content = b"mp3"

        def raise_for_status(self):
            pass

    def fake_post(url, headers, json, timeout, stream=False):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        captured["stream"] = stream
        return FakeResponse()

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=rewriter_output))]
    )

    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm", lambda *a, **kw: fake_response
    )

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
    assert captured["json"]["text"] == rewriter_output


def test_generate_xai_tts_uses_oauth_pinned_base_url(tmp_path, monkeypatch):
    """OAuth bearer tokens must not follow user/env base URL overrides."""
    captured = {}

    class FakeResponse:
        content = b"mp3"

        def raise_for_status(self):
            pass

    def fake_post(url, headers, json, timeout, stream=False):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setenv("XAI_BASE_URL", "https://attacker.example/v1")
    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        lambda: {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        },
    )
    monkeypatch.setattr("requests.post", fake_post)

    out = tmp_path / "out.mp3"
    _generate_xai_tts(
        "Bonjour.",
        str(out),
        {
            "xai": {
                "base_url": "https://attacker.example/config",
                "auto_speech_tags": False,
            }
        },
    )

    assert captured["url"] == "https://api.x.ai/v1/tts"
    assert captured["headers"]["Authorization"] == "Bearer oauth-bearer-token"


def test_auto_speech_tags_calls_auxiliary_rewriter_with_tts_audio_tags_task():
    """When input has no explicit speech tags, the function must call the
    auxiliary rewriter with task='tts_audio_tags' and a system prompt
    that documents the xAI inline + wrapping tag vocabulary.
    """
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[warmly] Hi."))]
    )

    with patch("agent.auxiliary_client.call_llm", return_value=response) as mock_call:
        result = _apply_xai_auto_speech_tags(
            "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
        )

    assert result == "[warmly] Hi."
    mock_call.assert_called_once()
    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs["task"] == "tts_audio_tags"
    assert call_kwargs["temperature"] == 0.7

    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    system_prompt = messages[0]["content"]
    # All documented inline + wrapping tag names must appear in the prompt
    # so the auxiliary model knows what's valid. The prompt lists them
    # comma-separated in two example lines ("Valid inline tags (use as
    # `[tag]`): pause, long-pause, ..." and a similar line for wrapping).
    for tag in _XAI_INLINE_SPEECH_TAGS:
        assert tag in system_prompt, (
            f"inline tag {tag!r} missing from system prompt"
        )
    for tag in _XAI_WRAPPING_SPEECH_TAGS:
        assert tag in system_prompt, (
            f"wrapping tag {tag!r} missing from system prompt"
        )
    # The prompt must explicitly show the BBCode-style closing syntax so
    # the rewriter uses [/tag] and not <tag>...</tag>.
    assert "[/tag]" in system_prompt

    # The user message carries the locally pause-tagged transcript (the
    # conservative fallback the rewriter is asked to enrich).
    assert "TRANSCRIPT TO TAG" in messages[1]["content"]
    assert "[pause]" in messages[1]["content"]


def test_auto_speech_tags_strips_markdown_fences_from_rewriter_output():
    """If the auxiliary model wraps its reply in ```...``` fences the
    function must strip them before returning.
    """
    fenced = "```\n[warmly] Bonjour. [soft laugh]\n```"
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=fenced))]
    )

    with patch("agent.auxiliary_client.call_llm", return_value=response):
        result = _apply_xai_auto_speech_tags(
            "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
        )

    assert result == "[warmly] Bonjour. [soft laugh]"


def test_generate_xai_tts_omits_text_normalization_when_explicit_false(
    tmp_path, monkeypatch
):
    """text_normalization: false is the API default; field is not sent."""
    captured = {}

    fake_response = Mock()
    fake_response.content = b"mp3"
    fake_response.raise_for_status.return_value = None

    def fake_post(url, headers, json, timeout, stream=False):
        captured["json"] = json
        return fake_response

    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setattr("requests.post", fake_post)

    _generate_xai_tts(
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en", "text_normalization": False}},
    )

    assert "text_normalization" not in captured["json"]
