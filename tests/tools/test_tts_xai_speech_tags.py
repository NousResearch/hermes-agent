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


def test_apply_xai_auto_speech_tags_preserves_all_documented_xai_tags():
    text = "Bonjour Monsieur Talbot. [sigh] <slow>Je parle lentement.</slow> <emphasis>Important.</emphasis>"

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


def test_apply_xai_auto_speech_tags_single_newline_still_gets_first_sentence_pause():
    """A single newline isn't a paragraph break — no ``[pause]`` injected by
    the paragraph pass, so the first-sentence pause MUST still fire.
    Guards against the fix being too greedy.
    """
    text = "Welcome to the demo of our new product line.\nIt has many features."
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

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
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


def test_auto_speech_tags_strips_markdown_fence_with_language_hint():
    """The fence regex accepts an optional language tag like ```text ...```."""
    fenced = "```text\n[warmly] Bonjour.\n```"
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=fenced))]
    )

    with patch("agent.auxiliary_client.call_llm", return_value=response):
        result = _apply_xai_auto_speech_tags(
            "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
        )

    assert result == "[warmly] Bonjour."


def test_auto_speech_tags_falls_back_to_local_on_auxiliary_exception(caplog):
    """If the auxiliary rewriter raises (timeout, network, provider error,
    anything) the function must silently fall back to the local
    pause-tagged text so the user still gets audio.
    """
    import logging

    with caplog.at_level(logging.DEBUG, logger="tools.tts_tool"), patch(
        "agent.auxiliary_client.call_llm",
        side_effect=RuntimeError("upstream provider timed out"),
    ):
        result = _apply_xai_auto_speech_tags(
            "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
        )

    # Local fallback: first sentence gets a [pause] inserted, single
    # paragraph, no other rewriter activity.
    assert result == (
        "Bonjour Monsieur Talbot. [pause] Ceci est un test de réponse vocale."
    )
    assert "xAI TTS audio tag rewrite failed" in caplog.text


def test_auto_speech_tags_falls_back_to_local_when_rewriter_returns_empty():
    """An empty / None rewriter response must also fall back to local."""
    empty_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
    )

    with patch(
        "agent.auxiliary_client.call_llm", return_value=empty_response
    ):
        result = _apply_xai_auto_speech_tags(
            "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
        )

    assert result == (
        "Bonjour Monsieur Talbot. [pause] Ceci est un test de réponse vocale."
    )


def test_auto_speech_tags_skips_auxiliary_when_input_has_explicit_tags():
    """If the user/model already supplied explicit speech tags we trust
    them and never call the rewriter — that would risk the rewriter
    overwriting intentional markup.
    """
    tagged = "Bonjour. [pause] <whisper>Déjà balisé.</whisper>"

    with patch("agent.auxiliary_client.call_llm") as mock_call:
        result = _apply_xai_auto_speech_tags(tagged)

    mock_call.assert_not_called()
    # The local pass is a no-op for already-tagged text (no double
    # paragraph normalization, no first-sentence pause injection).
    assert result == tagged


def test_auto_speech_tags_skips_auxiliary_for_empty_input():
    with patch("agent.auxiliary_client.call_llm") as mock_call:
        assert _apply_xai_auto_speech_tags("") == ""
        assert _apply_xai_auto_speech_tags("   \n  ") == "   \n  "

    mock_call.assert_not_called()


def test_auto_speech_tags_skips_auxiliary_for_whitespace_only_input():
    """Whitespace-only input short-circuits before the rewriter runs."""
    with patch("agent.auxiliary_client.call_llm") as mock_call:
        assert _apply_xai_auto_speech_tags("   ") == "   "

    mock_call.assert_not_called()


@pytest.mark.parametrize("bad_response", [None, SimpleNamespace(choices=[])])
def test_auto_speech_tags_falls_back_to_local_on_malformed_rewriter_response(
    bad_response,
):
    """Both ``None`` and a response with no choices must fall back to the
    conservative local pass rather than crash.
    """
    with patch(
        "agent.auxiliary_client.call_llm", return_value=bad_response
    ):
        result = _apply_xai_auto_speech_tags(
            "Bonjour Monsieur Talbot. Ceci est un test de réponse vocale."
        )

    assert result == (
        "Bonjour Monsieur Talbot. [pause] Ceci est un test de réponse vocale."
    )


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


def test_generate_xai_tts_omits_speed_and_latency_by_default(tmp_path, monkeypatch):
    """No speed / optimize_streaming_latency in the request body unless
    the user explicitly sets them. Keeps the existing minimal-payload
    contract for default configs.
    """
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
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en"}},
    )

    assert "speed" not in captured["json"]
    assert "optimize_streaming_latency" not in captured["json"]


def test_generate_xai_tts_sends_speed_when_set(tmp_path, monkeypatch):
    """tts.xai.speed flows into the POST body."""
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
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en", "speed": 1.5}},
    )

    assert captured["json"]["speed"] == 1.5


def test_generate_xai_tts_speed_clamped_to_valid_range(tmp_path, monkeypatch):
    """speed values outside xAI's 0.7..1.5 band are clamped, not sent raw."""
    captured = {}

    fake_response = Mock()
    fake_response.content = b"mp3"
    fake_response.raise_for_status.return_value = None

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return fake_response

    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setattr("requests.post", fake_post)

    # Below 0.7 -> 0.7
    _generate_xai_tts(
        "Hello.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "eve", "language": "en", "speed": 0.1}},
    )
    assert captured["json"]["speed"] == 0.7

    # Above 1.5 -> 1.5
    _generate_xai_tts(
        "Hello.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "eve", "language": "en", "speed": 3.0}},
    )
    assert captured["json"]["speed"] == 1.5


def test_generate_xai_tts_omits_speed_when_exactly_default(tmp_path, monkeypatch):
    """speed == 1.0 is the API default; the field stays out of the payload."""
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
        "Hello.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "eve", "language": "en", "speed": 1.0}},
    )

    assert "speed" not in captured["json"]


def test_generate_xai_tts_sends_optimize_streaming_latency_when_set(tmp_path, monkeypatch):
    """tts.xai.optimize_streaming_latency flows into the POST body."""
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
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en", "optimize_streaming_latency": 2}},
    )

    assert captured["json"]["optimize_streaming_latency"] == 2


def test_generate_xai_tts_optimize_streaming_latency_omitted_at_default(tmp_path, monkeypatch):
    """optimize_streaming_latency == 0 is the API default; field is not sent."""
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
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en", "optimize_streaming_latency": 0}},
    )

    assert "optimize_streaming_latency" not in captured["json"]


def test_generate_xai_tts_global_speed_used_as_fallback(tmp_path, monkeypatch):
    """Global tts.speed is the fallback when tts.xai.speed is unset."""
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
        "Hello.",
        str(tmp_path / "out.mp3"),
        {"speed": 0.8, "xai": {"voice_id": "ara", "language": "en"}},
    )

    assert captured["json"]["speed"] == 0.8


def test_generate_xai_tts_provider_speed_overrides_global(tmp_path, monkeypatch):
    """tts.xai.speed wins over the global tts.speed fallback."""
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
        "Hello.",
        str(tmp_path / "out.mp3"),
        {"speed": 1.5, "xai": {"voice_id": "ara", "language": "en", "speed": 0.7}},
    )

    assert captured["json"]["speed"] == 0.7


def test_generate_xai_tts_omits_text_normalization_by_default(tmp_path, monkeypatch):
    """text_normalization is not sent when unset (API default is False)."""
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
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en"}},
    )

    assert "text_normalization" not in captured["json"]


def test_generate_xai_tts_sends_text_normalization_when_enabled(tmp_path, monkeypatch):
    """tts.xai.text_normalization: true flows into the POST body."""
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
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en", "text_normalization": True}},
    )

    assert captured["json"]["text_normalization"] is True


def test_generate_xai_tts_omits_text_normalization_when_explicit_false(
    tmp_path, monkeypatch
):
    """text_normalization: false is the API default; field is not sent."""
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
        "Hello world.",
        str(tmp_path / "out.mp3"),
        {"xai": {"voice_id": "ara", "language": "en", "text_normalization": False}},
    )

    assert "text_normalization" not in captured["json"]
