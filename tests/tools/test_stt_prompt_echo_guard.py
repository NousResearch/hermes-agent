"""Tests for the STT prompt-echo guard wired into ``transcribe_audio``.

Whisper conditions on ``prompt`` as if it were transcript text that came
immediately before the audio, so a low-confidence decode can continue or
repeat the prompt and return THAT as the transcript. It arrives as an
ordinary 200 OK, so nothing downstream can tell it from real speech.

``stt.prompt`` and the ``pre_transcription`` hook (#84934) made that prompt
reachable for every prompt-capable backend, so the guard is tested at the
shared dispatch boundary rather than per provider:

1. An echoed transcript is replaced by an unbiased retry.
2. The retry call carries no prompt at all.
3. A genuine transcript is returned untouched, with no second call.
4. A short answer that merely shares prompt vocabulary is not an echo.
5. An echo that survives the unbiased retry yields no transcript rather
   than a hallucinated turn.
6. The guard covers faster-whisper's ``initial_prompt``, not just Groq.
7. With no prompt configured the wire traffic is unchanged.

Harness conventions mirror ``tests/tools/test_pre_transcription_hook.py``:
the real ``transcribe_audio`` entry point runs, only the API boundary is
stubbed, and no live model is loaded.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from tools import transcription_tools


# Synthetic fixtures only: never a real profile path or session identifier.
PROMPT = (
    "Quarterly review with Ada Lovelace and Grace Hopper. Topics: telemetry, "
    "backpressure, tail quantiles, Fenwick trees."
)
# Whisper usually returns the prompt lightly corrupted rather than verbatim.
ECHO = (
    "Qquarterly review with Ada Lovelace and Grace Hopper. Topics: telemetry, "
    "backpressure, tail quantiles, Fenwick trees."
)
REAL_SPEECH = "Could you summarise the telemetry section before the call"


def _make_audio(tmp_path):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"fake audio data")
    return str(audio)


def _dispatch_ctx(stt_config, provider):
    """Patch config load + provider resolution around transcribe_audio."""
    return (
        patch("tools.transcription_tools._load_stt_config", return_value=stt_config),
        patch("tools.transcription_tools._get_provider", return_value=provider),
    )


def _groq_client(*transcripts):
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = list(transcripts)
    return client


def _run_groq(monkeypatch, tmp_path, client, stt_config):
    audio = _make_audio(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    cfg_patch, prov_patch = _dispatch_ctx(stt_config, "groq")
    with cfg_patch, prov_patch, \
         patch("tools.transcription_tools._HAS_OPENAI", True), \
         patch("openai.OpenAI", return_value=client):
        return transcription_tools.transcribe_audio(audio)


class TestPromptEchoRetry:
    def test_echoed_prompt_is_replaced_by_the_unbiased_retry(
        self, monkeypatch, tmp_path,
    ):
        client = _groq_client(ECHO, REAL_SPEECH)

        result = _run_groq(monkeypatch, tmp_path, client, {
            "provider": "groq", "prompt": PROMPT,
        })

        assert result["success"] is True
        assert result["transcript"] == REAL_SPEECH

    def test_the_retry_call_carries_no_prompt(self, monkeypatch, tmp_path):
        client = _groq_client(ECHO, REAL_SPEECH)

        _run_groq(monkeypatch, tmp_path, client, {
            "provider": "groq", "prompt": PROMPT,
        })

        calls = client.audio.transcriptions.create.call_args_list
        assert len(calls) == 2, "an echoed decode must be retried exactly once"
        assert calls[0].kwargs.get("prompt") == PROMPT
        assert "prompt" not in calls[1].kwargs

    def test_echo_surviving_the_retry_yields_no_transcript(
        self, monkeypatch, tmp_path,
    ):
        """Two echoes means neither decode is trustworthy.

        Returning the echo would hand the agent a hallucinated turn, which is
        the actual harm; an empty transcript is the honest outcome.
        """
        client = _groq_client(ECHO, ECHO)

        result = _run_groq(monkeypatch, tmp_path, client, {
            "provider": "groq", "prompt": PROMPT,
        })

        assert result["transcript"] == ""


class TestGuardDoesNotOverTrigger:
    def test_genuine_transcript_is_returned_without_a_second_call(
        self, monkeypatch, tmp_path,
    ):
        client = _groq_client(REAL_SPEECH)

        result = _run_groq(monkeypatch, tmp_path, client, {
            "provider": "groq", "prompt": PROMPT,
        })

        assert result["transcript"] == REAL_SPEECH
        assert len(client.audio.transcriptions.create.call_args_list) == 1

    def test_short_answer_sharing_prompt_vocabulary_is_not_an_echo(
        self, monkeypatch, tmp_path,
    ):
        """A brief reply naming a term from the prompt is normal speech.

        This is the false-positive that matters: vocabulary hints exist
        precisely because the speaker is expected to say those words.
        """
        client = _groq_client("Fenwick trees")

        result = _run_groq(monkeypatch, tmp_path, client, {
            "provider": "groq", "prompt": PROMPT,
        })

        assert result["transcript"] == "Fenwick trees"
        assert len(client.audio.transcriptions.create.call_args_list) == 1

    def test_no_prompt_configured_makes_no_second_call(
        self, monkeypatch, tmp_path,
    ):
        """Without a prompt there is nothing to echo, so the wire is unchanged."""
        client = _groq_client(ECHO)

        result = _run_groq(monkeypatch, tmp_path, client, {"provider": "groq"})

        assert result["transcript"] == ECHO
        calls = client.audio.transcriptions.create.call_args_list
        assert len(calls) == 1
        assert "prompt" not in calls[0].kwargs


class TestGuardIsScriptAgnostic:
    def test_cjk_echo_is_retried(self, monkeypatch, tmp_path):
        """An echo in a non-Latin script is still an echo.

        Normalization folds to alphanumerics, so a filter that only kept
        [a-z0-9] would reduce both sides to empty and silently disable the
        guard for CJK, Cyrillic, Greek and every other non-Latin script.
        """
        prompt = "季度评审会议记录：遥测数据、背压控制、尾部分位数、树状数组、缓存命中率。"
        echo = "季季度评审会议记录：遥测数据、背压控制、尾部分位数、树状数组、缓存命中率。"
        speech = "请帮我总结一下遥测部分的内容然后发给我"
        client = _groq_client(echo, speech)

        result = _run_groq(monkeypatch, tmp_path, client, {
            "provider": "groq", "prompt": prompt,
        })

        assert result["transcript"] == speech
        calls = client.audio.transcriptions.create.call_args_list
        assert len(calls) == 2
        assert "prompt" not in calls[1].kwargs


class TestObservability:
    def test_discarded_transcript_is_logged_at_warning(
        self, monkeypatch, tmp_path, caplog,
    ):
        """The discard path must stay visible; a silent drop is unexplainable."""
        client = _groq_client(ECHO, ECHO)

        with caplog.at_level(logging.WARNING, logger="tools.transcription_tools"):
            result = _run_groq(monkeypatch, tmp_path, client, {
                "provider": "groq", "prompt": PROMPT,
            })

        assert result["transcript"] == ""
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno >= logging.WARNING]
        assert any("retrying once without prompt biasing" in m for m in warnings)
        assert any("discarding the" in m for m in warnings)


class TestGuardIsProviderAgnostic:
    def test_faster_whisper_initial_prompt_echo_is_also_retried(
        self, monkeypatch, tmp_path,
    ):
        """The guard sits at the dispatch seam, so it is not Groq specific."""
        audio = _make_audio(tmp_path)

        def _segments(text):
            segment = MagicMock()
            segment.text = text
            segment.no_speech_prob = 0.0
            segment.avg_logprob = 0.0
            return ([segment], MagicMock(language="en", duration=1.0))

        model = MagicMock()
        model.transcribe.side_effect = [_segments(ECHO), _segments(REAL_SPEECH)]

        cfg_patch, prov_patch = _dispatch_ctx(
            {"provider": "local", "prompt": PROMPT}, "local",
        )
        with cfg_patch, prov_patch, \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._load_local_whisper_model",
                   return_value=model), \
             patch("tools.transcription_tools._local_model", None):
            result = transcription_tools.transcribe_audio(audio)

        assert result["transcript"] == REAL_SPEECH
        calls = model.transcribe.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["initial_prompt"] == PROMPT
        assert not calls[1].kwargs.get("initial_prompt")
