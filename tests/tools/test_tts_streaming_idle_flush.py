"""Idle flushes must never split a word across two TTS requests.

Model deltas arrive on arbitrary character boundaries, so a producer that stalls
mid-generation routinely leaves the chunker buffer ending inside a word. Every
flushed fragment becomes its own synthesis request, and a half word synthesized
alone does not reassemble with its other half in the next request.
"""

from __future__ import annotations

import pytest

from tools.tts_streaming import SentenceChunker


def _stalled(buf: str) -> SentenceChunker:
    chunker = SentenceChunker()
    chunker.buf = buf
    return chunker


def test_idle_flush_emits_whole_words_and_keeps_the_partial_one():
    chunker = _stalled("the rehearsal covers regress")

    assert chunker.flush_complete_words() == ["the rehearsal covers"]
    assert chunker.buf == " regress"

    assert chunker.feed("ion tests.") == []  # no whitespace after the period yet: not a boundary
    assert chunker.flush() == ["regression tests."]


def test_idle_flush_holds_a_reply_that_is_one_unfinished_word():
    chunker = _stalled("Supercalifragilistic")

    assert chunker.flush_complete_words() == []
    assert chunker.feed("expialidocious is a long word. ") == ["Supercalifragilisticexpialidocious is a long word. "]


def test_idle_flush_without_a_word_to_hold_drains_whole_as_before():
    """A runaway token, or a script written without spaces: same as flush()."""
    for buf in ("a" * 3000, "这是regress", "然后我们再说第二句话这一句要长一些才行", "ภาษาไทยไม่มีช่องว่าง"):
        chunker = _stalled(buf)
        assert chunker.flush_complete_words() == [buf]
        assert chunker.buf == ""


def test_a_held_word_stays_held_on_the_next_idle_poll():
    chunker = _stalled("the rehearsal covers regress")
    chunker.flush_complete_words()

    assert chunker.flush_complete_words() == []
    assert chunker.flush() == ["regress"]


def test_idle_flush_never_holds_back_a_tail_longer_than_a_word():
    """A runaway token is not a word to protect; holding it would grow one request past the
    provider's text cap, which the speaker truncates."""
    buf = "hello " + "a" * 3000
    chunker = _stalled(buf)

    assert chunker.flush_complete_words() == [buf]
    assert chunker.buf == ""


def test_idle_flush_keeps_a_closed_markdown_construct_in_one_piece():
    """A buffer ending on a closing fence or bracket ends on a finished token, and the markdown
    stripper needs both fences in one fragment."""
    for buf in ("Here is code\n```py\nx = 1\n```", "See [the docs](https://example.com/a)"):
        chunker = _stalled(buf)
        assert chunker.flush_complete_words() == [buf.strip()]
        assert chunker.buf == ""


def test_idle_flush_holds_an_open_reasoning_block():
    """Same contract as feed(): nothing inside an unclosed <think> is speakable yet."""
    buf = "<think>" + "reasoning " * 20 + "SECRET"
    chunker = _stalled(buf)

    assert chunker.flush_complete_words() == []
    assert chunker.buf == buf
    assert chunker.feed("</think>Public reply is long enough. ") == ["Public reply is long enough. "]


def test_idle_flush_treats_unicode_whitespace_as_a_word_boundary():
    chunker = _stalled("hello\u00a0there\u00a0regress")

    assert chunker.flush_complete_words() == ["hello\u00a0there"]
    assert chunker.buf == "\u00a0regress"


def test_idle_flush_emits_everything_when_the_buffer_ends_on_whitespace():
    chunker = _stalled("all of these words are finished ")

    assert chunker.flush_complete_words() == ["all of these words are finished"]
    assert chunker.buf.strip() == ""


def test_idle_flush_drops_think_blocks_like_flush_does():
    chunker = _stalled("<think>plan</think>speaking now about somethi")

    assert chunker.flush_complete_words() == ["speaking now about"]
    assert "think" not in chunker.buf


def test_spoken_text_is_identical_with_or_without_idle_flushes():
    """Joining every emitted fragment reproduces the source text word for word,
    whichever character offset the producer stalls at."""
    text = "Streaming speech keeps words whole even when generation crawls along."
    for stall_at in range(1, len(text)):
        chunker = SentenceChunker()
        out = chunker.feed(text[:stall_at])
        out += chunker.flush_complete_words()
        out += chunker.feed(text[stall_at:])
        out += chunker.flush()
        assert " ".join(p.strip() for p in out).split() == text.split(), stall_at


# ── Both idle-flush call sites ───────────────────────────────────────────

_LONG = (
    "This reply is long enough that the speaker's idle flush fires while the model is "
    "still generating, and the stall lands halfway through the word regress"
)
_REST = "ion tests, which must still come out as one word."


class _StallingQueue:
    """Queue stand-in that reports one idle poll between the partial and the rest."""

    def __init__(self, items):
        self._items = list(items)

    def get(self, timeout=None):
        import queue

        item = self._items.pop(0)
        if item is _STALL:
            raise queue.Empty
        return item

    def get_nowait(self):
        import queue

        raise queue.Empty


_STALL = object()


def _assert_no_split_words(requests, source):
    """Every provider request holds whole words: rejoined, they are the source's words."""
    assert " ".join(r.strip() for r in requests).split() == source.split(), requests


def _run_speaker(items):
    import threading
    from unittest.mock import MagicMock, patch

    import tools.tts_streaming as ts
    from tools import tts_tool
    from tools.tts_tool_speaker import stream_tts_to_speaker

    requests: list[str] = []

    class _Tracking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            requests.append(text)
            yield b"\x00\x00" * 10

    sd = MagicMock()
    sd.OutputStream.return_value = MagicMock()
    stop, done = threading.Event(), threading.Event()
    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=_Tracking({}, {})), \
         patch.object(tts_tool, "_load_tts_config", return_value={}), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        stream_tts_to_speaker(_StallingQueue(items), stop, done)
    assert done.is_set()
    return requests


def test_cli_speaker_does_not_drop_a_partial_flush_as_a_repeated_sentence():
    """The speaker skips sentences it already said (LLM repetition). A word-safe idle fragment is
    not a sentence: dropping it because it matches an earlier sentence loses prose."""
    pytest.importorskip("numpy")
    phrase = ("The deployment finished and every service reported healthy within the expected window "
              "today, so the rollout can continue")
    requests = _run_speaker([phrase + ". ", phrase + " cand", _STALL, "idate.", None])

    assert " ".join(requests).split() == (phrase + ". " + phrase + " candidate.").split(), requests


def test_cli_speaker_still_skips_a_repeated_sentence_drained_at_idle():
    """Repetition filtering is unchanged when the idle flush drains the whole buffer."""
    pytest.importorskip("numpy")
    sentence = ("The deployment finished and every service reported healthy within the expected "
                "window today, so we continue.")
    requests = _run_speaker([sentence + " ", sentence, _STALL, None])

    assert requests == [sentence], requests


def test_cli_speaker_idle_flush_keeps_words_whole():
    pytest.importorskip("numpy")
    requests = _run_speaker([_LONG, _STALL, _REST, None])

    assert len(requests) >= 2, requests  # the idle flush really fired mid-reply
    assert not requests[0].rstrip().endswith("regress"), requests
    _assert_no_split_words(requests, _LONG + _REST)
