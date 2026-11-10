"""``SentenceChunker`` never cuts a bracketed delivery cue in half (issue #152238).

A ``transform_spoken_text`` plugin sees one sentence at a time while streaming, so a tag such as
``[whispers. softly]`` must reach it whole: a boundary inside an unclosed ``[`` waits for the
``]``, idle / segment flushes keep the open tag buffered, and the end-of-reply flush drains all.
"""

from __future__ import annotations

from tools.tts_streaming import SentenceChunker


def _feed_all(chunker, deltas):
    out = []
    for delta in deltas:
        out.extend(chunker.feed(delta))
    return out


def test_boundary_inside_an_open_tag_waits_for_the_close():
    chunker = SentenceChunker(min_len=1)

    assert chunker.feed("Here it comes. [whispers. softly") == ["Here it comes. "]
    assert chunker.feed("] It passed. ") == ["[whispers. softly] It passed. "]


def test_tag_split_across_deltas_arrives_whole():
    chunker = SentenceChunker(min_len=1)

    sentences = _feed_all(chunker, ["Done. [sig", "hs. long", "ly] Fine. ", "Next one."])
    sentences += chunker.flush()

    assert sentences == ["Done. ", "[sighs. longly] Fine. ", "Next one."]


def test_idle_flush_keeps_the_open_tag_buffered():
    chunker = SentenceChunker(min_len=1)
    chunker.feed("Let me check [whisp")

    assert chunker.flush(hold_open_tag=True) == ["Let me check"]
    assert chunker.feed("ers] quietly. ") == ["[whispers] quietly. "]


def test_idle_flush_without_an_open_tag_drains_everything():
    chunker = SentenceChunker(min_len=1)
    chunker.feed("Let me check [this] now")

    assert chunker.flush(hold_open_tag=True) == ["Let me check [this] now"]
    assert chunker.buf == ""


def test_final_flush_drains_an_unclosed_tag():
    chunker = SentenceChunker(min_len=1)
    chunker.feed("Bye [sigh")

    assert chunker.flush() == ["Bye [sigh"]


def test_bracket_left_open_past_the_cap_is_treated_as_text():
    chunker = SentenceChunker(min_len=1)
    text = "[" + "word " * 20 + "ends. Next sentence. "

    assert chunker.feed(text)[0] == "[" + "word " * 20 + "ends. "
