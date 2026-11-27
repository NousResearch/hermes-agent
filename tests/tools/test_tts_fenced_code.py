"""Fenced code stays silent when assistant replies are spoken."""

import pytest

from tools.tts_text_normalize import prepare_spoken_text


@pytest.mark.parametrize("fence", ["```", "~~~", "````", "~~~~"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_fenced_code_is_omitted_from_speech(fence, newline):
    reply = newline.join(["Before.", f"{fence}python", "print('do not read this')", fence, "After."])

    assert prepare_spoken_text(reply) == "Before. After."


def test_shorter_fence_inside_code_does_not_end_the_block():
    reply = "Before.\n````markdown\n```python\nprint('do not read this')\n```\n````\nAfter."

    assert prepare_spoken_text(reply) == "Before. After."


def test_inline_backticks_do_not_open_a_fenced_block():
    reply = "```inline code```\nordinary prose\n```\nsecret\n```\nAfter."

    spoken = prepare_spoken_text(reply)

    assert "inline code" in spoken
    assert "ordinary prose" in spoken
    assert "secret" not in spoken
    assert "After." in spoken
