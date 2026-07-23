from tools.tts_tool import _strip_reasoning_for_tts


def test_strip_reasoning_code_block_from_voice_text():
    text = "💭 **Reasoning:**\n```\nI should inspect the event first.\n```\n\nThe voice message is ready."
    assert _strip_reasoning_for_tts(text) == "The voice message is ready."


def test_strip_reasoning_blockquote_from_voice_text():
    text = "> 💭 **Reasoning:**\n> I should inspect the event first.\n\nThe voice message is ready."
    assert _strip_reasoning_for_tts(text) == "The voice message is ready."


def test_leave_normal_text_unchanged():
    assert _strip_reasoning_for_tts("The voice message is ready.") == "The voice message is ready."
