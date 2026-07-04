from tools.tts_tool import prepare_voice_mode_tts_text


def test_prepare_voice_mode_tts_text_prefers_explicit_voice_section():
    text = """<voice>Short version: I fixed the wiring. Details are in chat.</voice>

## Technical details
```bash
pytest tests/tools/test_tts.py
```
"""
    assert prepare_voice_mode_tts_text(text) == "Short version: I fixed the wiring. Details are in chat."


def test_prepare_voice_mode_tts_text_stops_before_detail_heading():
    text = """It is working now. The remaining risk is first-start latency.

## Details
- /Users/johngalt/private/path
- pytest tests/gateway/test_voice.py
"""
    assert prepare_voice_mode_tts_text(text) == "It is working now. The remaining risk is first-start latency."


def test_prepare_voice_mode_tts_text_skips_code_blocks_and_tables():
    text = """The short answer is no. That is the wrong trade-off.

```python
print('do not speak this')
```

| command | result |
| --- | --- |
"""
    spoken = prepare_voice_mode_tts_text(text)
    assert "print" not in spoken
    assert "command" not in spoken
    assert spoken.startswith("The short answer is no")


def test_prepare_voice_mode_tts_text_caps_long_text_on_sentence_boundary():
    text = "First sentence is useful. " + ("word " * 500)
    spoken = prepare_voice_mode_tts_text(text, max_chars=80)
    assert len(spoken) <= 81
    assert spoken.endswith(".")


def test_prepare_voice_mode_tts_text_does_not_truncate_without_cap():
    text = "First sentence is useful. " + ("word " * 500)
    spoken = prepare_voice_mode_tts_text(text)
    assert len(spoken) > 1800
    assert spoken.startswith("First sentence is useful.")
    assert spoken.endswith("word")


def test_prepare_voice_mode_tts_text_keeps_more_than_two_conversational_paragraphs():
    text = """First conversational paragraph.

Second conversational paragraph.

Third conversational paragraph.

Fourth conversational paragraph."""

    spoken = prepare_voice_mode_tts_text(text, max_chars=500)

    assert "First conversational paragraph" in spoken
    assert "Second conversational paragraph" in spoken
    assert "Third conversational paragraph" in spoken
    assert "Fourth conversational paragraph" in spoken
