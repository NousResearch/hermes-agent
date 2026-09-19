"""Test paragraph-pause synthesis for local TTS (issue #103103)."""

from tools.tts_tool_local import PARAGRAPH_MARKER, _split_paragraphs, _generate_silence_pcm


class TestParagraphSplit:
    def test_split_paragraphs_with_marker(self):
        text = f"First{PARAGRAPH_MARKER}Second{PARAGRAPH_MARKER}Third"
        paras = _split_paragraphs(text)
        assert paras == ["First", "Second", "Third"]

    def test_split_paragraphs_no_marker_returns_single(self):
        text = "One long sentence."
        paras = _split_paragraphs(text)
        assert paras == ["One long sentence."]

    def test_split_paragraphs_strips_whitespace(self):
        text = f"  First  {PARAGRAPH_MARKER}  Second  "
        paras = _split_paragraphs(text)
        assert paras == ["First", "Second"]


class TestSilenceGeneration:
    def test_generate_silence_pcm(self):
        # 24kHz, 500ms, mono → 12000 samples → 24000 bytes (16-bit PCM)
        pcm = _generate_silence_pcm(500, 24000, 1)
        assert len(pcm) == 24000  # 12000 samples × 2 bytes
        assert pcm == b"\x00" * 24000  # all zeros
