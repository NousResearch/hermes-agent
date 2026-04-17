"""Tests for sentence-aware TTS chunking helpers."""

from tools import tts_tool


class TestChunkTtsText:
    def test_short_text_returns_single_chunk(self):
        chunks = tts_tool.chunk_tts_text("A short answer.", max_chars=50)

        assert chunks == ["A short answer."]

    def test_prefers_sentence_boundaries(self):
        text = "First sentence. Second sentence is a little longer. Third sentence."

        chunks = tts_tool.chunk_tts_text(text, max_chars=45)

        assert chunks == [
            "First sentence.",
            "Second sentence is a little longer.",
            "Third sentence.",
        ]

    def test_combines_short_paragraphs_instead_of_flushing_every_break(self):
        text = "First short paragraph.\n\nSecond short paragraph.\n\nThird short paragraph."

        chunks = tts_tool.chunk_tts_text(text, max_chars=60)

        assert chunks == [
            "First short paragraph.\n\nSecond short paragraph.",
            "Third short paragraph.",
        ]

    def test_splits_long_sentence_at_word_boundaries(self):
        text = "word " * 30

        chunks = tts_tool.chunk_tts_text(text, max_chars=40)

        assert len(chunks) > 1
        assert all(len(chunk) <= 40 for chunk in chunks)
        assert " ".join(chunk.strip() for chunk in chunks).strip() == text.strip()
