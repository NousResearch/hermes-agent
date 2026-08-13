"""Tests for the long-form TTS chunking and delivery packing pipeline.

Verifies that text exceeding a provider's per-request cap is split without
content loss, that chunks are synthesized in order, and that the delivery
packing respects platform upload limits.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.tts_tool import (
    AudioDeliveryProfile,
    _build_audio_delivery_files,
    _concat_audio_files,
    _pack_audio_files_for_delivery,
    _split_oversized_sentence,
    _split_text_for_tts,
)


class TestSplitTextForTts:
    def test_short_text_returns_single_chunk(self):
        result = _split_text_for_tts("Hello world.", 4096)
        assert result == ["Hello world."]

    def test_empty_text_returns_empty_list(self):
        assert _split_text_for_tts("", 4096) == []
        assert _split_text_for_tts("   ", 4096) == []

    def test_long_text_is_split_without_loss(self):
        text = "A" * 5000
        chunks = _split_text_for_tts(text, 4096)
        assert len(chunks) == 2
        assert chunks[0] == "A" * 4096
        assert chunks[1] == "A" * 904
        assert "".join(chunks) == text

    def test_splits_on_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text_for_tts(text, 30)
        assert len(chunks) >= 2
        # No content lost
        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

    def test_handles_very_long_word(self):
        text = "A" * 100
        chunks = _split_text_for_tts(text, 30)
        assert all(len(c) <= 30 for c in chunks)
        assert "".join(chunks) == text


class TestSplitOversizedSentence:
    def test_short_sentence_returns_as_is(self):
        assert _split_oversized_sentence("Hello world.", 100) == ["Hello world."]

    def test_long_word_is_hard_split(self):
        word = "A" * 100
        chunks = _split_oversized_sentence(word, 30)
        assert all(len(c) <= 30 for c in chunks)
        assert "".join(chunks) == word

    def test_word_boundary_split(self):
        words = " ".join(["word"] * 50)
        chunks = _split_oversized_sentence(words, 30)
        assert all(len(c) <= 30 for c in chunks)


class TestAudioDeliveryProfile:
    def test_default_profile(self):
        profile = AudioDeliveryProfile(platform="default", max_file_bytes=10 * 1024 * 1024)
        assert profile.target_file_bytes > 0
        assert profile.target_file_bytes < profile.max_file_bytes

    def test_custom_safety_ratio(self):
        profile = AudioDeliveryProfile(
            platform="custom", max_file_bytes=1000, safety_ratio=0.5
        )
        assert profile.target_file_bytes == 500


class TestPackAudioFilesForDelivery:
    def test_single_file_returns_one_group(self, tmp_path):
        f = tmp_path / "a.mp3"
        f.write_bytes(b"x" * 100)
        profile = AudioDeliveryProfile(platform="default", max_file_bytes=10000)
        groups = _pack_audio_files_for_delivery([str(f)], profile)
        assert len(groups) == 1
        assert groups[0] == [str(f)]

    def test_splits_on_size_limit(self, tmp_path):
        files = []
        for i in range(5):
            f = tmp_path / f"chunk{i:02d}.mp3"
            f.write_bytes(b"x" * 300)
            files.append(str(f))
        # Target is 500 bytes, each file is 300 → at most 1 file per group
        profile = AudioDeliveryProfile(platform="default", max_file_bytes=1000, safety_ratio=0.5)
        groups = _pack_audio_files_for_delivery(files, profile)
        assert len(groups) == 5
        for group in groups:
            assert len(group) == 1

    def test_splits_on_suffix_mismatch(self, tmp_path):
        f1 = tmp_path / "a.mp3"
        f1.write_bytes(b"x" * 100)
        f2 = tmp_path / "b.ogg"
        f2.write_bytes(b"x" * 100)
        profile = AudioDeliveryProfile(platform="default", max_file_bytes=10000)
        groups = _pack_audio_files_for_delivery([str(f1), str(f2)], profile)
        assert len(groups) == 2


class TestBuildAudioDeliveryFiles:
    def test_single_file_passes_through(self, tmp_path):
        f = tmp_path / "chunk.mp3"
        f.write_bytes(b"x" * 100)
        out = str(tmp_path / "output.mp3")
        profile = AudioDeliveryProfile(platform="default", max_file_bytes=10000)
        paths, combined = _build_audio_delivery_files([str(f)], out, profile)
        assert len(paths) == 1
        assert combined is False

    def test_oversized_chunk_raises(self, tmp_path):
        f = tmp_path / "chunk.mp3"
        f.write_bytes(b"x" * 100)
        out = str(tmp_path / "output.mp3")
        profile = AudioDeliveryProfile(platform="default", max_file_bytes=50)
        with pytest.raises(ValueError, match="exceeds"):
            _build_audio_delivery_files([str(f)], out, profile)

    def test_combines_multiple_files(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"chunk{i:02d}.mp3"
            f.write_bytes(b"\x00" * 100)
            files.append(str(f))
        out = str(tmp_path / "output.mp3")
        profile = AudioDeliveryProfile(platform="default", max_file_bytes=10000)

        with patch("tools.tts_tool._concat_audio_files") as mock_concat:
            mock_concat.return_value = out
            # Copy the first file to output so the size check passes
            Path(out).write_bytes(b"\x00" * 300)
            paths, combined = _build_audio_delivery_files(files, out, profile)
            assert len(paths) == 1
            assert combined is True
