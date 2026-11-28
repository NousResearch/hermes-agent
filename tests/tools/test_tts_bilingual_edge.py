"""Unit tests for bilingual (Persian + English) Edge TTS synthesis and partitioning."""

import pytest
from tools.tts_tool_providers import (
    _has_persian_script,
    _has_latin_script,
    _partition_bilingual_edge_text,
)


def test_has_persian_script():
    assert _has_persian_script("درود بر شما")
    assert _has_persian_script("وضعیت docker چطوره؟")
    assert not _has_persian_script("Hello world, docker status is ok.")


def test_has_latin_script():
    assert _has_latin_script("Hello world")
    assert _has_latin_script("وضعیت docker چطوره؟")
    assert not _has_latin_script("درود بر شما وضعیت عالی است")


def test_partition_monolingual_persian():
    text = "سلام هرمس! وضعیت سیستم عالی است و همه‌چیز به درستی کار می‌کند."
    parts = _partition_bilingual_edge_text(text, "en-US-AndrewMultilingualNeural", "fa-IR-DilaraNeural")
    assert len(parts) == 1
    assert parts[0][1] == "fa-IR-DilaraNeural"
    assert parts[0][0] == text


def test_partition_monolingual_english():
    text = "Hello Hermes! All systems are operational. Have a wonderful day!"
    parts = _partition_bilingual_edge_text(text, "en-US-AndrewMultilingualNeural", "fa-IR-DilaraNeural")
    assert len(parts) == 1
    assert parts[0][1] == "en-US-AndrewMultilingualNeural"
    assert "All systems" in parts[0][0]


def test_partition_bilingual_mixed():
    text = (
        "درود بر شما! وضعیت سیستم بررسی شد.\n"
        "All docker containers are healthy and running on port 18434.\n"
        "در صورت نیاز به راهنمایی بیشتر، به من پیام دهید."
    )
    parts = _partition_bilingual_edge_text(text, "en-US-AndrewMultilingualNeural", "fa-IR-DilaraNeural")
    assert len(parts) == 3
    assert parts[0][1] == "fa-IR-DilaraNeural"
    assert "درود بر شما" in parts[0][0]
    assert parts[1][1] == "en-US-AndrewMultilingualNeural"
    assert "docker containers" in parts[1][0]
    assert parts[2][1] == "fa-IR-DilaraNeural"
    assert "پیام دهید" in parts[2][0]
