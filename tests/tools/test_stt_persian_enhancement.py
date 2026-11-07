"""Unit tests for Persian STT normalization and bilingual initial prompt enhancement."""

from tools.transcription_local import _normalize_stt_transcript, build_local_transcribe_kwargs


def test_persian_character_harmonization():
    raw = "درود اين يک پيام است"
    # Arabic yeh \u064a and kaf \u0643 should be converted to Persian \u06cc and \u06a9
    normalized = _normalize_stt_transcript(raw)
    assert "این" in normalized
    assert "یک" in normalized


def test_persian_colloquial_phonetic_fixes():
    raw = "تروت سلام چتوری خبستی آیا پیامن رو دریاف کردی خوب وزید هوا ایم روز در موصال چتوره"
    normalized = _normalize_stt_transcript(raw)
    assert "درود" in normalized
    assert "چطوری" in normalized
    assert "خسته نباشی" in normalized
    assert "پیامم" in normalized
    assert "دریافت کردی" in normalized
    assert "وضعیت آب و هوا" in normalized
    assert "امروز" in normalized
    assert "ماسال" in normalized


def test_build_local_transcribe_kwargs_has_default_prompt():
    kwargs = build_local_transcribe_kwargs({})
    assert "initial_prompt" in kwargs
    assert "فارسی" in kwargs["initial_prompt"]
