"""Korean-first spoken-content policy for gateway/persona TTS."""

from tools.tts_tool import prepare_korean_first_spoken_text


def test_english_answer_fails_closed_to_korean_notice():
    spoken = prepare_korean_first_spoken_text(
        "Deployment is complete. Check the logs for details.",
        user_text="status?",
    )

    assert "Deployment" not in spoken
    assert "logs" not in spoken
    assert "영어 답변" in spoken


def test_explicit_english_output_request_allows_english_verbatim():
    spoken = prepare_korean_first_spoken_text(
        "Deployment is complete. Check the logs for details.",
        user_text="Please answer in English voice for this turn.",
    )

    assert spoken == "Deployment is complete. Check the logs for details."


def test_korean_summary_is_used_and_embedded_technical_terms_are_preserved():
    spoken = prepare_korean_first_spoken_text(
        "Done. 요약: FastAPI provider 설정은 유지했고 Fish Audio 동작도 바꾸지 않았어요. "
        "Run `python -m pytest` and open https://example.com for details.",
        user_text="status?",
    )

    assert "요약:" in spoken
    assert "FastAPI provider" in spoken
    assert "Fish Audio" in spoken
    assert "Done" not in spoken
    assert "python -m pytest" not in spoken
    assert "https://example.com" not in spoken


def test_code_blocks_paths_urls_and_shell_commands_are_not_spoken_as_fallback_content():
    spoken = prepare_korean_first_spoken_text(
        """```bash
python -m pytest tests/gateway/test_voice_command.py
```
/tmp/hermes_voice/reply.mp3
https://example.com
git status --short
""",
        user_text="read this back",
    )

    assert "python -m pytest" not in spoken
    assert "/tmp/hermes_voice" not in spoken
    assert "https://example.com" not in spoken
    assert "git status" not in spoken
    assert "영어 답변" in spoken
