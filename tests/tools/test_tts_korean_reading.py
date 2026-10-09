import json
from pathlib import Path

from tools import tts_tool


KOREAN_EDGE_CONFIG = {
    "provider": "edge",
    "edge": {"voice": "ko-KR-SunHiNeural"},
}


def _normalize(text: str, config: dict | None = None) -> str:
    cfg = config or KOREAN_EDGE_CONFIG
    return tts_tool._normalize_korean_tts_reading(text, cfg["provider"], cfg)


def test_temperature_units_are_rewritten_for_korean_tts():
    result = _normalize("현재 기온은 23°C, 체감 -3.5℃, 최저기온 0도입니다.")

    assert "영상 이십삼 도" in result
    assert "영하 삼 점 오 도" in result
    assert "최저기온 영 도" in result


def test_dollar_notation_is_rewritten_for_korean_tts():
    result = _normalize("매출은 $1.2B, 비용은 US$35.50, 현금은 USD 1,250입니다.")

    assert "십이억 달러" in result
    assert "삼십오 달러 오십 센트" in result
    assert "천이백오십 달러" in result


def test_stock_indices_values_and_percentages_are_rewritten_for_korean_tts():
    result = _normalize("S&P 500 5,430.25, NASDAQ 100 +1.2%, KOSPI 3,100.5")

    assert "에스앤피 오백 오천사백삼십 점 이 오" in result
    assert "나스닥 백 플러스 일 점 이 퍼센트" in result
    assert "코스피 삼천백 점 오" in result


def test_korean_reading_normalization_is_auto_disabled_for_non_korean_voice():
    config = {"provider": "edge", "edge": {"voice": "en-US-AriaNeural"}}

    assert _normalize("현재 기온은 23°C, S&P 500 +1.2%", config) == "현재 기온은 23°C, S&P 500 +1.2%"


def test_text_to_speech_tool_sends_normalized_text_to_provider(tmp_path, monkeypatch):
    captured = {}

    async def fake_edge(text: str, output_path: str, _tts_config: dict) -> str:
        captured["text"] = text
        Path(output_path).write_bytes(b"mp3")
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: KOREAN_EDGE_CONFIG)
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", fake_edge)

    out = tmp_path / "speech.mp3"
    result = json.loads(tts_tool.text_to_speech_tool("기온 23°C, S&P 500 +1.2%, $12.34", output_path=str(out)))

    assert result["success"] is True
    assert captured["text"] == "기온 영상 이십삼 도, 에스앤피 오백 플러스 일 점 이 퍼센트, 십이 달러 삼십사 센트"
