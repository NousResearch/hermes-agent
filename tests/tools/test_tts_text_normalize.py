from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from tools.tts_text_normalize import prepare_spoken_text


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        raise AssertionError("not used")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def test_prepare_spoken_text_expands_celsius_and_weather_units():
    raw = """## Christchurch today\n\n- **Now:** about **14°C**, feels like **14°C**\n- **Wind:** 9 km/h\n- **Rain:** 1.3 mm\n- **Range:** 11\u201317°C\n"""

    spoken = prepare_spoken_text(raw)

    assert "##" not in spoken
    assert "**" not in spoken
    assert "14 degrees Celsius" in spoken
    assert "11 to 17 degrees Celsius" in spoken
    assert "9 kilometres per hour" in spoken
    assert "1.3 millimetres" in spoken
    assert "°C" not in spoken
    assert "km/h" not in spoken


def test_prepare_spoken_text_polish_edge_cases():
    # Heading folds into the next sentence as a lead-in, not a bare label.
    assert prepare_spoken_text("## Weather\nIt will be sunny") == "Weather, It will be sunny."
    # Bare degree unit (no leading number) still expands.
    assert "degrees Celsius" in prepare_spoken_text("measured in °C")
    # Trailing comma is not swallowed into the amount.
    assert "300 US dollars" in prepare_spoken_text("US$300, next")
    # Real numeric rates expand, but and/or, N/A, IDs and dates are left intact.
    assert "5 dollars per month" in prepare_spoken_text("$5/month")
    assert "and/or" in prepare_spoken_text("choose and/or option")
    assert "N/A" in prepare_spoken_text("status N/A here")
    assert "2026/06/02" in prepare_spoken_text("due 2026/06/02 ok")


def test_prepare_spoken_text_preserves_inline_tts_control_tags():
    # Inline TTS tags (Boson/Higgs style `<|emotion:sadness|>`, `<|prosody:pause|>` etc.)
    # are engine control tokens, not spoken text: they must survive cleanup byte-for-byte
    # so the speech backend can parse them (regression: pipe/colon rewriting mangled them).
    raw = (
        "<|emotion:sadness|><|prosody:speed_slow|>寒蝉凄切，"
        "<|prosody:long_pause|>对长亭晚。<|sfx:laughter|>Haha"
    )
    spoken = prepare_spoken_text(raw)
    assert "<|emotion:sadness|>" in spoken
    assert "<|prosody:speed_slow|>" in spoken
    assert "<|prosody:long_pause|>" in spoken
    assert "<|sfx:laughter|>" in spoken
    assert "寒蝉凄切" in spoken


def test_prepare_spoken_text_tag_protection_keeps_table_pipe_rewrite():
    # Markdown table pipes still become pauses; only tag pipes are protected.
    assert prepare_spoken_text("a | b | c") == "a; b; c"
