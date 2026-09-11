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


def test_prepare_spoken_text_strips_reasoning_display_blocks():
    # Gateway reasoning display blocks (show_reasoning) are display-only — never speech.
    # subtext style (Discord default)
    assert prepare_spoken_text("-# 💭 Reasoning\n-# think about it\n\nAnswer.") == "Answer."
    # blockquote style
    assert prepare_spoken_text("> 💭 **Reasoning:**\n> think\n\nAnswer.") == "Answer."
    # code-fence style
    assert prepare_spoken_text("💭 **Reasoning:**\n```\nthink\n```\n\nAnswer.") == "Answer."


def test_prepare_spoken_text_literal_think_mention_not_swallowed():
    # A literal "<think>" mention (not a real block) must not eat the rest of the message.
    assert "Here is the answer" in prepare_spoken_text(
        "You asked about <think> blocks. Here is the answer.")
    # Same inside a reasoning display block: it is stripped WITH the reasoning, answer survives.
    assert prepare_spoken_text("-# 💭 Reasoning\n-# rather than <think> variants\n\nReal answer.") == "Real answer."
