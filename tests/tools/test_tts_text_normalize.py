import pytest

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


def test_prepare_spoken_text_strips_media_file_links():
    # "Open inference-server-shopping-list.xlsx" style tokens must never reach
    # the voice: hyphenated slugs + odd extensions make TTS loop ("eeeeee").
    raw = "The files are below.\nMEDIA:/Users/ricardo.mendes/Documents/inference-server-shopping-list.xlsx\nBye."
    spoken = prepare_spoken_text(raw)
    assert "MEDIA" not in spoken
    assert "shopping-list" not in spoken
    assert "xlsx" not in spoken
    assert "below" in spoken
    assert "Bye" in spoken


def test_prepare_spoken_text_keeps_sentence_break_after_inline_media_link():
    spoken = prepare_spoken_text("See MEDIA:/tmp/report-2026-q3.xlsx. Then reply.")
    assert "report" not in spoken
    assert spoken == "See. Then reply."


def test_prepare_spoken_text_closes_trailing_colons():
    # "the regex list:" + a now-removed raw token would leave the voice hanging
    # on an open colon-pause (the "aaaa" stutter). Close it with a period.
    spoken = prepare_spoken_text("Here is the list:\nMEDIA:/tmp/x.py\nMore text")
    assert "list:" not in spoken
    assert "list." in spoken


def test_prepare_spoken_text_closes_colon_on_single_line():
    # Multi-line text gets colons closed per line; single-line text reaches the
    # end-of-text rule instead.
    assert prepare_spoken_text("Here is the list:") == "Here is the list."
    # ...but a digit-preceded colon is a ratio and must stay intact.
    assert prepare_spoken_text("Final score 3:2") == "Final score 3:2"


@pytest.mark.parametrize("raw,spoken", [
    ("The startup raised $5M last year.", "5 million dollars"),
    ("A $1.5 billion round.", "1.5 billion dollars"),
    ("A €2bn deal.", "2 billion euros"),
    ("A US$300k salary.", "300 thousand US dollars"),
])
def test_money_magnitude_is_spoken_before_the_currency_and_never_as_metres(raw, spoken):
    """A magnitude belongs to the amount ("$5M" is five million dollars); an uppercase M is a
    magnitude, never metres, while a lowercase length still expands."""
    assert spoken in prepare_spoken_text(raw)
    assert "metres" not in prepare_spoken_text(raw + " We hit 10M users.")
    assert "5 metres" in prepare_spoken_text(raw + " The pool is 5 m deep.")


def test_tilde_is_about_only_before_an_amount():
    """"~5 km" is "about 5 km", but the "~" of a home path is part of the path (silenced like
    any path), never the spoken word "about"."""
    assert prepare_spoken_text("It is ~5 km away, roughly ~$20.") == "It is about 5 km away, roughly about 20 dollars."
    assert "about" not in prepare_spoken_text("Saved to ~/notes/todo.md for later.")
