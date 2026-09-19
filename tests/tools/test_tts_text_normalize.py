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


def test_prepare_spoken_text_closes_trailing_colons():
    # "the regex list:" + a now-removed raw token would leave the voice hanging
    # on an open colon-pause (the "aaaa" stutter). Close it with a period.
    spoken = prepare_spoken_text("Here is the list:\nMEDIA:/tmp/x.py\nMore text")
    assert "list:" not in spoken
    assert "list." in spoken


def test_prepare_spoken_text_expands_times_sign():
    spoken = prepare_spoken_text("Runs on 2× RTX 5090 cards.")
    assert "×" not in spoken
    assert "times" in spoken


def test_prepare_spoken_text_tames_em_dashes():
    # Real repro: "if you want to peek — the app password lives..." read as a
    # growl. Lowercase after the dash is a comma pause; an uppercase letter
    # after the dash opens a new sentence. A digit-to-digit range becomes
    # "to" ("pages 5–10" reads "pages 5 to 10"), and a dash next to a digit
    # on one side ("Step 1 — open") still gets tamed, not left raw.
    spoken = prepare_spoken_text(
        "The config file's below if you want to peek — the app password lives in your keychain."
    )
    assert "—" not in spoken
    assert "peek, the app password" in spoken
    assert "peek. Done." in prepare_spoken_text("You can peek — Done.")
    assert "red — green — blue" not in prepare_spoken_text("Mix red — green — blue.")
    assert "5 to 10" in prepare_spoken_text("See pages 5–10.")
    assert "Step 1, open the file" in prepare_spoken_text("Step 1 — open the file.")


def test_prepare_spoken_text_expands_suffix_euro():
    # PT/ES writes the euro sign after the amount ("1.499,90 €"); a bare
    # sign ("prices in €") also resolves to a word.
    assert "1.499,90 euros" in prepare_spoken_text("Costs 1.499,90 € per unit.")
    assert "prices in euros" in prepare_spoken_text("prices in € are stable")


def test_prepare_spoken_text_closes_colon_on_single_line():
    # Multi-line text gets colons closed per line; single-line text reaches the
    # end-of-text rule instead.
    assert prepare_spoken_text("Here is the list:") == "Here is the list."
    # ...but a digit-preceded colon is a ratio and must stay intact.
    assert prepare_spoken_text("Final score 3:2") == "Final score 3:2"


def test_prepare_spoken_text_tames_bare_paths():
    # A bare filesystem path in prose (no MEDIA: marker) loops the voice the
    # same way ("aaaa" at "~/.config/himalaya/config.toml"). A path is a
    # screen address, not speech: it becomes "the path".
    assert "the path" in prepare_spoken_text("check the config at ~/.config/himalaya/config.toml")
    assert "toml" not in prepare_spoken_text("check the config at ~/.config/himalaya/config.toml")
    assert "the path" in prepare_spoken_text("read /etc/hosts for the mapping")
    assert "the path" in prepare_spoken_text("open src/lib/app.ts to edit")
    assert "the path" in prepare_spoken_text("see /Users/me/Documents/file.xlsx now")
    # Not paths: slashed words, N/A, dates, decimal fractions, rates, and
    # tilde approximations all stay intact.
    assert "and/or" in prepare_spoken_text("pick A and/or B")
    assert "N/A" in prepare_spoken_text("status is N/A")
    assert "2026/06/02" in prepare_spoken_text("due 2026/06/02")
    assert "1.5/2.5" in prepare_spoken_text("ratio 1.5/2.5")
    assert "5 per month" in prepare_spoken_text("pay 5/month")
    assert "about 100" in prepare_spoken_text("~100 users")
