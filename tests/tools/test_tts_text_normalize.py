import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
# Only import symbols that already exist on main here, so the behavioural
# reasoning-leak tests below collect and fail genuinely without the fix
# (rather than erroring on a not-yet-added import).
from tools.tts_text_normalize import prepare_spoken_text, strip_nonspoken_blocks


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


# Reasoning must never be spoken (#34213) — for EVERY tag variant the canonical
# scrubber handles, not just <think>. Non-<think> models (Gemini/Gemma/GLM via
# the OpenAI-compatible path) emit <thinking>/<reasoning>/<thought>; before the
# fix strip_nonspoken_blocks passed those straight through to the speech engine.
@pytest.mark.parametrize(
    "tag", ["think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD"]
)
def test_reasoning_blocks_never_reach_speech(tag):
    raw = f"The answer is 42.<{tag}>hidden private reasoning</{tag}> Done."
    spoken = prepare_spoken_text(raw)
    assert "hidden" not in spoken and "reasoning" not in spoken
    assert "The answer is 42." in spoken
    assert "Done." in spoken


def test_reasoning_block_stripped_case_insensitively():
    assert "secret" not in prepare_spoken_text("Hi.<THINKING>secret</THINKING>")
    assert "secret" not in prepare_spoken_text("Hi.<Reasoning>secret</Reasoning>")


@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "thought"])
def test_unterminated_reasoning_block_is_dropped(tag):
    # A streaming cut-off leaves an open tag with no close — still not spoken.
    assert strip_nonspoken_blocks(f"visible <{tag}>runaway reasoning to the end").strip() == "visible"


def test_has_unclosed_reasoning_tag_probe():
    from tools.tts_text_normalize import has_unclosed_reasoning_tag

    assert has_unclosed_reasoning_tag("text <reasoning>open but not closed")
    assert has_unclosed_reasoning_tag("<thinking>partial")
    assert not has_unclosed_reasoning_tag("<think>closed</think> rest")
    assert not has_unclosed_reasoning_tag("plain text, no tags")


def test_tag_set_is_sourced_from_the_canonical_scrubber():
    # One source of truth: the TTS suppression list must not drift behind the
    # display scrubber. Importing the same object guarantees a new variant
    # added to think_scrubber is covered here without a second edit.
    from agent.think_scrubber import REASONING_TAG_NAMES as canonical
    from tools.tts_text_normalize import REASONING_TAG_NAMES as tts

    assert tts is canonical
