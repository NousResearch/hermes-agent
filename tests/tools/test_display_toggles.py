"""The desktop's Appearance switches, as the AGENT experiences them.

Each of these toggles decides whether a tool is in the model's schema at all,
so the contract under test is end to end: a real ``config.yaml`` in a temp
``HERMES_HOME``, read through the real config loader, answering a real
``check_fn``. Mocking the loader here would test nothing that matters — the
whole failure this guards against was a value that never reached the config.
"""

import textwrap

import pytest

from hermes_constants import get_hermes_home
from tools import desktop_ui
from tools.react_to_message_tool import check_react_requirements
from tools.tip_tool import check_tips_enabled
from tools.tour_tool import check_tours_enabled


@pytest.fixture
def display_config():
    """Write a ``display:`` section into this test's HERMES_HOME."""

    def _write(**flags: bool) -> None:
        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        body = "".join(f"  {name}: {str(on).lower()}\n" for name, on in flags.items())
        (home / "config.yaml").write_text(textwrap.dedent("display:\n") + body)

    return _write


def test_a_missing_setting_leaves_the_feature_at_its_default():
    """Absence is not an answer: an opt-out feature stays on, opt-in stays off."""
    assert desktop_ui.user_enabled("in_app_tips", default=True) is True
    assert desktop_ui.user_enabled("message_reactions", default=False) is False


def test_the_users_answer_wins_in_both_directions(display_config):
    display_config(in_app_tips=False, message_reactions=True)

    assert desktop_ui.user_enabled("in_app_tips", default=True) is False
    assert desktop_ui.user_enabled("message_reactions", default=False) is True


@pytest.mark.parametrize(
    ("setting", "check", "default_on"),
    [
        ("in_app_tips", check_tips_enabled, True),
        ("in_app_tours", check_tours_enabled, True),
        ("message_reactions", check_react_requirements, False),
    ],
)
def test_switching_a_feature_off_withdraws_its_tool(display_config, setting, check, default_on):
    """The point of the mirror: off means the model is never told it exists."""
    assert check() is default_on

    display_config(**{setting: False})
    assert check() is False

    display_config(**{setting: True})
    assert check() is True
