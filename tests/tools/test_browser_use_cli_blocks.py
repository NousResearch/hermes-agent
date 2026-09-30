"""browser_exec flags bot-protection walls and carries the configured next-step hint."""

import pytest

from tools.browser_use_cli_blocks import DEFAULT_HINT, blocked_page_hint, detect_block

# Real outputs captured from browser_exec runs against these sites (trimmed).
OPENTABLE_EDGE_DROP = (
    "{'url': 'chrome-error://chromewebdata/', 'title': '\U0001f434 www.opentable.com', 'w': 1280}\n"
    "This site can\u2019t be reached\n\nThe webpage at https://www.opentable.com/r/x might be "
    "temporarily down or it may have moved permanently to a new web address.\n\nERR_HTTP2_PROTOCOL_ERROR\n"
)
STREETEASY_PX = (
    "{'url': 'https://streeteasy.com/for-rent/west-village', 'title': '\U0001f434 Access to this page "
    "has been denied', 'w': 1280}\nPress & Hold to confirm you are a human (and not a bot)."
)
AKAMAI = "Access Denied\nYou don't have permission to access \"http://www.opentable.com/\" on this server.\nReference #18.4c1d2417.1759000000.8a2b3c"
CLOUDFLARE = "{'url': 'https://example.com/', 'title': 'Just a moment...'}\nVerify you are human by completing the action below."


@pytest.mark.parametrize(
    ("output", "vendor"),
    [
        (STREETEASY_PX, "perimeterx"),
        (AKAMAI, "akamai"),
        (CLOUDFLARE, "cloudflare"),
        ("Access Denied\nERR_HTTP2_PROTOCOL_ERROR", "akamai"),
        ("ERR_HTTP2_PROTOCOL_ERROR\nAccess Denied", "akamai"),
        ("Reference #18.abc\nERR_HTTP2_PROTOCOL_ERROR", "akamai"),
        ("ERR_HTTP2_PROTOCOL_ERROR\nReference #18.abc", "akamai"),
        ("PerimeterX\nPress & Hold to continue", "perimeterx"),
        ("Press & Hold to continue\nPerimeterX", "perimeterx"),
        ("Human verification\nPress & Hold", "perimeterx"),
        ("Press & Hold\nHuman verification", "perimeterx"),
    ],
)
def test_known_walls_are_detected(output, vendor):
    assert detect_block(output) == vendor


@pytest.mark.parametrize(
    "output",
    [
        "",
        OPENTABLE_EDGE_DROP,
        "{'url': 'https://support.apple.com/guide/iphone', 'title': 'iPhone guide'}\n"
        "Press & Hold on an image to open the contextual menu",
        "{'url': 'https://developer.android.com/touch', 'title': 'Touch gestures'}\n"
        "Press & Hold to select an item",
        "nginx error log: upstream connection failed: ERR_HTTP2_PROTOCOL_ERROR",
        "If you see ERR_HTTP2_PROTOCOL_ERROR the peer sent an invalid frame",
        "{'title': 'How to Fix Access Denied Errors in Nginx'}\nCheck file permissions.",
        "Access Denied" + "x" * 401 + "ERR_HTTP2_PROTOCOL_ERROR",
        "ERR_HTTP2_PROTOCOL_ERROR" + "x" * 401 + "Reference #18.abc",
        "PerimeterX" + "x" * 401 + "Press & Hold",
        "Press & Hold" + "x" * 401 + "human verification",
        "{'url': 'https://news.ycombinator.com/', 'title': 'Hacker News'}\n1. Show HN: ...",
        # An article ABOUT bot walls must not trip it.
        "{'url': 'https://blog.example/x', 'title': 'Why CAPTCHAs fail'}\nMany sites show an access denied page to bots.",
        "{'url': 'https://www.opentable.com/r/gramercy-tavern-new-york', 'title': 'Gramercy Tavern'}\n6:45 PM 7:00 PM",
    ],
)
def test_normal_pages_are_not_flagged(output):
    assert detect_block(output) is None


def test_hint_defaults_overrides_and_disables():
    assert blocked_page_hint({}) == DEFAULT_HINT
    assert blocked_page_hint({"blocked_page_hint": None}) == DEFAULT_HINT
    assert blocked_page_hint({"blocked_page_hint": "Use Aside."}) == "Use Aside."
    assert blocked_page_hint({"blocked_page_hint": ""}) == ""


def test_config_yaml_hint_reaches_browser_exec_reader(tmp_path, monkeypatch):
    """browser.blocked_page_hint set in config.yaml is what browser_exec's own config reader returns."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("browser:\n  blocked_page_hint: 'Retry in Aside.'\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from tools import browser_use_cli

    assert blocked_page_hint(browser_use_cli._read_browser_cfg()) == "Retry in Aside."
