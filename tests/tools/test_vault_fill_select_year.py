"""Regression tests for the expiry-year select defect.

A card expiry reaches the vault as ``exp_year = "2031"`` (4 digits). A checkout's year
``<select>`` almost always offers 2-digit option values ("26".."45"). The fill matched options
literally, so ``"2031"`` found nothing, the control kept whatever default it had, and **no
error was raised anywhere**:

* the page ended up self-consistent (control "26", hidden companion "26") and simply wrong;
* the fill still reported ``{"filled": 1}``, so the caller saw success;
* the card reached the acquirer with a bad expiry and the bank declined it.

That combination cost four declined payments on a real booking before a human spotted it. These
tests pin both halves of the repair: the year-width-tolerant matching, and the honest reporting
that makes a silent miss impossible to overlook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.vault_login_classifier import build_fill_js, build_inspection_js  # noqa: E402

# --- a real <select> shape: 2-digit option values, defaulting to "26" --------------------

TWO_DIGIT_YEAR_PAGE = """<!doctype html><meta charset="utf-8"><title>t</title>
<select name="cc-exp-year" id="y">
  <option value="26">26</option><option value="27">27</option>
  <option value="31">31</option><option value="45">45</option>
</select>
<select name="cc-exp-month" id="m">
  <option value="10">10</option><option value="11">11</option>
</select>
<script>document.getElementById('y').value='26';</script>
"""

FOUR_DIGIT_YEAR_PAGE = """<!doctype html><meta charset="utf-8"><title>t</title>
<select name="cc-exp-year" id="y">
  <option value="2026">2026</option><option value="2031">2031</option>
</select>
<select name="cc-exp-month" id="m"><option value="10">10</option></select>
"""

# a select whose options are non-numeric text: the year fallback must never touch these
COUNTRY_PAGE = """<!doctype html><meta charset="utf-8"><title>t</title>
<select name="country" id="c">
  <option value="GB">United Kingdom</option>
  <option value="CM">Cameroon</option>
</select>
"""


def _page_html(html: str) -> str:
    return "data:text/html," + html.replace("#", "%23")


class TestFillScriptStringContract:
    """The script is also asserted on as text; these pin the new contract."""

    def test_reports_requested_and_skipped_alongside_filled(self):
        js = build_fill_js([{"index": 0, "token": "cc-exp-year", "value": "x"}],
                           expected_origin="https://example.com")
        assert "requested" in js and "skipped" in js
        assert "fills.length" in js

    def test_skip_reasons_are_named(self):
        js = build_fill_js([{"index": 0, "token": "cc-exp-year", "value": "x"}],
                           expected_origin="https://example.com")
        for reason in ("no_matching_option", "no_target", "write_failed", "empty_after_set"):
            assert reason in js, f"reason {reason!r} must be reported, not swallowed"

    def test_still_leaves_no_dom_marker(self):
        js = build_fill_js([{"index": 0, "token": "cc-exp-year", "value": "x"}],
                           expected_origin="https://example.com")
        assert js.index('removeAttribute("data-hermes-vault-slot")') > js.index("el.focus()")

    def test_secrets_never_appear(self):
        js = build_fill_js([{"index": 0, "token": "cc-number", "value": "4111111111111111"}],
                           expected_origin="https://example.com")
        # the value is in the payload by necessity, but nothing about it is echoed back
        assert "skipped.push" in js and "vaultSecret" not in js


@pytest.fixture(scope="module")
def browser():
    """A real Chromium, because the defect is JS behaviour, not a string."""
    playwright = pytest.importorskip("playwright.sync_api")
    launch = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
    for exe in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"):
        if Path(exe).exists():
            launch["executable_path"] = exe
            break
    with playwright.sync_playwright() as pw:
        try:
            br = pw.chromium.launch(**launch)
        except Exception as exc:                      # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        yield br
        br.close()


def _fill(page, token_value_pairs, nonce="n1"):
    """Inspect the page, then fill the named tokens. Returns (parsed_result, inspected_controls)."""
    inspected = json.loads(page.evaluate(build_inspection_js(nonce)))
    fills = []
    for token, value in token_value_pairs:
        ctl = next(c for c in inspected if token in (c.get("name") or ""))
        fills.append({"index": ctl["index"], "token": token, "value": value})
    res = page.evaluate(build_fill_js(fills, expected_origin="null", nonce=nonce))
    return (json.loads(res) if isinstance(res, str) else res), inspected


class TestTheDefectItself:
    """The exact production shape, end to end in a real browser."""

    def test_four_digit_vault_year_lands_on_a_two_digit_control(self, browser):
        page = browser.new_page()
        page.goto(_page_html(TWO_DIGIT_YEAR_PAGE), wait_until="domcontentloaded")
        res, _ = _fill(page, [("cc-exp-year", "2031"), ("cc-exp-month", "10")])
        held = page.evaluate("() => document.getElementById('y').value")
        page.close()
        # THE assertion: the control must express the card's real year, not its default
        assert held == "31", f"requested 2031, control holds {held!r}"
        assert held != "26", "the control kept its default -- this is the bug returning"
        assert res["filled"] == 2 and res["skipped"] == []

    def test_two_digit_vault_year_still_works(self, browser):
        page = browser.new_page()
        page.goto(_page_html(TWO_DIGIT_YEAR_PAGE), wait_until="domcontentloaded")
        res, _ = _fill(page, [("cc-exp-year", "31")])
        held = page.evaluate("() => document.getElementById('y').value")
        page.close()
        assert held == "31" and res["filled"] == 1

    def test_two_digit_year_lands_on_a_four_digit_control(self, browser):
        page = browser.new_page()
        page.goto(_page_html(FOUR_DIGIT_YEAR_PAGE), wait_until="domcontentloaded")
        res, _ = _fill(page, [("cc-exp-year", "31")])
        held = page.evaluate("() => document.getElementById('y').value")
        page.close()
        assert held == "2031" and res["filled"] == 1

    def test_an_unexpressible_year_is_reported_not_silently_defaulted(self, browser):
        page = browser.new_page()
        page.goto(_page_html(TWO_DIGIT_YEAR_PAGE), wait_until="domcontentloaded")
        # 2099 is not among 26/27/31/45
        res, _ = _fill(page, [("cc-exp-year", "2099")])
        held = page.evaluate("() => document.getElementById('y').value")
        page.close()
        assert res["filled"] == 0, "a fill that could not land must not count as written"
        assert res["requested"] == 1
        assert res["skipped"] and res["skipped"][0]["reason"] == "no_matching_option"
        assert res["skipped"][0]["token"] == "cc-exp-year"
        assert held == "26", "the control is expected to keep its default here"
        # the point: the DEFAULT is still there, but now the result says so

    def test_the_year_fallback_does_not_touch_text_selects(self, browser):
        page = browser.new_page()
        page.goto(_page_html(COUNTRY_PAGE), wait_until="domcontentloaded")
        # a numeric value must not hijack a text option list
        res, _ = _fill(page, [("country", "31")])
        held = page.evaluate("() => document.getElementById('c').value")
        page.close()
        assert held in ("GB", ""), f"country select was wrongly set to {held!r}"
        assert res["filled"] == 0
        assert res["skipped"][0]["reason"] == "no_matching_option"

    def test_a_genuine_text_match_still_works(self, browser):
        page = browser.new_page()
        page.goto(_page_html(COUNTRY_PAGE), wait_until="domcontentloaded")
        res, _ = _fill(page, [("country", "Cameroon")])
        held = page.evaluate("() => document.getElementById('c').value")
        page.close()
        assert held == "CM" and res["filled"] == 1

    def test_partial_fill_is_reported_field_by_field(self, browser):
        page = browser.new_page()
        page.goto(_page_html(TWO_DIGIT_YEAR_PAGE), wait_until="domcontentloaded")
        res, _ = _fill(page, [("cc-exp-year", "2099"),   # cannot land
                                    ("cc-exp-month", "10")])   # lands fine
        page.close()
        assert res["filled"] == 1 and res["requested"] == 2
        assert [s["token"] for s in res["skipped"]] == ["cc-exp-year"]
