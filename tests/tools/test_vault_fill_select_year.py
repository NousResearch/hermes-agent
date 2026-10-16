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
    """A real browser when Playwright + Chromium are present, else a Node DOM harness.

    This NEVER skips. A Playwright-only fixture silently skips wherever Playwright is absent
    -- including upstream's unit-test lane -- and a skip among passes reads as a clean run.
    Node is a hard dependency of this repo, so the Node path is always exercisable; the
    Chromium path is used when available for maximum fidelity. Both run the same assertions,
    including the one that matters: the control must end up holding the card's real year.
    """
    from tests.tools._vault_js_harness import has_chromium, node_available

    chromium = has_chromium()
    if chromium:
        try:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            br = pw.chromium.launch(headless=True, executable_path=chromium,
                                    args=["--no-sandbox", "--disable-dev-shm-usage"])
            yield _PlaywrightDriver(br)
            br.close()
            pw.stop()
            return
        except Exception:
            pass  # fall through to Node rather than skipping

    if not node_available():
        pytest.fail("neither Chromium+Playwright nor node is available; the vault fill JS "
                    "cannot be verified. Do not treat this as a pass.")
    yield _NodeDriver()


def _maybe_json(value):
    """Playwright returns an object when the JS returns one and a string when it stringifies.

    Playwright's ``evaluate`` already deserialises a returned object, so calling ``json.loads``
    on it raises ``TypeError: not list``. Only a string needs parsing.
    """
    return json.loads(value) if isinstance(value, str) else value


class _PlaywrightDriver:
    """Runs inspection+fill in a real Chromium."""

    name = "chromium"

    def __init__(self, br):
        self._br = br

    def inspect(self, html, inspection_js):
        pg = self._br.new_page()
        pg.goto("data:text/html," + html.replace("#", "%23"), wait_until="domcontentloaded")
        self._pg = pg
        return _maybe_json(pg.evaluate(inspection_js))

    def fill(self, fill_js):
        pg = self._pg
        out = _maybe_json(pg.evaluate(fill_js))
        held_raw = pg.evaluate(
            "() => Array.from(document.querySelectorAll('select'))"
            ".map(s => ({name: s.name || s.id, value: s.value}))")
        held = {c["name"]: c["value"] for c in _maybe_json(held_raw)}
        pg.close()
        return {"result": out, "held": held}


class _NodeDriver:
    """Runs inspection+fill against a minimal DOM in Node. Never skips."""

    name = "node"

    def __init__(self):
        self._html = None

    def inspect(self, html, inspection_js):
        from tests.tools._vault_js_harness import run_in_node
        self._html = html
        return run_in_node(html, inspection_js)["inspection"]

    def fill(self, fill_js):
        from tests.tools._vault_js_harness import run_in_node
        # same html -> identical DOM -> the inspection's index stamps still resolve
        return run_in_node(self._html, "null", fill_js)


def _fill(driver, html, token_value_pairs, nonce="n1"):
    """Inspect ``html``, then fill the named tokens. Returns the driver's fill result.

    ``driver`` is the object yielded by the ``browser`` fixture; ``driver.name`` says which
    engine actually ran ('chromium' or 'node').
    """
    from agent.vault_login_classifier import build_fill_js, build_inspection_js

    inspected = driver.inspect(html, build_inspection_js(nonce))
    fills = []
    for token, value in token_value_pairs:
        ctl = next(c for c in inspected if token in (c.get("name") or ""))
        fills.append({"index": ctl["index"], "token": token, "value": value})
    return driver.fill(build_fill_js(fills, expected_origin="null", nonce=nonce))


class TestTheDefectItself:
    """The exact production shape, end to end. Runs in Chromium when available, else Node.

    These assertions are the point of the PR: they fail against the pre-fix code. The engine
    used is reported so a reviewer can see the tests actually executed instead of being skipped.
    """

    def test_the_engine_is_reported_and_never_a_silent_skip(self, browser):
        # a skip among passes is how this defect hid; make the engine visible
        assert browser.name in ("chromium", "node"), browser.name

    def test_four_digit_vault_year_lands_on_a_two_digit_control(self, browser):
        r = _fill(browser, TWO_DIGIT_YEAR_PAGE,
                  [("cc-exp-year", "2031"), ("cc-exp-month", "10")])
        held = r["held"].get("cc-exp-year")
        # THE assertion: the control must express the card's real year, not its default
        assert held == "31", f"requested 2031, control holds {held!r}"
        assert held != "26", "the control kept its default -- this is the bug returning"
        assert r["result"]["filled"] == 2 and r["result"]["skipped"] == []

    def test_two_digit_vault_year_still_works(self, browser):
        r = _fill(browser, TWO_DIGIT_YEAR_PAGE, [("cc-exp-year", "31")])
        assert r["held"].get("cc-exp-year") == "31" and r["result"]["filled"] == 1

    def test_two_digit_year_lands_on_a_four_digit_control(self, browser):
        r = _fill(browser, FOUR_DIGIT_YEAR_PAGE, [("cc-exp-year", "31")])
        assert r["held"].get("cc-exp-year") == "2031" and r["result"]["filled"] == 1

    def test_an_unexpressible_year_is_reported_not_silently_defaulted(self, browser):
        # 2099 is not among 26/27/31/45
        r = _fill(browser, TWO_DIGIT_YEAR_PAGE, [("cc-exp-year", "2099")])
        res = r["result"]
        assert res["filled"] == 0, "a fill that could not land must not count as written"
        assert res["requested"] == 1
        assert res["skipped"] and res["skipped"][0]["reason"] == "no_matching_option"
        assert res["skipped"][0]["token"] == "cc-exp-year"
        assert r["held"].get("cc-exp-year") == "26", "the control keeps its default here"
        # the point: the DEFAULT is still there, but now the result says so

    def test_the_year_fallback_does_not_touch_text_selects(self, browser):
        # a numeric value must not hijack a text option list
        r = _fill(browser, COUNTRY_PAGE, [("country", "31")])
        assert r["held"].get("country") in ("GB", ""), \
            f"country select was wrongly set to {r['held'].get('country')!r}"
        assert r["result"]["filled"] == 0
        assert r["result"]["skipped"][0]["reason"] == "no_matching_option"

    def test_a_genuine_text_match_still_works(self, browser):
        r = _fill(browser, COUNTRY_PAGE, [("country", "Cameroon")])
        assert r["held"].get("country") == "CM" and r["result"]["filled"] == 1

    def test_partial_fill_is_reported_field_by_field(self, browser):
        r = _fill(browser, TWO_DIGIT_YEAR_PAGE,
                  [("cc-exp-year", "2099"),      # cannot land
                   ("cc-exp-month", "10")])      # lands fine
        res = r["result"]
        assert res["filled"] == 1 and res["requested"] == 2
        assert [s["token"] for s in res["skipped"]] == ["cc-exp-year"]
