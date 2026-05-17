"""Pytest fixtures for Playwright UI tests.

All tests import fixtures from here via conftest.py — no duplication.
"""
from __future__ import annotations

import pytest
from playwright.async_api import async_playwright, Page, Browser

# The dashboard runs on 127.0.0.1:9119.
DASHBOARD_URL = "http://127.0.0.1:9119"
MODELS_PAGE_URL = f"{DASHBOARD_URL}/models"


@pytest.fixture
async def browser():
    """Launch a headless Chromium browser for the test suite."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser: Browser) -> Page:
    """Create a new page with the session token injected."""
    context = await browser.new_context()
    page = await context.new_page()

    # Fetch the session token from the running dashboard
    await page.goto(DASHBOARD_URL)
    token = await page.evaluate(
        "() => { const m = document.body.innerHTML.match(/__HERMES_SESSION_TOKEN__=\\\"([^\\\"]+)\\\"/); return m ? m[1] : null; }",
    )

    if token:
        await page.add_init_script(
            f"window.__HERMES_SESSION_TOKEN__ = \"{token}\";",
        )

    yield page
    await context.close()
