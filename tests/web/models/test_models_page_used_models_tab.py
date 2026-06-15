"""Playwright UI tests for the Models page Used Models tab."""
from __future__ import annotations

import pytest
from playwright.async_api import Page, expect

from tests.web.models.conftest import MODELS_PAGE_URL


pytestmark = pytest.mark.xdist_group("web_models")

async def _go_to_used_models(page: Page) -> None:
    await page.goto(MODELS_PAGE_URL)
    await page.locator("[data-testid='models-settings-used-tab']").click()
    await expect(page.locator("[data-testid='used-models-tab-panel']")).to_be_visible()


@pytest.mark.asyncio
async def test_used_models_tab_renders_analytics_surface(page: Page):
    """The Used Models tab should render period controls and model cards or an empty state."""
    await _go_to_used_models(page)

    await expect(page.locator("[data-testid='used-models-period-7']")).to_be_visible()
    await expect(page.locator("[data-testid='used-models-period-30']")).to_be_visible()
    await expect(page.locator("[data-testid='used-models-period-90']")).to_be_visible()
    await expect(page.locator("[data-testid='used-models-refresh-button']")).to_be_visible()

    cards = page.locator("[data-testid='used-model-card']")
    empty_state = page.locator("[data-testid='used-models-empty-state']")
    await expect(cards.first.or_(empty_state)).to_be_visible(timeout=10000)


@pytest.mark.asyncio
async def test_used_models_tab_hides_settings_inner_tabs(page: Page):
    """Switching to Used Models should not show Settings inner tab triggers."""
    await _go_to_used_models(page)

    # The Settings inner tab list (role='tablist' inside settings panel) should be gone
    inner_tablist = page.locator("[data-testid='settings-tab-panel'] [role='tablist']")
    count = await inner_tablist.count()
    assert count == 0, "Settings inner tablist should not exist"


@pytest.mark.asyncio
async def test_used_models_cards_expose_use_as_menu(page: Page):
    """Model cards in Used Models should keep the Use as assignment menu working."""
    await _go_to_used_models(page)

    cards = page.locator("[data-testid='used-model-card']")
    if await cards.count() == 0:
        await expect(page.locator("[data-testid='used-models-empty-state']")).to_be_visible()
        return

    await cards.first.locator("[data-testid='used-model-use-as-button']").click()
    await expect(page.locator("[data-testid='used-model-use-as-menu']")).to_be_visible()
    await expect(page.get_by_role("button", name="Main model", exact=True)).to_be_visible()
    await expect(page.get_by_role("button", name="All auxiliary tasks", exact=True)).to_be_visible()
