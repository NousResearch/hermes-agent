"""Playwright UI tests for the Models page layout.

New structure:
- Single outer Tab level with 3 tabs:
  - Main Model: Main Model card + Stats card + Fallback Chain card (below)
  - Auxiliary Tasks: Inline panel (no modal)
  - Used Models: Analytics grid with period controls
"""
from __future__ import annotations

import pytest
from playwright.async_api import Page, expect

from tests.ui.conftest import MODELS_PAGE_URL


async def _go_to_tab(page: Page, tab_name: str) -> None:
    """Navigate to a specific tab."""
    await page.goto(MODELS_PAGE_URL)
    await page.wait_for_timeout(2000)  # Wait for initial render
    tab_selectors = {
        "Main Model": "[data-testid='models-settings-main-tab']",
        "Auxiliary Tasks": "[data-testid='models-settings-aux-tab']",
        "Used Models": "[data-testid='models-used-models-tab']",
    }
    selector = tab_selectors.get(tab_name)
    if selector:
        await page.locator(selector).click()
        await page.wait_for_timeout(500)


@pytest.mark.asyncio
async def test_settings_has_no_outer_tab(page: Page):
    """There should be no outer 'Settings' tab - all tabs are at the same level."""
    await page.goto(MODELS_PAGE_URL)
    await page.wait_for_timeout(2000)
    # The old 'settings' outer tab should NOT exist
    settings_outer = page.locator("[data-testid='models-settings-tab']")
    count = await settings_outer.count()
    assert count == 0, "There should be no outer 'Settings' tab"


@pytest.mark.asyncio
async def test_all_tabs_visible(page: Page):
    """All 3 tabs should be visible at the top."""
    await page.goto(MODELS_PAGE_URL)
    await page.wait_for_timeout(2000)
    await expect(page.locator("[data-testid='models-settings-main-tab']")).to_be_visible()
    await expect(page.locator("[data-testid='models-settings-aux-tab']")).to_be_visible()
    await expect(page.locator("[data-testid='models-used-models-tab']")).to_be_visible()


@pytest.mark.asyncio
async def test_fallback_chain_tab_does_not_exist(page: Page):
    """There should be no standalone Fallback Chain tab."""
    await page.goto(MODELS_PAGE_URL)
    await page.wait_for_timeout(2000)
    fallback_tab = page.locator("[data-testid='models-settings-fallback-tab']")
    count = await fallback_tab.count()
    assert count == 0, "There should be no standalone Fallback Chain tab"


@pytest.mark.asyncio
async def test_main_model_tab_has_card(page: Page):
    """Main Model tab should show the Main Model card."""
    await _go_to_tab(page, "Main Model")
    await expect(page.locator("[data-testid='main-model-card']")).to_be_visible()


@pytest.mark.asyncio
async def test_main_model_tab_has_fallback_chain(page: Page):
    """Main Model tab should also show the Fallback Chain card below the main model."""
    await _go_to_tab(page, "Main Model")
    await expect(page.locator("[data-testid='fallback-chain-card']")).to_be_visible()


@pytest.mark.asyncio
async def test_main_model_tab_has_stats(page: Page):
    """Main Model tab should show the stats card."""
    await _go_to_tab(page, "Main Model")
    stats = page.locator("[data-testid='settings-tab-panel'] > div > div > div")
    count = await stats.count()
    assert count >= 2, "Should have at least 2 cards (main model + stats)"


@pytest.mark.asyncio
async def test_fallback_chain_has_add_button(page: Page):
    """Fallback Chain card (inside Main Model tab) should have Add button."""
    await _go_to_tab(page, "Main Model")
    add_btn = page.locator("[data-testid='fallback-chain-card'] [data-testid='fallback-add-button']")
    await expect(add_btn).to_be_visible()


@pytest.mark.asyncio
async def test_fallback_chain_has_save_button(page: Page):
    """Fallback Chain card should have Save button."""
    await _go_to_tab(page, "Main Model")
    save_btn = page.locator("[data-testid='fallback-chain-card'] [data-testid='fallback-save-button']")
    await expect(save_btn).to_be_visible()


@pytest.mark.asyncio
async def test_fallback_chain_add_opens_picker(page: Page):
    """Add button in Fallback Chain card should open the model picker."""
    await _go_to_tab(page, "Main Model")
    add_btn = page.locator("[data-testid='fallback-add-button']")
    await add_btn.click()
    await page.wait_for_timeout(500)
    picker = page.locator("[data-testid='model-picker-dialog']")
    visible = await picker.count() > 0
    if not visible:
        visible = await page.locator("text=Add Fallback Provider").count() > 0
    assert visible, "Model picker dialog should be visible after clicking Add"


@pytest.mark.asyncio
async def test_auxiliary_tab_has_inline_panel(page: Page):
    """Auxiliary Tasks tab should show the inline configure panel."""
    await _go_to_tab(page, "Auxiliary Tasks")
    panel = page.locator("[data-testid='auxiliary-tasks-tab-panel']")
    await expect(panel).to_be_visible()
    task_items = page.locator("[data-testid='auxiliary-task-item']")
    count = await task_items.count()
    assert count > 0, "Should have at least one auxiliary task item"


@pytest.mark.asyncio
async def test_auxiliary_tab_no_configure_button(page: Page):
    """Auxiliary Tasks tab should NOT have a 'Configure' button."""
    await _go_to_tab(page, "Auxiliary Tasks")
    await expect(page.get_by_role("button", name="Configure", exact=True)).to_have_count(0)


@pytest.mark.asyncio
async def test_auxiliary_tab_task_items_have_labels(page: Page):
    """Auxiliary Tasks tab should show task items with labels."""
    await _go_to_tab(page, "Auxiliary Tasks")
    task_items = page.locator("[data-testid='auxiliary-task-item']")
    count = await task_items.count()
    assert count >= 1, "Should have at least one task item"
    first_item = await task_items.first.inner_text()
    assert len(first_item) > 0, "Task item should have text content"


@pytest.mark.asyncio
async def test_used_models_tab_has_period_controls(page: Page):
    """Used Models tab should have period controls (7d, 30d, 90d)."""
    await _go_to_tab(page, "Used Models")
    await expect(page.locator("[data-testid='used-models-period-7']")).to_be_visible()
    await expect(page.locator("[data-testid='used-models-period-30']")).to_be_visible()
    await expect(page.locator("[data-testid='used-models-period-90']")).to_be_visible()
    await expect(page.locator("[data-testid='used-models-refresh-button']")).to_be_visible()


@pytest.mark.asyncio
async def test_used_models_tab_renders_content(page: Page):
    """Used Models tab should render either cards or empty state."""
    await _go_to_tab(page, "Used Models")
    cards = page.locator("[data-testid='used-model-card']")
    empty_state = page.locator("[data-testid='used-models-empty-state']")
    cards_count = await cards.count()
    empty_count = await empty_state.count()
    assert cards_count > 0 or empty_count > 0, "Should show model cards or empty state"
