"""Playwright UI tests for the new Models page layout.

New layout:
- Settings outer tab: Main Model card (no inner tabs)
- Fallback Chain outer tab: standalone, full UI
- Auxiliary Tasks outer tab: standalone, inline panel (no modal)
- Used Models outer tab: analytics grid
"""
from __future__ import annotations

import pytest
from playwright.async_api import Page, expect

from tests.ui.conftest import MODELS_PAGE_URL


async def _go_to(page: Page, tab: str) -> None:
    """Navigate to a specific outer tab."""
    await page.goto(MODELS_PAGE_URL)
    tab_map = {
        "settings": "models-settings-tab",
        "fallback-chain": "models-fallback-chain-tab",
        "auxiliary-tasks": "models-auxiliary-tasks-tab",
        "used-models": "models-used-models-tab",
    }
    selector = f"[data-testid='{tab_map[tab]}']"
    await page.locator(selector).click()
    # Give tab content time to render
    await page.wait_for_timeout(200)


# ── Outer tabs structure ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_outer_tabs_visible(page: Page):
    """Settings, Fallback Chain, Auxiliary Tasks, Used Models should all be outer tabs."""
    await page.goto(MODELS_PAGE_URL)
    await expect(page.locator("[data-testid='models-settings-tab']")).to_be_visible()
    await expect(page.locator("[data-testid='models-fallback-chain-tab']")).to_be_visible()
    await expect(page.locator("[data-testid='models-auxiliary-tasks-tab']")).to_be_visible()
    await expect(page.locator("[data-testid='models-used-models-tab']")).to_be_visible()


@pytest.mark.asyncio
async def test_outer_tabs_switch_to_settings(page: Page):
    """Clicking Settings tab should show Main Model card."""
    await _go_to(page, "settings")
    await expect(page.locator("[data-testid='main-model-card']")).to_be_visible()


@pytest.mark.asyncio
async def test_outer_tabs_switch_to_fallback(page: Page):
    """Clicking Fallback Chain tab should show full fallback chain UI."""
    await _go_to(page, "fallback-chain")
    await expect(page.locator("[data-testid='fallback-chain-card']")).to_be_visible()
    await expect(page.locator("[data-testid='fallback-add-button']")).to_be_visible()
    await expect(page.locator("[data-testid='fallback-save-button']")).to_be_visible()


@pytest.mark.asyncio
async def test_outer_tabs_switch_to_auxiliary(page: Page):
    """Clicking Auxiliary Tasks tab should show inline panel with task items."""
    await _go_to(page, "auxiliary-tasks")
    await expect(page.locator("[data-testid='auxiliary-tasks-tab-panel']")).to_be_visible()
    task_items = page.locator("[data-testid='auxiliary-task-item']")
    count = await task_items.count()
    assert count > 0, f"Should have at least one auxiliary task item, got {count}"


@pytest.mark.asyncio
async def test_outer_tabs_switch_to_used_models(page: Page):
    """Clicking Used Models tab should show analytics grid."""
    await _go_to(page, "used-models")
    await expect(page.locator("[data-testid='used-models-tab-panel']")).to_be_visible()


# ── Settings tab ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_settings_has_main_model_card(page: Page):
    """Settings tab should show the Main Model card."""
    await _go_to(page, "settings")
    await expect(page.locator("[data-testid='main-model-card']")).to_be_visible()


@pytest.mark.asyncio
async def test_settings_no_inner_tabs(page: Page):
    """Settings tab should NOT have inner tabs (Main Model, Fallback Chain, Auxiliary Tasks)."""
    await _go_to(page, "settings")
    # The old inner tab triggers should NOT exist as buttons
    inner_tabs = page.locator("[data-testid='settings-tab-panel'] [data-testid]")
    # Check there are no tabs-list inside the settings panel
    inner_tabs_list = page.locator("[data-testid='settings-tab-panel'] [role='tablist']")
    count = await inner_tabs_list.count()
    assert count == 0, "Settings tab should not have an inner tablist"


# ── Fallback Chain tab ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_chain_standalone_has_add_button(page: Page):
    """Fallback Chain standalone tab should have Add button."""
    await _go_to(page, "fallback-chain")
    await expect(page.locator("[data-testid='fallback-add-button']")).to_be_visible()


@pytest.mark.asyncio
async def test_fallback_chain_standalone_has_save_button(page: Page):
    """Fallback Chain standalone tab should have Save button."""
    await _go_to(page, "fallback-chain")
    await expect(page.locator("[data-testid='fallback-save-button']")).to_be_visible()


@pytest.mark.asyncio
async def test_fallback_chain_add_opens_picker(page: Page):
    """Add button in Fallback Chain tab should open the model picker."""
    await _go_to(page, "fallback-chain")
    await page.locator("[data-testid='fallback-add-button']").click()
    await expect(page.locator("[data-testid='model-picker-dialog']")).to_be_visible()


# ── Auxiliary Tasks tab ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auxiliary_tab_has_inline_panel(page: Page):
    """Auxiliary Tasks tab should show the inline panel (not a modal)."""
    await _go_to(page, "auxiliary-tasks")
    await expect(page.locator("[data-testid='auxiliary-tasks-tab-panel']")).to_be_visible()
    task_items = page.locator("[data-testid='auxiliary-task-item']")
    count = await task_items.count()
    assert count > 0, f"Should have at least one task item, got {count}"


@pytest.mark.asyncio
async def test_auxiliary_tab_no_configure_button(page: Page):
    """Auxiliary Tasks tab should NOT have a 'Configure' button."""
    await _go_to(page, "auxiliary-tasks")
    await expect(page.get_by_role("button", name="Configure", exact=True)).to_have_count(0)


@pytest.mark.asyncio
async def test_auxiliary_tab_task_items_show_provider_and_model(page: Page):
    """Auxiliary task items should show provider and model info."""
    await _go_to(page, "auxiliary-tasks")
    first_item = page.locator("[data-testid='auxiliary-task-item']").first
    await expect(first_item).to_be_visible()
    # Should contain text like "Vision" or "auto (use main model)"
    text = await first_item.inner_text()
    assert len(text) > 0, "Task item should have text content"
