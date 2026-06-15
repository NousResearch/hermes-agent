"""Playwright UI tests for the Models page layout.

New structure:
- Single outer Tab level with 3 tabs:
  - Main Model: Main Model card + Fallback Chain card
  - Auxiliary Tasks: Inline panel (no modal)
  - Used Models: Stats card + analytics grid with period controls
"""
from __future__ import annotations

import pytest
from playwright.async_api import Page, expect

from tests.web.models.conftest import MODELS_PAGE_URL


pytestmark = pytest.mark.xdist_group("web_models")

async def _go_to_tab(page: Page, tab_name: str) -> None:
    """Navigate to a specific tab."""
    await page.goto(MODELS_PAGE_URL)
    await expect(page.locator("[data-testid='models-settings-main-tab']")).to_be_visible()
    tab_selectors = {
        "Main Model": "[data-testid='models-settings-main-tab']",
        "Auxiliary Tasks": "[data-testid='models-settings-aux-tab']",
        "Used Models": "[data-testid='models-settings-used-tab']",
    }
    # Content landmark that appears once the tab is active
    tab_content_selectors = {
        "Main Model": "[data-testid='main-model-card']",
        "Auxiliary Tasks": "[data-testid='auxiliary-tasks-tab-panel']",
        "Used Models": "[data-testid='used-models-period-7']",
    }
    selector = tab_selectors.get(tab_name)
    content_selector = tab_content_selectors.get(tab_name)
    if selector:
        await page.locator(selector).click()
        if content_selector:
            await expect(page.locator(content_selector)).to_be_visible()


@pytest.mark.asyncio
async def test_settings_has_no_outer_tab(page: Page):
    """There should be no outer 'Settings' tab - all tabs are at the same level."""
    await page.goto(MODELS_PAGE_URL)
    await expect(page.locator("[data-testid='models-settings-main-tab']")).to_be_visible()
    # The old 'settings' outer tab should NOT exist
    settings_outer = page.locator("[data-testid='models-settings-tab']")
    count = await settings_outer.count()
    assert count == 0, "There should be no outer 'Settings' tab"


@pytest.mark.asyncio
async def test_all_tabs_visible(page: Page):
    """All 3 tabs should be visible at the top."""
    await page.goto(MODELS_PAGE_URL)
    await expect(page.locator("[data-testid='models-settings-main-tab']")).to_be_visible()
    await expect(page.locator("[data-testid='models-settings-aux-tab']")).to_be_visible()
    await expect(page.locator("[data-testid='models-settings-used-tab']")).to_be_visible()


@pytest.mark.asyncio
async def test_main_model_tab_has_card(page: Page):
    """Main Model tab should show the Main Model card."""
    await _go_to_tab(page, "Main Model")
    await expect(page.locator("[data-testid='main-model-card']")).to_be_visible()


@pytest.mark.asyncio
async def test_main_model_tab_has_stats(page: Page):
    """Main Model tab should show the stats card."""
    await _go_to_tab(page, "Main Model")
    stats = page.locator("[data-testid='settings-tab-panel'] > div > div > div")
    count = await stats.count()
    assert count >= 2, "Should have at least 2 cards (main model + stats)"


@pytest.mark.asyncio
async def test_main_model_card_refreshes_after_assignment(page: Page):
    """After changing the main model via the picker, the card should update without a reload."""
    await _go_to_tab(page, "Main Model")

    # Record the current main model text before changing
    card = page.locator("[data-testid='main-model-card']")
    before_text = await card.inner_text()

    # Read available providers to pick a different one
    options = await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const r = await fetch('/api/model/options', {
            headers: { 'X-Hermes-Session-Token': token, 'Content-Type': 'application/json' }
        });
        return await r.json();
    }""")
    providers = options.get("providers", [])
    usable = [p for p in providers if p.get("models")]
    if not usable:
        pytest.skip("No providers with models available")

    # Pick a provider whose slug is NOT currently shown in the card
    target = None
    for p in usable:
        if p["slug"].lower() not in before_text.lower():
            target = p
            break
    if target is None:
        target = usable[0]

    target_slug = target["slug"]
    target_model = target["models"][0]
    target_name = target.get("name") or target_slug

    # Open the "Change" picker on the main model card
    change_btn = card.get_by_role("button", name="Change")
    await change_btn.click()

    picker = page.get_by_role("dialog", name="Set Main Model")
    await expect(picker).to_be_visible()
    if not await picker.is_visible():
        pytest.skip("Main model picker did not open")

    # Select provider
    prov_loc = picker.get_by_text(target_name, exact=True)
    if await prov_loc.count() == 0:
        await page.keyboard.press("Escape")
        pytest.skip(f"Provider '{target_name}' not found in picker")
    await prov_loc.first.click()
    # Wait for model list to refresh after provider selection
    await expect(picker.get_by_role("button", name="Switch", exact=True)).to_be_visible()

    # Select model
    model_loc = picker.get_by_text(target_model, exact=True)
    if await model_loc.count() == 0:
        search = picker.locator("input[placeholder*='Filter']")
        if await search.count() > 0:
            await search.fill(target_model)
            await expect(model_loc).to_be_visible()
            model_loc = picker.get_by_text(target_model, exact=True)
    if await model_loc.count() > 0:
        await model_loc.first.click()

    confirm_btn = picker.get_by_role("button", name="Switch", exact=True)
    if not await confirm_btn.is_enabled():
        await page.keyboard.press("Escape")
        pytest.skip("Could not select model in picker")

    await confirm_btn.click()
    await expect(picker).to_be_hidden()

    # Verify the card updated WITHOUT a reload
    after_text = (await card.inner_text()).lower()
    assert target_slug.lower() in after_text or target_model.lower() in after_text, \
        f"Main model card should show '{target_slug}' or '{target_model}' after assignment (no reload), got: {after_text}"


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


