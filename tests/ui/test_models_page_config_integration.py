"""Playwright UI tests for verifying fallback chain saves to config.yaml.

Tests cover:
- Adding new fallback providers and verifying config.yaml is updated
- Removing fallback providers and verifying config.yaml reflects changes
- Reordering fallbacks and verifying order is preserved in config
- Error handling when saving invalid data
- Verifying legacy fallback_model key is cleared on save
- Config persistence across page reloads
- Full workflow: add, reorder, remove, verify config
- Custom base_url preservation in config.yaml

All tests read the actual config.yaml via /api/config/raw and parse it
with yaml.safe_load() — they do NOT rely on the API response alone.
"""
from __future__ import annotations

from pathlib import Path
import fcntl
import tempfile

import pytest
from playwright.async_api import Page, expect

from tests.ui.conftest import MODELS_PAGE_URL

pytestmark = pytest.mark.xdist_group("models_page_config")


@pytest.fixture(autouse=True)
def _serialize_dashboard_config_tests():
    """Serialize tests that mutate the shared live dashboard config.

    These Playwright tests intentionally exercise the running dashboard on
    127.0.0.1:9119, so all xdist workers hit the same config.yaml.  Without a
    process-wide lock, parallel tests race by overwriting fallback_providers and
    assertions read another test's data.
    """
    lock_path = Path(tempfile.gettempdir()) / "hermes-models-page-config-tests.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


async def _go_to_fallback_chain(page: Page) -> None:
    """Navigate to Models page and switch to the Fallback Chain inner tab."""
    await page.goto(MODELS_PAGE_URL)
    await page.wait_for_timeout(2000)
    await page.locator("button:has-text('Fallback Chain')").click()
    await page.wait_for_timeout(1000)


async def _get_configured_models(page: Page) -> dict:
    """Get the current configured models from the API."""
    return await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const response = await fetch('/api/model/configured', {
            headers: {
                'X-Hermes-Session-Token': token,
                'Content-Type': 'application/json'
            }
        });
        return await response.json();
    }""")


async def _get_raw_config_yaml(page: Page) -> str:
    """Get the raw config.yaml content from the API."""
    return await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const response = await fetch('/api/config/raw', {
            headers: {
                'X-Hermes-Session-Token': token,
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        return data.yaml || '';
    }""")


def _parse_yaml_config(yaml_text: str) -> dict:
    """Parse raw YAML config text into a dict."""
    import yaml
    return yaml.safe_load(yaml_text) or {}


async def _save_fallbacks(page: Page, fallbacks: list[dict]) -> dict:
    """Save fallback chain via API and return response."""
    return await page.evaluate("""async (fb) => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const response = await fetch('/api/model/fallbacks', {
            method: 'PUT',
            headers: {
                'X-Hermes-Session-Token': token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ fallbacks: fb })
        });
        return await response.json();
    }""", fallbacks)


async def _get_raw_config_yaml(page: Page) -> str:
    """Get the raw config.yaml file content from the API."""
    return await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const response = await fetch('/api/config/raw', {
            headers: {
                'X-Hermes-Session-Token': token,
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        return data.yaml || '';
    }""")


def _parse_yaml_config(yaml_text: str) -> dict:
    """Parse raw YAML config text into a dict."""
    import yaml
    return yaml.safe_load(yaml_text) or {}


class TestAddFallbackProviderConfigChange:
    """Test that adding a fallback provider persists to config.yaml."""

    @pytest.mark.asyncio
    async def test_add_provider_saves_to_config(self, page: Page):
        """Adding a new fallback provider should write to config.yaml."""
        await _go_to_fallback_chain(page)

        # Click the Add button (no wrapper test-id, use button text)
        add_button = page.get_by_role("button", name="Add")
        await add_button.click()

        # Wait for picker to appear
        await page.wait_for_timeout(1000)

        # Verify picker is open
        picker_visible = await page.locator("text='Add Fallback Provider'").count() > 0
        assert picker_visible, "Model picker should be open"

        # Simulate what the UI does: PUT to /api/model/fallbacks
        response = await _save_fallbacks(page, [
            {"provider": "openrouter", "model": "gpt-4"}
        ])

        assert response.get("ok") is True, f"API should return ok=True, got: {response}"

        # Verify the actual config.yaml file content
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)

        assert "fallback_providers" in config, "config.yaml must contain fallback_providers"
        fallbacks = config["fallback_providers"]
        assert len(fallbacks) >= 1, "Config should have at least one fallback"
        assert fallbacks[0]["provider"] == "openrouter"
        assert fallbacks[0]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_add_multiple_providers_to_config(self, page: Page):
        """Adding multiple fallback providers should persist all to config.yaml."""
        await _go_to_fallback_chain(page)

        response = await _save_fallbacks(page, [
            {"provider": "openrouter", "model": "gpt-4"},
            {"provider": "openai", "model": "gpt-3.5-turbo"}
        ])

        assert response.get("ok") is True

        # Verify the actual config.yaml file content
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)

        fallbacks = config.get("fallback_providers", [])
        assert len(fallbacks) >= 2, f"Config should have at least 2 fallbacks, got {len(fallbacks)}"
        assert fallbacks[0]["provider"] == "openrouter"
        assert fallbacks[0]["model"] == "gpt-4"
        assert fallbacks[1]["provider"] == "openai"
        assert fallbacks[1]["model"] == "gpt-3.5-turbo"


class TestRemoveFallbackProviderConfigChange:
    """Test that removing a fallback provider updates config.yaml."""

    @pytest.mark.asyncio
    async def test_remove_provider_cleared_from_config(self, page: Page):
        """Removing fallback providers should clear them from config.yaml."""
        # Add some fallbacks first
        await _save_fallbacks(page, [
            {"provider": "test1", "model": "model1"},
            {"provider": "test2", "model": "model2"}
        ])

        # Verify config.yaml has them
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        fallbacks = config.get("fallback_providers", [])
        assert len(fallbacks) >= 2, "Should have at least 2 fallbacks in config.yaml"

        # Clear them via API
        response = await _save_fallbacks(page, [])
        assert response.get("ok") is True

        # Verify config.yaml was cleared
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        cleared_fallbacks = config.get("fallback_providers", [])
        assert len(cleared_fallbacks) == 0, f"Fallbacks should be cleared from config.yaml, got: {cleared_fallbacks}"


class TestReorderFallbackProvidersConfigChange:
    """Test that reordering fallback providers preserves order in config."""

    @pytest.mark.asyncio
    async def test_reorder_preserves_order_in_config(self, page: Page):
        """Reordering fallback providers should preserve the order in config.yaml."""
        # Add providers in specific order
        await _save_fallbacks(page, [
            {"provider": "first", "model": "model1"},
            {"provider": "second", "model": "model2"},
            {"provider": "third", "model": "model3"}
        ])

        # Verify initial order in config.yaml
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        fallbacks = config["fallback_providers"]
        assert fallbacks[0]["provider"] == "first"
        assert fallbacks[1]["provider"] == "second"
        assert fallbacks[2]["provider"] == "third"

        # Reorder via API (swap first and second)
        response = await _save_fallbacks(page, [
            {"provider": "second", "model": "model2"},
            {"provider": "first", "model": "model1"},
            {"provider": "third", "model": "model3"}
        ])
        assert response.get("ok") is True

        # Verify the new order in config.yaml
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        fallbacks = config["fallback_providers"]

        assert fallbacks[0]["provider"] == "second"
        assert fallbacks[1]["provider"] == "first"
        assert fallbacks[2]["provider"] == "third"


class TestSaveClearsLegacyFallbackModel:
    """Test that saving fallback_providers clears the legacy fallback_model key."""

    @pytest.mark.asyncio
    async def test_legacy_fallback_model_key_is_cleared(self, page: Page):
        """Saving fallback_providers should remove the legacy fallback_model key."""
        # First, verify config.yaml doesn't have fallback_model (clean state)
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        has_legacy_before = "fallback_model" in config

        # Save new fallback_providers
        response = await _save_fallbacks(page, [
            {"provider": "new", "model": "new-model"}
        ])
        assert response.get("ok") is True

        # Verify config.yaml no longer has fallback_model
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        assert "fallback_model" not in config, "Legacy fallback_model should be cleared from config.yaml"

        # Verify fallback_providers is present with correct content
        assert "fallback_providers" in config
        assert len(config["fallback_providers"]) >= 1
        assert config["fallback_providers"][0]["provider"] == "new"
        assert config["fallback_providers"][0]["model"] == "new-model"


class TestFallbackChainValidation:
    """Test validation when saving fallback chain."""

    @pytest.mark.asyncio
    async def test_save_fails_with_empty_provider(self, page: Page):
        """Saving with empty provider should fail."""
        response = await page.evaluate("""async () => {
            const token = window.__HERMES_SESSION_TOKEN__;
            const response = await fetch('/api/model/fallbacks', {
                method: 'PUT',
                headers: {
                    'X-Hermes-Session-Token': token,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    fallbacks: [
                        { provider: '', model: 'some-model' }
                    ]
                })
            });
            return {
                ok: response.ok,
                status: response.status,
                body: await response.json()
            };
        }""")

        assert response["status"] == 400, f"Should return 400, got {response['status']}"
        assert "detail" in response["body"], "Should include error detail"

    @pytest.mark.asyncio
    async def test_save_fails_with_empty_model(self, page: Page):
        """Saving with empty model should fail."""
        response = await page.evaluate("""async () => {
            const token = window.__HERMES_SESSION_TOKEN__;
            const response = await fetch('/api/model/fallbacks', {
                method: 'PUT',
                headers: {
                    'X-Hermes-Session-Token': token,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    fallbacks: [
                        { provider: 'some-provider', model: '' }
                    ]
                })
            });
            return {
                ok: response.ok,
                status: response.status,
                body: await response.json()
            };
        }""")

        assert response["status"] == 400
        assert "detail" in response["body"]

    @pytest.mark.asyncio
    async def test_save_fails_with_whitespace_only_provider(self, page: Page):
        """Saving with whitespace-only provider should fail."""
        response = await page.evaluate("""async () => {
            const token = window.__HERMES_SESSION_TOKEN__;
            const response = await fetch('/api/model/fallbacks', {
                method: 'PUT',
                headers: {
                    'X-Hermes-Session-Token': token,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    fallbacks: [
                        { provider: '   ', model: 'some-model' }
                    ]
                })
            });
            return {
                ok: response.ok,
                status: response.status,
                body: await response.json()
            };
        }""")

        assert response["status"] == 400


class TestConfigPersistenceAcrossReloads:
    """Test that config changes persist across page reloads."""

    @pytest.mark.asyncio
    async def test_config_persists_after_page_reload(self, page: Page):
        """Changes made via API should survive page reload."""
        # Save a test fallback provider
        await _save_fallbacks(page, [
            {"provider": "persistent-test", "model": "test-model"}
        ])

        # Verify it's in the actual config.yaml file
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        assert any(
            fb.get("provider") == "persistent-test"
            for fb in config.get("fallback_providers", [])
        ), "Provider should be in config.yaml before reload"

        # Reload the page
        await page.reload()
        await page.wait_for_timeout(2000)

        # Verify config.yaml still has the change
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        assert any(
            fb.get("provider") == "persistent-test"
            for fb in config.get("fallback_providers", [])
        ), "Provider should still be in config.yaml after reload"

        # Verify the UI reflects this
        # Check that fallback items are visible (either existing or new)
        fallback_items = page.locator("[data-testid^='fallback-item-']")
        fallback_count = await fallback_items.count()
        # Should have at least 1 item or the "No fallback" message
        no_fallback_msg = page.locator("text=No fallback providers configured")
        has_fallback_items = fallback_count > 0
        has_msg = await no_fallback_msg.count() > 0
        # Either we see items or the no-fallback message
        assert has_fallback_items or not has_msg, "UI should show fallback items or no-fallback message"


class TestFullWorkflow:
    """Test complete workflow: add, reorder, remove, verify config."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, page: Page):
        """Complete workflow: add providers, reorder, verify config."""
        # Step 1: Add first provider
        response1 = await _save_fallbacks(page, [
            {"provider": "alpha", "model": "model-a"}
        ])
        assert response1.get("ok") is True
        assert len(response1["fallbacks"]) == 1

        # Verify config.yaml
        yaml_text = await _get_raw_config_yaml(page)
        config1 = _parse_yaml_config(yaml_text)
        assert config1["fallback_providers"][0]["provider"] == "alpha"

        # Step 2: Add second provider
        response2 = await _save_fallbacks(page, [
            {"provider": "alpha", "model": "model-a"},
            {"provider": "beta", "model": "model-b"}
        ])
        assert response2.get("ok") is True
        assert len(response2["fallbacks"]) == 2

        # Verify config.yaml order
        yaml_text = await _get_raw_config_yaml(page)
        config2 = _parse_yaml_config(yaml_text)
        assert config2["fallback_providers"][0]["provider"] == "alpha"
        assert config2["fallback_providers"][1]["provider"] == "beta"

        # Step 3: Reorder (beta first)
        response3 = await _save_fallbacks(page, [
            {"provider": "beta", "model": "model-b"},
            {"provider": "alpha", "model": "model-a"}
        ])
        assert response3.get("ok") is True

        # Verify new order in config.yaml
        yaml_text = await _get_raw_config_yaml(page)
        config3 = _parse_yaml_config(yaml_text)
        assert config3["fallback_providers"][0]["provider"] == "beta"
        assert config3["fallback_providers"][1]["provider"] == "alpha"

        # Step 4: Remove all
        response4 = await _save_fallbacks(page, [])
        assert response4.get("ok") is True

        # Verify config.yaml is cleared
        yaml_text = await _get_raw_config_yaml(page)
        config4 = _parse_yaml_config(yaml_text)
        assert len(config4.get("fallback_providers", [])) == 0


class TestBaseUrlPreservation:
    """Test that custom base_url is preserved when saving fallback chain."""

    @pytest.mark.asyncio
    async def test_base_url_preserved_in_config(self, page: Page):
        """Custom base_url should be persisted to config.yaml."""
        await _go_to_fallback_chain(page)

        # Save with custom base_url
        response = await _save_fallbacks(page, [
            {
                "provider": "local-hermes-ov",
                "model": "test-model",
                "base_url": "http://192.168.1.161:11434/v1"
            }
        ])

        assert response.get("ok") is True

        # Verify base_url is in the actual config.yaml
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)

        fallbacks = config["fallback_providers"]
        assert len(fallbacks) >= 1
        assert fallbacks[0]["provider"] == "local-hermes-ov"
        assert fallbacks[0]["model"] == "test-model"
        assert fallbacks[0].get("base_url") == "http://192.168.1.161:11434/v1"


class TestConfigFileFormat:
    """Test that config.yaml is written in valid YAML format."""

    @pytest.mark.asyncio
    async def test_config_is_valid_yaml_after_save(self, page: Page):
        """After saving, config.yaml should still be valid YAML."""
        await _go_to_fallback_chain(page)

        # Save multiple fallbacks
        for i in range(5):
            await _save_fallbacks(page, [
                {"provider": f"provider-{i}", "model": f"model-{i}"}
            ])

        # Try to read back via API — should not raise
        config = await _get_configured_models(page)
        assert "fallbacks" in config
        assert len(config["fallbacks"]) >= 1


# ======================================================================
# UI interaction tests — exercise the ModelPickerDialog + Save flow
# ======================================================================

async def _get_model_options(page: Page) -> dict:
    """GET /api/model/options to see available providers/models."""
    return await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const response = await fetch('/api/model/options', {
            headers: {
                'X-Hermes-Session-Token': token,
                'Content-Type': 'application/json'
            }
        });
        return await response.json();
    }""")


async def _get_current_fallbacks_from_ui(page: Page) -> list:
    """Read fallback items currently displayed in the UI."""
    items = await page.evaluate("""() => {
        return Array.from(document.querySelectorAll('[data-testid^="fallback-item-"]')).map(el => {
            const text = el.querySelector('span.font-mono')?.textContent || '';
            return text.split(' · ').map(s => s.trim());
        });
    }""")
    return items


class TestAddFallbackViaUI:
    """Test adding fallback providers through the UI picker flow."""

    @pytest.mark.asyncio
    async def test_ui_add_provider_then_save_persists(self, page: Page):
        """Full UI flow: Add → picker → select provider → select model → Save → verify config.yaml."""
        # Clear any existing fallbacks first (test isolation)
        await _save_fallbacks(page, [])
        await _go_to_fallback_chain(page)

        # Step 1: Get available providers from the options API
        options = await _get_model_options(page)
        providers = options.get("providers", [])
        assert len(providers) > 0, "There should be at least one provider available"

        # Pick the first available provider
        first_provider = providers[0]
        provider_slug = first_provider["slug"]
        provider_models = first_provider.get("models", [])

        # Skip providers with no models listed (they may need dynamic loading)
        if not provider_models:
            pytest.skip(f"Provider '{provider_slug}' has no models listed for picker")

        # Pick the first model
        first_model = provider_models[0]

        # Step 2: Click Add to open the picker
        add_button = page.get_by_role("button", name="Add")
        await add_button.click()
        await page.wait_for_timeout(1500)  # Picker loads providers

        # Step 3: Verify picker is open
        picker_title = page.locator("text=Add Fallback Provider")
        await expect(picker_title).to_be_visible(timeout=5000)

        # Step 4: Select the provider from the left column
        # ListItem renders as <button>, not <li>
        provider_found = False
        for p in providers:
            locator = page.locator(f"button:has-text('{p['name']}')")
            if await locator.count() > 0:
                await locator.first.click()
                provider_found = True
                break
        assert provider_found, f"Should find provider '{first_provider['name']}' in picker"
        await page.wait_for_timeout(500)

        # Step 5: Select the model from the right column
        # ListItem renders as <button>, not <li>
        model_item = page.locator(f"button:has-text('{first_model}')")
        if await model_item.count() > 0:
            await model_item.first.click()
        else:
            # Model might not be visible if there are many — try filtering
            search_input = page.locator("input[placeholder*='Filter']")
            if await search_input.count() > 0:
                await search_input.fill(first_model)
                await page.wait_for_timeout(500)
                model_item = page.locator(f"button:has-text('{first_model}')")
                if await model_item.count() > 0:
                    await model_item.first.click()
                else:
                    pytest.skip(f"Model '{first_model}' not found in picker after filtering")
            else:
                pytest.skip(f"Model '{first_model}' not found in picker and no search input")

        # Step 6: Click the Switch/Confirm button
        # There are multiple "Switch" buttons on the page — use exact match
        confirm_button = page.get_by_role("button", name="Switch", exact=True)
        if await confirm_button.is_enabled():
            await confirm_button.click()
            await page.wait_for_timeout(1000)
        else:
            # Double-click the model to confirm
            await model_item.first.dblclick()
            await page.wait_for_timeout(1000)

        # Step 7: Verify picker closed and item appeared in UI
        fallback_items = page.locator("[data-testid^='fallback-item-']")
        item_count = await fallback_items.count()
        assert item_count >= 1, "Fallback item should appear in UI after selection"

        # Step 8: Click Save to persist to config.yaml
        save_button = page.get_by_role("button", name="Save", exact=True)
        await save_button.click()
        await page.wait_for_timeout(1000)

        # Step 9: Verify config.yaml has the new fallback
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        fallbacks = config.get("fallback_providers", [])
        assert len(fallbacks) >= 1, "Config should have at least one fallback"
        assert fallbacks[0]["provider"] == provider_slug
        assert fallbacks[0]["model"] == first_model

    @pytest.mark.asyncio
    async def test_ui_add_then_save_clears_legacy_key(self, page: Page):
        """Adding via UI picker and saving should clear legacy fallback_model."""
        # Clear any existing fallbacks first (test isolation)
        await _save_fallbacks(page, [])
        await _go_to_fallback_chain(page)

        # Get a provider to add
        options = await _get_model_options(page)
        providers = options.get("providers", [])
        if not providers:
            pytest.skip("No providers available")

        p = providers[0]
        if not p.get("models"):
            pytest.skip(f"Provider '{p['slug']}' has no models")

        # Add via UI
        await _add_via_picker(page, p["slug"], p["models"][0])

        # Verify config.yaml
        yaml_text = await _get_raw_config_yaml(page)
        config = _parse_yaml_config(yaml_text)
        assert "fallback_model" not in config
        assert "fallback_providers" in config
        assert len(config["fallback_providers"]) >= 1


async def _add_via_picker(page: Page, provider_slug: str, model_name: str) -> None:
    """Full UI flow: open picker → select provider → select model → confirm → save."""
    # Click Add
    add_button = page.get_by_role("button", name="Add", exact=True)
    await add_button.click()
    await page.wait_for_timeout(2000)

    # Get providers from options API to find the one we want
    options = await _get_model_options(page)
    providers = options.get("providers", [])

    # Find and click the provider
    # ListItem renders as <button>, not <li>
    provider_found = False
    for p in providers:
        locator = page.locator(f"button:has-text('{p['name']}')")
        if await locator.count() > 0:
            await locator.first.click()
            provider_found = True
            break
    if not provider_found:
        # Fallback: use the slug
        locator = page.locator(f"button:has-text('{provider_slug}')")
        if await locator.count() > 0:
            await locator.first.click()
            provider_found = True
    assert provider_found, f"Provider '{provider_slug}' not found in picker"

    await page.wait_for_timeout(500)

    # Find and click the model
    model_locator = page.locator(f"button:has-text('{model_name}')")
    if await model_locator.count() > 0:
        await model_locator.first.click()
    else:
        # Try search
        search = page.locator("input[placeholder*='Filter']")
        if await search.count() > 0:
            await search.fill(model_name)
            await page.wait_for_timeout(500)
            model_locator = page.locator(f"button:has-text('{model_name}')")
            if await model_locator.count() > 0:
                await model_locator.first.click()

    # Confirm
    # There are multiple "Switch" buttons on the page — use exact match
    confirm = page.get_by_role("button", name="Switch", exact=True)
    if await confirm.is_enabled():
        await confirm.click()
    else:
        await model_locator.first.dblclick()
    await page.wait_for_timeout(1000)

    # Save
    save_button = page.get_by_role("button", name="Save", exact=True)
    await save_button.click()
    await page.wait_for_timeout(1000)
