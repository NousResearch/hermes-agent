"""Playwright UI tests for the fallback chain.

Covers layout, add, remove, reorder, auto-save, validation, persistence,
and error handling — everything related to fallback providers in one suite.
"""
from __future__ import annotations

import pytest
from playwright.async_api import Page, expect

from tests.web.models.conftest import MODELS_PAGE_URL

pytestmark = pytest.mark.xdist_group("web_models")


# ─── Helpers ────────────────────────────────────────────────────────────────


async def _goto(page: Page) -> None:
    await page.goto(MODELS_PAGE_URL)
    await expect(page.get_by_test_id("fallback-chain-card")).to_be_visible(timeout=10000)


async def _click_and_wait_fallback_put(page: Page, locator) -> None:
    async with page.expect_response(
        lambda response: "/api/model/fallbacks" in response.url
        and response.request.method == "PUT"
    ) as response_info:
        await locator.click()
    response = await response_info.value
    assert response.ok, f"fallback save failed with HTTP {response.status}"


async def _get_model_options(page: Page) -> dict:
    return await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const r = await fetch('/api/model/options', {
            headers: { 'X-Hermes-Session-Token': token, 'Content-Type': 'application/json' }
        });
        return await r.json();
    }""")


async def _get_configured_models(page: Page) -> dict:
    return await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const r = await fetch('/api/model/fallbacks', {
            headers: { 'X-Hermes-Session-Token': token, 'Content-Type': 'application/json' }
        });
        return await r.json();
    }""")


async def _get_raw_yaml(page: Page) -> str:
    return await page.evaluate("""async () => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const r = await fetch('/api/config/raw', {
            headers: { 'X-Hermes-Session-Token': token, 'Content-Type': 'application/json' }
        });
        const d = await r.json();
        return d.yaml || '';
    }""")


def _parse_yaml(text: str) -> dict:
    import yaml
    return yaml.safe_load(text) or {}


async def _save_api(page: Page, fallbacks: list[dict]) -> dict:
    return await page.evaluate("""async (fb) => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const r = await fetch('/api/model/fallbacks', {
            method: 'PUT',
            headers: { 'X-Hermes-Session-Token': token, 'Content-Type': 'application/json' },
            body: JSON.stringify({ fallbacks: fb })
        });
        return await r.json();
    }""", fallbacks)


async def _ui_order(page: Page) -> list[str]:
    return await page.evaluate("""() => {
        return Array.from(document.querySelectorAll('[data-testid^="fallback-item-"]')).map(el => {
            const spans = el.querySelectorAll('span.font-mono');
            return spans.length >= 2 ? spans[1].textContent.trim() : (spans[0]?.textContent.trim() || '');
        });
    }""")


async def _add_via_picker(page: Page, provider_name: str, model_name: str) -> None:
    add_btn = page.get_by_role("button", name="Add", exact=True)
    try:
        if await add_btn.is_disabled():
            raise AssertionError("Add button is disabled")
        await add_btn.click(timeout=3000)
        await page.get_by_role("dialog", name="Add Fallback Provider").wait_for(timeout=3000)
    except Exception as exc:
        raise AssertionError(f"Could not open fallback picker: {exc}") from exc

    picker = page.get_by_role("dialog", name="Add Fallback Provider")
    provider = picker.get_by_text(provider_name, exact=True)
    try:
        await expect(provider.first).to_be_visible(timeout=3000)
    except Exception as exc:
        raise AssertionError(f"Provider '{provider_name}' not found in picker") from exc
    await provider.first.click(timeout=2000)

    model_loc = picker.get_by_text(model_name, exact=True)
    if await model_loc.count() > 0:
        await model_loc.first.click(timeout=2000)
    else:
        search = picker.locator("input[placeholder*='Filter']")
        if await search.count() > 0:
            await search.fill(model_name)
            model_loc = picker.get_by_text(model_name, exact=True)
            if await model_loc.count() > 0:
                await model_loc.first.click(timeout=2000)

    confirm = picker.get_by_role("button", name="Save", exact=True)
    assert await confirm.is_enabled(), f"Model '{model_name}' could not be selected in picker"
    async with page.expect_response(
        lambda response: "/api/model/fallbacks" in response.url
        and response.request.method == "PUT"
    ) as response_info:
        await confirm.click(timeout=2000)
    response = await response_info.value
    assert response.ok, f"fallback save failed with HTTP {response.status}"
    await expect(picker).not_to_be_visible(timeout=5000)


async def _put_invalid(page: Page, fallbacks: list[object]) -> dict:
    return await page.evaluate("""async (fb) => {
        const token = window.__HERMES_SESSION_TOKEN__;
        const r = await fetch('/api/model/fallbacks', {
            method: 'PUT',
            headers: { 'X-Hermes-Session-Token': token, 'Content-Type': 'application/json' },
            body: JSON.stringify({ fallbacks: fb })
        });
        return { ok: r.ok, status: r.status, body: await r.json() };
    }""", fallbacks)


# ─── Layout ─────────────────────────────────────────────────────────────────


class TestLayout:

    @pytest.mark.asyncio
    async def test_fallback_chain_card_visible(self, page: Page):
        await _goto(page)
        await expect(page.locator("[data-testid='fallback-chain-card']")).to_be_visible()

    @pytest.mark.asyncio
    async def test_add_button_visible(self, page: Page):
        await _goto(page)
        add_btn = page.locator("[data-testid='fallback-chain-card'] [data-testid='fallback-add-button']")
        await expect(add_btn).to_be_visible()

    @pytest.mark.asyncio
    async def test_no_save_button(self, page: Page):
        await _goto(page)
        save_btn = page.locator("[data-testid='fallback-save-button']")
        assert await save_btn.count() == 0, "There should be no Save button — all actions auto-save"

    @pytest.mark.asyncio
    async def test_add_opens_picker(self, page: Page):
        await _goto(page)
        await page.locator("[data-testid='fallback-add-button']").click()
        await expect(page.get_by_role("dialog", name="Add Fallback Provider")).to_be_visible(timeout=5000)

    @pytest.mark.asyncio
    async def test_move_buttons_visible_with_multiple_items(self, page: Page):
        await _save_api(page, [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ])
        await _goto(page)
        await expect(page.locator("[data-testid='fallback-move-down-0']")).to_be_visible()
        await expect(page.locator("[data-testid='fallback-move-up-1']")).to_be_visible()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "selector,expected_disabled",
        [
            ("[data-testid='fallback-move-up-0']", True),
            ("[data-testid='fallback-move-down-1']", True),
            ("[data-testid='fallback-move-down-0']", False),
            ("[data-testid='fallback-move-up-1']", False),
        ],
    )
    async def test_move_boundary_buttons_disabled(self, page: Page, selector: str, expected_disabled: bool):
        await _save_api(page, [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ])
        await _goto(page)
        btn = page.locator(selector)
        if expected_disabled:
            assert await btn.is_disabled()
        else:
            assert await btn.is_enabled()

    @pytest.mark.asyncio
    async def test_no_error_on_normal_state(self, page: Page):
        await _save_api(page, [{"provider": "test", "model": "model"}])
        await _goto(page)
        error = page.locator("[data-testid='fallback-error']")
        assert not await error.is_visible()

    @pytest.mark.asyncio
    async def test_error_shown_when_initial_load_fails(self, page: Page):
        """When the fallback GET fails on page load, an error should be visible."""
        await page.route("**/api/model/fallbacks", lambda route: route.fulfill(
            status=500, content_type="application/json", body='{"detail":"server error"}',
        ))
        await _goto(page)
        error = page.locator("[data-testid='fallback-error']")
        await expect(error).to_be_visible(timeout=5000)
        await page.unroute("**/api/model/fallbacks")


# ─── Add ────────────────────────────────────────────────────────────────────


class TestAdd:

    @pytest.mark.asyncio
    async def test_add_via_picker_auto_saves(self, page: Page):
        await _save_api(page, [])
        await _goto(page)
        options = await _get_model_options(page)
        providers = options.get("providers", [])
        assert len(providers) > 0, "No providers available from /api/model/options"
        p = providers[0]
        assert p.get("models"), f"Provider '{p['slug']}' has no models"
        model = p["models"][0]
        await _add_via_picker(page, p.get("name") or p["slug"], model)
        config = _parse_yaml(await _get_raw_yaml(page))
        fallbacks = config.get("fallback_providers", [])
        assert len(fallbacks) >= 1, "Add via picker should auto-save"
        assert fallbacks[0]["provider"] == p["slug"]
        assert fallbacks[0]["model"] == model

    @pytest.mark.asyncio
    async def test_add_via_picker_then_reorder(self, page: Page):
        await _save_api(page, [])
        await _goto(page)
        options = await _get_model_options(page)
        providers = options.get("providers", [])
        assert providers, "No providers available"
        items = page.locator("[data-testid^='fallback-item-']")
        for p in providers:
            if not p.get("models"):
                continue
            if await items.count() >= 3:
                break
            name = p.get("name") or p.get("slug")
            if not name:
                continue
            try:
                await _add_via_picker(page, name, p["models"][0])
            except AssertionError:
                picker = page.get_by_role("dialog", name="Add Fallback Provider")
                if await picker.is_visible(timeout=500):
                    await page.keyboard.press("Escape")
                    await expect(picker).not_to_be_visible(timeout=3000)
                continue
        items = page.locator("[data-testid^='fallback-item-']")
        count = await items.count()
        assert count >= 2, "Could not add enough providers via picker"
        last_idx = count - 1
        await _click_and_wait_fallback_put(page, page.locator(f"[data-testid='fallback-move-up-{last_idx}']"))
        ui = await _ui_order(page)
        config = _parse_yaml(await _get_raw_yaml(page))
        fallbacks = config["fallback_providers"]
        assert len(fallbacks) >= 2
        for i, fb in enumerate(fallbacks):
            expected = f"{fb['provider']} · {fb['model']}"
            assert ui[i] == expected

    @pytest.mark.asyncio
    async def test_base_url_preserved(self, page: Page):
        response = await _save_api(page, [{
            "provider": "local-hermes-ov",
            "model": "test-model",
            "base_url": "http://192.168.1.161:11434/v1",
        }])
        assert response.get("ok") is True
        config = _parse_yaml(await _get_raw_yaml(page))
        fb = config["fallback_providers"][0]
        assert fb["provider"] == "local-hermes-ov"
        assert fb.get("base_url") == "http://192.168.1.161:11434/v1"


# ─── Remove ─────────────────────────────────────────────────────────────────


class TestRemove:

    @pytest.mark.asyncio
    async def test_remove_auto_saves(self, page: Page):
        await _save_api(page, [
            {"provider": "prov-a", "model": "model-a"},
            {"provider": "prov-b", "model": "model-b"},
        ])
        await _goto(page)
        await _click_and_wait_fallback_put(page, page.locator("[data-testid='fallback-remove-0']"))
        config = _parse_yaml(await _get_raw_yaml(page))
        fallbacks = config.get("fallback_providers", [])
        assert len(fallbacks) == 1
        assert fallbacks[0]["provider"] == "prov-b"

    @pytest.mark.asyncio
    async def test_remove_preserves_order(self, page: Page):
        await _save_api(page, [
            {"provider": "keep-a", "model": "ma"},
            {"provider": "remove-b", "model": "mb"},
            {"provider": "keep-c", "model": "mc"},
        ])
        await _goto(page)
        await _click_and_wait_fallback_put(page, page.locator("[data-testid='fallback-remove-1']"))
        config = _parse_yaml(await _get_raw_yaml(page))
        fallbacks = config["fallback_providers"]
        assert len(fallbacks) == 2
        assert fallbacks[0]["provider"] == "keep-a"
        assert fallbacks[1]["provider"] == "keep-c"


# ─── Reorder ────────────────────────────────────────────────────────────────


class TestReorder:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "click_selector,expected_order",
        [
            ("[data-testid='fallback-move-up-1']", ["second · mb", "first · ma", "third · mc"]),
            ("[data-testid='fallback-move-down-0']", ["second · mb", "first · ma", "third · mc"]),
        ],
    )
    async def test_move_changes_ui_order(self, page: Page, click_selector: str, expected_order: list[str]):
        await _save_api(page, [
            {"provider": "first", "model": "ma"},
            {"provider": "second", "model": "mb"},
            {"provider": "third", "model": "mc"},
        ])
        await _goto(page)
        await _click_and_wait_fallback_put(page, page.locator(click_selector))
        order = await _ui_order(page)
        assert order[0] == expected_order[0]
        assert order[1] == expected_order[1]
        assert order[2] == expected_order[2]

    @pytest.mark.asyncio
    async def test_reorder_auto_saves_to_config(self, page: Page):
        await _save_api(page, [
            {"provider": "first", "model": "ma"},
            {"provider": "second", "model": "mb"},
            {"provider": "third", "model": "mc"},
        ])
        await _goto(page)
        await _click_and_wait_fallback_put(page, page.locator("[data-testid='fallback-move-up-2']"))
        config = _parse_yaml(await _get_raw_yaml(page))
        fallbacks = config["fallback_providers"]
        assert fallbacks[0]["provider"] == "first"
        assert fallbacks[1]["provider"] == "third"
        assert fallbacks[2]["provider"] == "second"

    @pytest.mark.asyncio
    async def test_multiple_reorders(self, page: Page):
        await _save_api(page, [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
            {"provider": "c", "model": "mc"},
            {"provider": "d", "model": "md"},
        ])
        await _goto(page)
        for idx in range(3, 0, -1):
            await _click_and_wait_fallback_put(page, page.locator(f"[data-testid='fallback-move-up-{idx}']"))
        order = await _ui_order(page)
        assert order[0] == "d · md"
        config = _parse_yaml(await _get_raw_yaml(page))
        assert config["fallback_providers"][0]["provider"] == "d"
        assert config["fallback_providers"][-1]["provider"] == "c"

    @pytest.mark.asyncio
    async def test_ui_order_matches_config(self, page: Page):
        await _save_api(page, [
            {"provider": "match-a", "model": "ma"},
            {"provider": "match-b", "model": "mb"},
            {"provider": "match-c", "model": "mc"},
        ])
        await _goto(page)
        await _click_and_wait_fallback_put(page, page.locator("[data-testid='fallback-move-up-2']"))
        ui = await _ui_order(page)
        config = _parse_yaml(await _get_raw_yaml(page))
        for i, fb in enumerate(config["fallback_providers"]):
            expected = f"{fb['provider']} · {fb['model']}"
            assert ui[i] == expected, f"UI[{i}]='{ui[i]}' != config[{i}]='{expected}'"


# ─── Validation ─────────────────────────────────────────────────────────────


class TestValidation:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload,expected_status",
        [
            ([{"provider": "", "model": "some-model"}], 422),
            ([{"provider": "some-provider", "model": ""}], 422),
            ([{"provider": "   ", "model": "some-model"}], 422),
            ([1], 422),
        ],
    )
    async def test_invalid_fallback_rejected(self, page: Page, payload: list, expected_status: int):
        r = await _put_invalid(page, payload)
        assert r["status"] == expected_status
        if isinstance(r.get("body"), dict) and "detail" in r["body"]:
            assert r["body"]["detail"]


# ─── Busy state ─────────────────────────────────────────────────────────────


class TestBusyState:

    @pytest.mark.asyncio
    async def test_move_buttons_disabled_during_save(self, page: Page):
        """Move up/down/remove buttons should be disabled while a save is in flight."""
        await _save_api(page, [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ])
        await _goto(page)

        # Intercept PUT to stall the save (using asyncio.sleep)
        import asyncio as _aio

        async def stall_put(route):
            if route.request.method == "PUT":
                await _aio.sleep(3)
            await route.continue_()

        await page.route("**/api/model/fallbacks", stall_put)

        # Click move up (fires async save — does NOT await the PUT response)
        await page.locator("[data-testid='fallback-move-up-1']").click()

        await expect(page.locator("[data-testid='fallback-move-up-1']")).to_be_disabled()
        await expect(page.locator("[data-testid='fallback-move-down-0']")).to_be_disabled()
        await expect(page.locator("[data-testid='fallback-remove-0']")).to_be_disabled()

        # Unregister the slow route so teardown can proceed quickly
        await page.unroute("**/api/model/fallbacks")

    @pytest.mark.asyncio
    async def test_api_mode_preserved_through_save(self, page: Page):
        """api_mode on a fallback entry should survive a reorder/save round-trip."""
        # Seed a fallback with api_mode set
        await _save_api(page, [
            {"provider": "prov-a", "model": "ma", "api_mode": "chat_completions"},
            {"provider": "prov-b", "model": "mb"},
        ])
        await _goto(page)

        # Trigger a reorder (save round-trip)
        await _click_and_wait_fallback_put(page, page.locator("[data-testid='fallback-move-down-0']"))

        # Verify api_mode is preserved in config
        config = _parse_yaml(await _get_raw_yaml(page))
        fallbacks = config["fallback_providers"]
        prov_a = next((f for f in fallbacks if f["provider"] == "prov-a"), None)
        assert prov_a is not None
        assert prov_a.get("api_mode") == "chat_completions", \
            f"api_mode should be preserved after reorder, got: {prov_a}"


# ─── Persistence ────────────────────────────────────────────────────────────


class TestPersistence:

    @pytest.mark.asyncio
    async def test_persists_after_reload(self, page: Page):
        await _save_api(page, [{"provider": "persistent", "model": "test-model"}])
        config = _parse_yaml(await _get_raw_yaml(page))
        assert any(fb["provider"] == "persistent" for fb in config.get("fallback_providers", []))
        await _goto(page)
        await page.reload()
        await expect(page.get_by_test_id("fallback-chain-card")).to_be_visible(timeout=10000)
        await expect(page.get_by_test_id("fallback-item-0")).to_be_visible(timeout=10000)
        config = _parse_yaml(await _get_raw_yaml(page))
        assert any(fb["provider"] == "persistent" for fb in config.get("fallback_providers", []))

    @pytest.mark.asyncio
    async def test_reorder_persists_after_reload(self, page: Page):
        await _save_api(page, [
            {"provider": "persist-a", "model": "ma"},
            {"provider": "persist-b", "model": "mb"},
            {"provider": "persist-c", "model": "mc"},
        ])
        await _goto(page)
        for idx in range(2, 0, -1):
            await _click_and_wait_fallback_put(page, page.locator(f"[data-testid='fallback-move-up-{idx}']"))
        await page.reload()
        await expect(page.get_by_test_id("fallback-chain-card")).to_be_visible(timeout=10000)
        await expect(page.get_by_test_id("fallback-item-0")).to_be_visible(timeout=10000)
        order = await _ui_order(page)
        assert order[0] == "persist-c · mc"
        assert order[1] == "persist-a · ma"
        assert order[2] == "persist-b · mb"
        config = _parse_yaml(await _get_raw_yaml(page))
        assert config["fallback_providers"][0]["provider"] == "persist-c"
