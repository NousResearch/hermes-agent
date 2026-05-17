"""Playwright configuration for Hermes dashboard UI tests."""
from playwright.async_api import async_playwright

import asyncio


async def main():
    """Run the fallback chain UI tests."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = await browser.new_context()
        page = await context.new_page()

        # Get the session token from the running dashboard
        await page.goto("http://127.0.0.1:9119")
        token = await page.evaluate("""() => {
            const match = document.body.innerHTML.match(/__HERMES_SESSION_TOKEN__="([^"]+)"/);
            return match ? match[1] : null;
        }""")

        if token:
            await page.add_init_script(f"""
                window.__HERMES_SESSION_TOKEN__ = "{token}";
            """)

        # Navigate to the Models page
        await page.goto("http://127.0.0.1:9119/models")
        await page.wait_for_timeout(2000)  # Wait for data to load

        # Test 1: Fallback chain section exists
        print("Test 1: Checking fallback chain section...")
        fallback_chain = page.get_by_test_id("fallback-chain")
        try:
            await fallback_chain.wait_for(state="visible", timeout=5000)
            print("✓ Fallback chain section is visible")
        except Exception as e:
            print(f"✗ Fallback chain section not visible: {e}")

        # Test 2: Add button exists
        print("\nTest 2: Checking Add button...")
        add_button = page.get_by_test_id("fallback-chain").get_by_role("button", name="Add")
        try:
            await add_button.wait_for(state="visible", timeout=5000)
            print("✓ Add button is visible")
        except Exception as e:
            print(f"✗ Add button not visible: {e}")

        # Test 3: Save button exists
        print("\nTest 3: Checking Save button...")
        save_button = page.get_by_test_id("fallback-chain").get_by_role("button", name="Save")
        try:
            await save_button.wait_for(state="visible", timeout=5000)
            print("✓ Save button is visible")
        except Exception as e:
            print(f"✗ Save button not visible: {e}")

        # Test 4: Check for fallback items
        print("\nTest 4: Checking for fallback items...")
        fallback_items = page.get_by_test_id("fallback-chain").locator("[data-testid^='fallback-item-']")
        count = await fallback_items.count()
        print(f"Found {count} fallback items")

        if count > 0:
            # Test 5: Remove button exists
            print("\nTest 5: Checking remove button...")
            first_item = page.get_by_test_id("fallback-item-0")
            remove_button = first_item.get_by_role("button", name="×")
            try:
                await remove_button.wait_for(state="visible", timeout=5000)
                print("✓ Remove button is visible")
            except Exception as e:
                print(f"✗ Remove button not visible: {e}")

            # Test 6: Move up button exists (if not first item)
            if count > 1:
                print("\nTest 6: Checking move up button...")
                second_item = page.get_by_test_id("fallback-item-1")
                move_up_button = second_item.get_by_role("button", name="↑")
                try:
                    await move_up_button.wait_for(state="visible", timeout=5000)
                    print("✓ Move up button is visible")
                except Exception as e:
                    print(f"✗ Move up button not visible: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
