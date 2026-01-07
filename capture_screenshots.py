"""Capture screenshots of the dashboard and API for README."""

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

async def capture_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # Dashboard screenshots
        print("Capturing dashboard screenshots...")
        page = await browser.new_page(viewport={"width": 1400, "height": 900})

        try:
            # Main dashboard - wait longer for full load
            await page.goto("http://localhost:8505", timeout=60000)
            await page.wait_for_load_state("networkidle", timeout=60000)

            # Wait for models to load (look for the sidebar or main content)
            print("  Waiting for dashboard to fully load...")
            await asyncio.sleep(8)  # Give more time for Streamlit to render

            # Try to wait for specific element
            try:
                await page.wait_for_selector("text=Patient Information", timeout=30000)
            except:
                pass

            await asyncio.sleep(2)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "dashboard_main.png"), full_page=False)
            print("  - Captured dashboard_main.png")

            # Scroll down to show clinical input section
            await page.evaluate("window.scrollTo(0, 400)")
            await asyncio.sleep(1)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "dashboard_clinical.png"), full_page=False)
            print("  - Captured dashboard_clinical.png")

            # Scroll to show more
            await page.evaluate("window.scrollTo(0, 800)")
            await asyncio.sleep(1)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "dashboard_features.png"), full_page=False)
            print("  - Captured dashboard_features.png")

        except Exception as e:
            print(f"  Dashboard error: {e}")

        # API docs screenshots
        print("Capturing API documentation screenshots...")
        try:
            await page.goto("http://localhost:8006/docs", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=30000)
            await asyncio.sleep(3)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "api_docs.png"), full_page=False)
            print("  - Captured api_docs.png")

            # Expand predict/tabular endpoint
            endpoints = await page.query_selector_all(".opblock-summary")
            if len(endpoints) > 1:
                await endpoints[1].click()  # Click on second endpoint
                await asyncio.sleep(1)
                await page.screenshot(path=str(SCREENSHOTS_DIR / "api_endpoint.png"), full_page=False)
                print("  - Captured api_endpoint.png")

        except Exception as e:
            print(f"  API docs error: {e}")

        await browser.close()
        print("\nScreenshots saved to docs/screenshots/")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
