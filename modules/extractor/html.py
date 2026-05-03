from __future__ import annotations

from pathlib import Path

from safe_result import safe_async
from playwright.async_api import async_playwright

from modules.utils.text import validate_url


_SNAPSHOT_SCRIPT_PATH = Path(__file__).with_name("semantic_snapshot.js")


def _semantic_snapshot_script() -> str:
    """Return the bundled DOM semantic snapshot script."""
    return _SNAPSHOT_SCRIPT_PATH.read_text(encoding="utf-8")


@safe_async
async def read_page_content(url: str) -> str:
    """Fetch a page and return a semantic YAML snapshot for LLM use."""
    resolved_url = validate_url(url).unwrap().geturl()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            try:
                await page.goto(resolved_url, wait_until="networkidle", timeout=30000)
            except Exception:
                await page.goto(
                    resolved_url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
            snapshot = await page.evaluate(_semantic_snapshot_script())
        finally:
            await browser.close()

    return (
        f"Page URL: {snapshot['url']}\n"
        f"Page Title: {snapshot['title']}\n"
        "Page Snapshot:\n"
        "```yaml\n"
        f"{snapshot['yaml']}\n"
        "```"
    )
