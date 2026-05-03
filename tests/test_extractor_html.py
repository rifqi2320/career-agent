from __future__ import annotations

from playwright.async_api import async_playwright
import pytest

from modules.extractor.html import _semantic_snapshot_script


@pytest.mark.asyncio
async def test_semantic_snapshot_includes_collapsed_main_accordion_panel() -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(
                """
                <main>
                  <button aria-expanded="false" aria-controls="qualification">
                    Qualification
                  </button>
                  <div id="qualification" style="visibility: hidden">
                    <p><b>BONUS POINTS IF YOU HAVE</b></p>
                    <ul>
                      <li>
                        Experience with tools such as DBT, Matillion, Airflow,
                        or similar.
                      </li>
                    </ul>
                  </div>
                  <div style="visibility: hidden">
                    Hidden non-accordion text
                  </div>
                </main>
                <button aria-expanded="false" aria-controls="cookie-details">
                  Cookie Details
                </button>
                <div id="cookie-details" style="visibility: hidden">
                  Tracking preferences
                </div>
                """
            )

            snapshot = await page.evaluate(_semantic_snapshot_script())
        finally:
            await browser.close()

    yaml = snapshot["yaml"]
    assert "BONUS POINTS IF YOU HAVE" in yaml
    assert "DBT, Matillion, Airflow" in yaml
    assert "Hidden non-accordion text" not in yaml
    assert "Tracking preferences" not in yaml
