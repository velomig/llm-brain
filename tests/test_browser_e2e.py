"""Optional browser smoke tests for Neural Map.

Run these after installing Playwright:
    ./.venv/bin/python -m pip install 'pytest-playwright>=0.5,<1'
    ./.venv/bin/python -m playwright install chromium
    ./.venv/bin/python -m pytest -m e2e

The normal suite skips this file when Playwright is absent, so backend work stays
fast while a real browser check is available when desired.
"""
from __future__ import annotations

import threading
import time
import os

import pytest
import uvicorn

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:  # pragma: no cover - exercised only when Playwright missing
    pytestmark = pytest.mark.skip(reason="Playwright is not installed")
else:
    pytestmark = pytest.mark.e2e

import server

if not os.getenv("NEURAL_MAP_RUN_E2E"):
    pytestmark = pytest.mark.skip(reason="set NEURAL_MAP_RUN_E2E=1 to run browser e2e tests")


@pytest.fixture(scope="module")
def live_server():
    config = uvicorn.Config(server.app, host="127.0.0.1", port=8765, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    for _ in range(50):
        if instance.started:
            break
        time.sleep(0.1)
    if not instance.started:
        instance.should_exit = True
        thread.join(timeout=5)
        pytest.skip("live uvicorn server did not start")
    yield "http://127.0.0.1:8765"
    instance.should_exit = True
    thread.join(timeout=5)


def test_browser_can_open_core_panels(live_server):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(live_server, wait_until="networkidle")

        expect(page.locator("#messages")).to_be_visible()
        expect(page.locator("#model")).to_be_visible()
        expect(page.locator("#systemsBtn")).to_be_visible()
        expect(page.locator("#observabilityBtn")).to_be_visible()

        page.locator("#systemsBtn").click()
        expect(page.locator("#systemsPanel")).to_be_visible()
        expect(page.locator("#integrityList")).to_be_visible()
        expect(page.locator("#homeostasisList")).to_be_visible()
        expect(page.locator("#evolutionList")).to_be_visible()

        page.locator("#observabilityBtn").click()
        expect(page.locator("#observabilityPanel")).to_be_visible()
        expect(page.locator("#observabilityLive")).to_be_visible()
        expect(page.locator("#observabilityTraceList")).to_be_visible()

        page.locator("#retrievalBtn").click()
        expect(page.locator("#retrievalPanel")).to_be_visible()
        expect(page.locator("#recallQuery")).to_be_visible()

        page.locator("#manageTopBtn").click()
        expect(page.locator("#manager")).to_be_visible()
        expect(page.locator("#summarizeCategory")).to_be_visible()

        browser.close()
