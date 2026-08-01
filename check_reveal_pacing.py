"""Browser check for the natural-reveal pacing in web/app.js.

Stubs POST /api/chat with a long canned reply (so no Ollama is needed), sends
a message in the real UI, and samples the assistant bubble over time. Passes
only if the text grows progressively — partial early, complete later — and
lands exactly on the full reply.

Run:  ./.venv/bin/python check_reveal_pacing.py
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time

os.environ.setdefault("NEURAL_MAP_DB", tempfile.mkdtemp(prefix="nm-pacing-"))

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

import server  # noqa: E402

PORT = 8791
REPLY = (
    "I've been thinking about this, and honestly, I'd go with the second option. "
    "It fits what you told me last week about wanting fewer moving parts, and it "
    "keeps the door open for the finance work later. The first one is flashier, "
    "but flashy isn't what you need right now. If I'm remembering right, you also "
    "said mornings are your focus time — so I'd schedule the hard part there. "
    "Want me to sketch the first three steps?"
) * 3  # ~1.1k chars: long enough to reveal over several seconds


def main() -> int:
    config = uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    for _ in range(50):
        if instance.started:
            break
        time.sleep(0.1)
    if not instance.started:
        print("FAIL: server did not start")
        return 1

    samples: list[tuple[float, int]] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            def stub_chat(route):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({
                        "reply": REPLY,
                        "memories": [],
                        "tool_results": [],
                        "tools_used": [],
                        "count": 0,
                    }),
                )

            page.route("**/api/chat", stub_chat)
            page.goto(f"http://127.0.0.1:{PORT}", wait_until="networkidle")
            # The bubble shows formatAssistantDisplayText(reply), not the raw
            # reply — compute the expectation with the page's own formatter.
            expected = page.evaluate("reply => formatAssistantDisplayText(reply)", REPLY)
            page.locator("#input").fill("which option should I pick?")
            start = time.monotonic()
            page.locator("#composer").evaluate("form => form.requestSubmit()")

            full_len = 0
            for _ in range(240):  # up to 24s
                elapsed = time.monotonic() - start
                text = page.locator(".msg.assistant .bubble-content").last.text_content() or ""
                full_len = len(text)
                samples.append((round(elapsed, 2), full_len))
                if full_len >= len(expected):
                    break
                time.sleep(0.1)
            browser.close()
    finally:
        instance.should_exit = True
        thread.join(timeout=5)

    early = next((n for t, n in samples if t >= 0.5), 0)
    mid = next((n for t, n in samples if t >= 1.5), 0)
    final_t, final_n = samples[-1]

    print(f"reply length: {len(expected)} chars (formatted)")
    print(f"t=0.5s: {early} chars | t=1.5s: {mid} chars | done at t={final_t}s with {final_n} chars")

    ok = True
    if final_n != len(expected):
        print("FAIL: final bubble text does not match the full reply")
        ok = False
    if not (0 < early < len(expected)):
        print("FAIL: reply was not mid-reveal at t=0.5s (appeared all at once?)")
        ok = False
    if not (early < mid <= len(expected)):
        print("FAIL: reveal did not progress between t=0.5s and t=1.5s")
        ok = False
    if not (1.0 <= final_t <= 12.0):
        print("FAIL: reveal finished outside the 1-12s pacing envelope")
        ok = False

    print("PASS: reply revealed progressively and landed complete" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
