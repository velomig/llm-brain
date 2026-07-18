"""Frontend/backend contract checks for the static web UI.

These are intentionally lightweight: they run with the existing pytest stack and
catch the drift where a backend endpoint exists but the served UI no longer has
the controls or selectors that exercise it.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "index.html"
APP_JS = ROOT / "web" / "app.js"
DYNAMIC_IDS = {
    "emotionHappiness",
    "emotionStress",
    "emotionConfidence",
    "emotionCuriosity",
    "emotionReason",
    "evolutionLogText",
    "homeostasisRulesText",
}


def _ids(html: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', html))


def _selector_ids(js: str) -> set[str]:
    return set(re.findall(r"\$\(['\"]#([^'\"]+)['\"]\)", js))


def _api_literals(js: str) -> set[str]:
    return set(re.findall(r"['\"](/api/[A-Za-z0-9_/?=&{}.: -]+)", js))


def test_served_frontend_uses_current_static_files():
    client = TestClient(server.app, base_url="http://127.0.0.1")

    index = client.get("/")
    app_js = client.get("/app.js")

    assert index.status_code == 200
    assert app_js.status_code == 200
    assert index.text == INDEX.read_text()
    assert app_js.text == APP_JS.read_text()
    assert "no-cache" in index.headers["cache-control"]
    assert "no-cache" in app_js.headers["cache-control"]


def test_all_javascript_id_selectors_exist_in_html():
    html = INDEX.read_text()
    js = APP_JS.read_text()

    missing = _selector_ids(js) - _ids(html) - DYNAMIC_IDS

    assert missing == set()


def test_systems_panel_exposes_recent_backend_layers():
    html = INDEX.read_text()
    js = APP_JS.read_text()
    ids = _ids(html)

    expected_ids = {
        "systemsBtn",
        "systemsPanel",
        "observabilityBtn",
        "observabilityPanel",
        "observabilityLive",
        "observabilityTraceList",
        "integrityList",
        "homeostasisList",
        "evolutionList",
        "continuityList",
        "ctxusage",
        "ctxmeter",
        "recallBtn",
        "recallQuery",
        "summarizeCategory",
    }
    expected_functions = {
        "loadSystems",
        "repairIntegrity",
        "saveHomeostasisRules",
        "appendEvolutionLog",
        "restoreEvolutionSnapshot",
        "runRecallQuery",
        "summarizeSelectedCategory",
        "setEmotionManual",
        "loadObservability",
        "renderObservabilityTurn",
        "renderContextUsage",
    }

    assert expected_ids <= ids
    for name in expected_functions:
        assert f"function {name}" in js or f"async function {name}" in js


def test_frontend_calls_backend_routes_that_have_user_facing_workflows():
    js = APP_JS.read_text()
    literals = _api_literals(js)

    expected_prefixes = {
        "/api/chat",
        "/api/chat/cancel",
        "/api/memory",
        "/api/memories?",
        "/api/ingest",
        "/api/ingest/preview",
        "/api/recall",
        "/api/summarize",
        "/api/integrity",
        "/api/integrity/repair",
        "/api/homeostasis",
        "/api/homeostasis/rules",
        "/api/homeostasis/reset",
        "/api/evolution",
        "/api/evolution/log",
        "/api/evolution/propose",
        "/api/evolution/approve",
        "/api/evolution/reject",
        "/api/evolution/rollback",
        "/api/evolution/snapshots/restore",
        "/api/emotion",
        "/api/emotion/reset",
        "/api/awareness",
        "/api/awareness/reset",
        "/api/attention",
        "/api/self_model",
        "/api/self_model/reset",
        "/api/tools/pending",
        "/api/tools/permissions",
        "/api/tools/audit",
        "/api/workspace",
        "/api/grounding",
        "/api/presence",
        "/api/will",
        "/api/sleep",
        "/api/reflect",
        "/api/prune",
        "/api/compact",
        "/api/health",
        "/api/traces",
        "/api/timeline",
        "/api/graph",
    }

    missing = {
        prefix
        for prefix in expected_prefixes
        if not any(literal.startswith(prefix) for literal in literals)
    }
    assert missing == set()


def test_failed_approval_does_not_auto_queue_same_action_again():
    js = APP_JS.read_text()
    approve_body = js.split("async function approveAction(id)", 1)[1].split("async function rejectAction", 1)[0]

    assert "if (!executionFailed)" in approve_body
    assert "await send(continuation" in approve_body


def test_context_usage_persists_per_model_and_clear_resets_it():
    js = APP_JS.read_text()

    assert "CONTEXT_USAGE_KEY" in js
    assert "function restoreContextUsage(model)" in js
    assert "localStorage.setItem(CONTEXT_USAGE_KEY" in js
    clear_body = js.split("function clearChatState()", 1)[1].split("function handleLocalCommand", 1)[0]
    assert "clearContextUsage();" in clear_body


def test_assistant_replies_use_a_readable_measure_and_line_height():
    html = INDEX.read_text()
    js = APP_JS.read_text()

    assert ".msg.assistant .bubble { width: fit-content; max-width: min(82%, 72ch); }" in html
    assert ".msg.assistant .bubble-content { font-size: 16px; line-height: 1.72; }" in html
    assert "function formatAssistantDisplayText(text)" in js
    assert "bubble.dataset.role = role;" in js
    assert "formatAssistantDisplayText(value)" in js


def test_clear_is_local_and_purge_starts_memory_disconnected_chat():
    js = APP_JS.read_text()
    command_body = js.split("async function handleLocalCommand(text)", 1)[1].split("async function send", 1)[0]
    clear_block = command_body.split("if (cmd === '/clear')", 1)[1].split("if (cmd === '/purge')", 1)[0]
    purge_block = command_body.split("if (cmd === '/purge')", 1)[1].split("if (cmd === '/trim')", 1)[0]

    assert "clearChatState();" in clear_block
    assert "/api/memory/purge" not in clear_block
    assert "confirm(" not in purge_block
    assert "/api/" not in purge_block
    assert "sessionMemoryEnabled = false" in purge_block
    assert "clearChatState();" in purge_block
    assert "Stored Neural Map memories were not deleted" in purge_block


def test_memory_on_reconnects_fresh_session():
    js = APP_JS.read_text()
    command_body = js.split("async function handleLocalCommand(text)", 1)[1].split("async function send", 1)[0]
    memory_block = command_body.split("if (cmd === '/memory')", 1)[1].split("if (cmd === '/trim')", 1)[0]

    assert "sessionMemoryEnabled = true" in memory_block
    assert "SESSION_MEMORY_KEY" in memory_block
    assert "memory_enabled: sessionMemoryEnabled" in js


def test_missing_context_telemetry_surfaces_restart_hint():
    js = APP_JS.read_text()
    assert "function markContextUsageUnavailable()" in js
    assert "usage unavailable · restart server" in js
    assert "else if (!res.pending_decision) markContextUsageUnavailable();" in js


def test_send_commits_history_only_as_completed_pairs():
    # Regression: send() pushed the user turn at send time and the reply at
    # completion time, so overlapping requests interleaved half-turns and
    # cleanHistory() then collapsed/mispaired them.
    js = APP_JS.read_text()
    send_body = js.split("async function send(message, options = {})", 1)[1].split("function autosize()", 1)[0]

    assert "history.push" not in send_body            # no mid-flight mutation
    assert "requestHistory" in send_body              # context built as a copy
    assert "commitChatTurn(message, res.reply);" in send_body

    markstopped_body = js.split("function markStopped(bubble)", 1)[1].split("function commitChatTurn", 1)[0]
    assert "history.pop" not in markstopped_body      # nothing to roll back


def test_concurrent_sends_cannot_mispair_history():
    # Execute the real cleanHistory/commitChatTurn logic under node: two
    # overlapping sends whose replies arrive in reverse order must yield two
    # intact user/assistant pairs, not collapsed or cross-paired turns.
    if not shutil.which("node"):
        pytest.skip("node not available")
    js = APP_JS.read_text()
    logic = js[js.index("function commitChatTurn") : js.index("function restoreChatState")]
    script = "\n".join(
        [
            "let history = [];",
            "function saveHistory() {}",
            "const HISTORY_LIMIT = 12;",
            "const MAX_HISTORY_MESSAGE_CHARS = 20000;",
            logic,
            "const reqA = chatHistoryForRequest();",
            "const reqB = chatHistoryForRequest();",
            "commitChatTurn('question B', 'reply B');",  # B's reply lands first
            "commitChatTurn('question A', 'reply A');",
            "console.log(JSON.stringify({ reqA, reqB, history }));",
        ]
    )
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)

    assert out["history"] == [
        {"role": "user", "content": "question B"},
        {"role": "assistant", "content": "reply B"},
        {"role": "user", "content": "question A"},
        {"role": "assistant", "content": "reply A"},
    ]
    assert out["reqA"] == []
    assert out["reqB"] == []


def test_full_history_window_reaches_backend_without_sacrificing_a_turn():
    if not shutil.which("node"):
        pytest.skip("node not available")
    js = APP_JS.read_text()
    logic = js[js.index("function commitChatTurn") : js.index("function restoreChatState")]
    script = "\n".join(
        [
            "const HISTORY_LIMIT = 12;",
            "const MAX_HISTORY_MESSAGE_CHARS = 20000;",
            "function saveHistory() {}",
            "let history = Array.from({ length: 12 }, (_, index) => ({",
            "  role: index % 2 === 0 ? 'user' : 'assistant',",
            "  content: `prior ${index}`",
            "}));",
            logic,
            "console.log(JSON.stringify(chatHistoryForRequest()));",
        ]
    )
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    request_history = json.loads(proc.stdout)

    assert len(request_history) == 12
    assert request_history[0] == {"role": "user", "content": "prior 0"}
    assert request_history[-1] == {"role": "assistant", "content": "prior 11"}


def test_frontend_discards_oversized_reply_from_persisted_history():
    if not shutil.which("node"):
        pytest.skip("node not available")
    js = APP_JS.read_text()
    logic = js[js.index("function cleanHistory") : js.index("function restoreChatState")]
    script = "\n".join(
        [
            "const MAX_HISTORY_MESSAGE_CHARS = 20000;",
            logic,
            "const poisoned = 'Hh4e'.repeat(5001);",
            "const clean = cleanHistory([",
            "  { role: 'user', content: 'old question' },",
            "  { role: 'assistant', content: poisoned },",
            "  { role: 'user', content: 'new question' },",
            "]);",
            "console.log(JSON.stringify(clean));",
        ]
    )
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)

    assert json.loads(proc.stdout) == []


def test_action_panel_decisions_are_preserved_in_chat_history():
    js = APP_JS.read_text()

    assert "function rememberAssistantEvent(content)" in js
    approve_body = js.split("async function approveAction(id)", 1)[1].split(
        "async function rejectAction(id)", 1
    )[0]
    reject_body = js.split("async function rejectAction(id)", 1)[1].split(
        "function renderPermissions(tools)", 1
    )[0]

    assert "rememberAssistantEvent(message);" in approve_body
    assert "rememberAssistantEvent(message);" in reject_body


def test_approval_resumes_interrupted_turn_after_last_pending_action():
    js = APP_JS.read_text()
    approve_body = js.split("async function approveAction(id)", 1)[1].split(
        "async function rejectAction(id)", 1
    )[0]

    assert "const remaining = await loadPendingActions();" in approve_body
    assert "if (!remaining.length)" in approve_body
    assert "await send(continuation, { showUser: false });" in approve_body
    assert "Do not ask for this approval again" in approve_body

    load_body = js.split("async function loadPendingActions()", 1)[1].split(
        "function openActions()", 1
    )[0]
    assert "return actions;" in load_body

    send_body = js.split("async function send(message, options = {})", 1)[1].split(
        "function autosize()", 1
    )[0]
    assert "if (options.showUser !== false) addBubble('user', message);" in send_body
