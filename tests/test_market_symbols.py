"""get_stock_quotes validates symbols before they reach the provider URL.

The model can pass arbitrary strings to the tool, so fetch_quotes must reject
anything that isn't a real ticker shape instead of interpolating it into the
Yahoo request — junk is reported as an error and never triggers a network call.

Shape validation alone still leaves a covert channel: symbols ride outbound in
the provider URL, so a poisoned memory or file could have the model encode
private text as ticker-shaped chunks. The outbound-provenance guard
(market.allowed_quote_symbols + pipeline._stock_quote_refusal) closes it: only
user-typed symbols — plus the deterministic portfolio harvest for "my
holdings" questions — may be sent.
"""
from __future__ import annotations

import config
from brain import market, pipeline


def test_invalid_symbol_never_hits_the_network(monkeypatch):
    calls = []
    monkeypatch.setattr(market, "_fetch_chart_quote", lambda s: calls.append(s) or {"symbol": s})

    out = market.fetch_quotes(["AAPL", "FOO/BAR", "evil.com", "A B", ""])

    # Only the well-formed ticker reached the fetch path.
    assert calls == ["AAPL"]
    bad = {e["symbol"] for e in out["errors"]}
    assert {"FOO/BAR", "evil.com", "A B"} <= bad
    assert all(e["error"] == "invalid ticker symbol" for e in out["errors"] if e["symbol"] != "AAPL")


def test_dotted_and_dashed_tickers_are_valid(monkeypatch):
    seen = []
    monkeypatch.setattr(market, "_fetch_chart_quote", lambda s: seen.append(s) or {"symbol": s})

    market.fetch_quotes(["BRK.B", "RDS-A"])
    assert seen == ["BRK.B", "RDS-A"]


# --- outbound-provenance guard --------------------------------------------------

def test_allowed_quote_symbols_come_from_user_text_case_insensitively():
    allowed = market.allowed_quote_symbols("check aapl and $BRK.B price please")
    assert "AAPL" in allowed
    assert "BRK.B" in allowed
    assert "MSFT" not in allowed


def test_allowed_quote_symbols_harvest_memory_only_for_ownership_questions():
    mems = [{
        "text": "Portfolio holdings: SCHD1,234.567 and VTI 88 shares",
        "category": "portfolio",
        "pinned": True,
        "source": None,
    }]
    portfolio = market.allowed_quote_symbols("update my portfolio prices", mems)
    assert "SCHD" in portfolio
    assert "VTI" in portfolio
    # Without "my holdings/portfolio" phrasing, memory tickers must not ride out.
    generic = market.allowed_quote_symbols("what is the AAPL price today?", mems)
    assert "SCHD" not in generic
    assert "VTI" not in generic


def test_portfolio_harvest_rejects_uploads_and_unapproved_memories():
    mems = [
        {
            "text": "Uploaded report says SECRETLEAK",
            "category": "portfolio",
            "pinned": True,
            "source": "upload:abcd:report.txt",
        },
        {
            "text": "Generated note says XKZQ",
            "category": "portfolio",
            "pinned": False,
            "source": None,
        },
    ]

    allowed = market.allowed_quote_symbols("update my portfolio prices", mems)

    assert "SECRETLEAK" not in allowed
    assert "XKZQ" not in allowed


def _run_tool_turn(monkeypatch, user_text, mems, tool_calls):
    """Run pipeline.answer with a fake tool loop that performs tool_calls and
    records each dispatch result."""
    results = []

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        for name, args in tool_calls:
            results.append(dispatch(name, args))
        return "ok", []

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_needs_financial_prefetch", lambda *a, **k: False)
    pipeline.answer(user_text, mems)
    return results


def test_dispatch_refuses_memory_suggested_symbol_but_allows_user_named(monkeypatch):
    fetched = []
    monkeypatch.setattr(market, "_fetch_chart_quote", lambda s: fetched.append(s) or {"symbol": s, "price": 1.0})

    mems = [{"id": "m1", "text": "note to self: also check SECR quote", "source": None}]
    refused, allowed = _run_tool_turn(
        monkeypatch,
        "what is the VTI price today?",
        mems,
        [("get_stock_quotes", {"tickers": ["SECR"]}),
         ("get_stock_quotes", {"tickers": ["VTI"]})],
    )
    assert "SECR" in refused["error"]           # memory-suggested symbol never leaves
    assert fetched == ["VTI"]                    # only the user-named ticker went out
    assert allowed["quotes"][0]["symbol"] == "VTI"


def test_dispatch_allows_portfolio_harvest_for_ownership_questions(monkeypatch):
    fetched = []
    monkeypatch.setattr(market, "_fetch_chart_quote", lambda s: fetched.append(s) or {"symbol": s, "price": 1.0})

    mems = [{
        "id": "m1",
        "text": "Portfolio holdings: SCHD1,234.567",
        "category": "portfolio",
        "pinned": True,
        "source": None,
    }]
    (result,) = _run_tool_turn(
        monkeypatch,
        "update my portfolio prices",
        mems,
        [("get_stock_quotes", {"tickers": ["SCHD"]})],
    )
    assert not result.get("error")
    assert fetched == ["SCHD"]


def test_dispatch_allows_pinned_holdings_note_for_portfolio_analysis(monkeypatch):
    fetched = []
    monkeypatch.setattr(market, "_fetch_chart_quote", lambda s: fetched.append(s) or {"symbol": s, "price": 1.0})

    mems = [{
        "id": "m1",
        "text": "I currently have 1175.808 shares of SCHD, 59.421 shares of FXAIX, 69.156 shares of SGOV, 10 shares of SPCX",
        "category": "note",
        "pinned": True,
        "source": None,
    }]
    (result,) = _run_tool_turn(
        monkeypatch,
        "lets do a portfolio analysis",
        mems,
        [("get_stock_quotes", {"tickers": ["SCHD", "FXAIX", "SGOV", "SPCX"]})],
    )
    assert not result.get("error")
    assert fetched == ["SCHD", "FXAIX", "SGOV", "SPCX"]


def test_dispatch_still_refuses_generic_pinned_note_for_portfolio_analysis(monkeypatch):
    fetched = []
    monkeypatch.setattr(market, "_fetch_chart_quote", lambda s: fetched.append(s) or {"symbol": s, "price": 1.0})

    mems = [{
        "id": "m1",
        "text": "Research note: compare SCHD with SECR next quarter.",
        "category": "note",
        "pinned": True,
        "source": None,
    }]
    (result,) = _run_tool_turn(
        monkeypatch,
        "lets do a portfolio analysis",
        mems,
        [("get_stock_quotes", {"tickers": ["SECR"]})],
    )
    assert "SECR" in result["error"]
    assert fetched == []


def test_trusted_finance_file_tickers_are_available_for_portfolio_analysis():
    finance = """# Finance\n\n| SCHD shares | 1,175.808 |\n| FXAIX shares | 59.421 |\n| SGOV shares | 69.156 |\n| SPCX shares | 10 |\n"""

    allowed = market.allowed_quote_symbols(
        "lets do a portfolio analysis",
        [],
        trusted_portfolio_text=finance,
    )

    assert {"SCHD", "FXAIX", "SGOV", "SPCX"} <= allowed


def test_tool_path_prefetches_finance_file_quotes_for_portfolio_analysis(monkeypatch, tmp_path):
    finance_path = tmp_path / "finance.md"
    finance_path.write_text(
        "# Finance\n\n## Confirmed Financial Facts\n\n"
        "| SCHD shares | 1,175.808 |\n| FXAIX shares | 59.421 |\n"
        "| SGOV shares | 69.156 |\n| SPCX shares | 10 |\n\n"
        "## Unverified or Unresolved\n\n- Whether SHLD and FZROX are still held.",
        encoding="utf-8",
    )
    fetched = []
    captured = {}

    def fake_fetch(symbol):
        fetched.append(symbol)
        return {"symbol": symbol, "price": 100.0}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        return "grounded", []

    monkeypatch.setattr(config, "FINANCE_PATH", finance_path)
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(market, "_fetch_chart_quote", fake_fetch)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)

    result = pipeline.answer("lets do a portfolio analysis", [])
    names = {spec["function"]["name"] for spec in captured["specs"]}

    assert fetched == ["SCHD", "FXAIX", "SGOV", "SPCX"]
    assert result["live_data"]["quotes"]
    assert "SCHD" in captured["system"]
    assert "get_stock_quotes" not in names


def test_finance_quote_context_excludes_unresolved_symbols(monkeypatch, tmp_path):
    finance_path = tmp_path / "finance.md"
    finance_path.write_text(
        "# Finance\n\n## Confirmed Financial Facts\n\n| SCHD shares | 12 |\n\n"
        "## Unverified or Unresolved\n\n- Whether SHLD and FZROX are still held.",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "FINANCE_PATH", finance_path)

    text = pipeline._trusted_finance_text_for_quotes("portfolio analysis")
    tickers = market.tickers_from_memories([], trusted_portfolio_text=text)

    assert tickers == ["SCHD"]


def test_market_context_preserves_provider_errors_for_trusted_portfolio(monkeypatch):
    monkeypatch.setattr(
        market,
        "fetch_quotes",
        lambda symbols: {
            "provider": "test",
            "quotes": [],
            "errors": [{"symbols": symbols, "error": "network unavailable"}],
        },
    )

    result = market.market_context_for(
        "portfolio analysis",
        trusted_portfolio_text="| SCHD shares | 12 |",
    )

    assert result["errors"][0]["error"] == "network unavailable"
    assert result["source"] == "portfolio tickers resolved from trusted finance calibration"
