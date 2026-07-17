"""Small live market-data helper for quote-style questions.

This is intentionally modest: it detects ticker-looking symbols in a user
message, fetches quotes from Yahoo Finance's public quote endpoint, and returns
structured data for prompt injection. The model never calls tools itself.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

import config

# A real ticker is short and uses only letters, digits, dot and dash. The model
# can pass arbitrary strings to the get_stock_quotes tool, so validate before the
# symbol is interpolated into the provider URL — junk can't reshape the request.
_VALID_SYMBOL = re.compile(r"^[A-Z0-9.\-]{1,12}$")

_QUOTE_TERMS = re.compile(
    r"\b(price|prices|quote|quotes|market|markets|ticker|tickers|stock|stocks|etf|etfs|fund|funds|holding|holdings|portfolio|position|positions|share|shares|owned|owns|value|values)\b",
    re.IGNORECASE,
)
_TICKER = re.compile(r"\$?[A-Z][A-Z0-9.-]{1,9}\b")
_COMMON_WORDS = {
    "AI",
    "API",
    "CPU",
    "ETF",
    "ETFS",
    "FAQ",
    "IRA",
    "JSON",
    "LLM",
    "PDF",
    "README",
    "SSA",
    "SSDI",
    "UI",
    "URL",
    "US",
    "USA",
    "VA",
}

# When harvesting symbols from free-text memories (a portfolio chunk, prior
# conversation) the prose is noisier than a direct question, so we filter a
# wider set of finance/benefits acronyms that are not tradable tickers.
_MEMORY_STOPWORDS = _COMMON_WORDS | {
    "BTC", "CBOE", "CDC", "CPI", "DCA", "EOD", "ER", "ETN", "FED", "GDP",
    "GTC", "HOA", "HSA", "IPO", "IRS", "LTCG", "MGIB", "NIH", "RMD", "RN",
    "ROTH", "TWP", "YTD",
}

# "my portfolio", "our holdings", "my assets", etc. — gates the memory harvest
# so we only guess tickers when the user is clearly asking about THEIR holdings.
_OWNERSHIP_TERMS = re.compile(
    r"\b(my|our)\b.{0,30}\b(portfolio|holdings|positions?|assets?|investments?|stocks?|etfs?|funds?)\b",
    re.IGNORECASE,
)

# Memory-derived symbols are an explicit outbound-data exception.  A generic
# retrieved fact or uploaded document is not sufficient authorization: the user
# must have pinned a source-free memory in a portfolio-specific category.  This
# makes the exception inspectable and prevents a coincidentally retrieved (or
# prompt-injected) document from smuggling ticker-shaped strings to the quote
# provider.
_PORTFOLIO_CATEGORY_TERMS = {
    "portfolio",
    "holding",
    "holdings",
    "position",
    "positions",
    "investment",
    "investments",
}


def _approved_portfolio_memory(memory: dict) -> bool:
    if not isinstance(memory, dict) or not memory.get("pinned") or memory.get("source"):
        return False
    category_terms = set(re.findall(r"[a-z0-9]+", str(memory.get("category") or "").casefold()))
    return bool(category_terms & _PORTFOLIO_CATEGORY_TERMS)


def quote_request_tickers(text: str) -> list[str]:
    """Return ticker-like symbols only when the message appears quote-related."""
    if not config.MARKET_QUOTES_ENABLED or not _QUOTE_TERMS.search(text):
        return []
    symbols = []
    for raw in _TICKER.findall(text):
        symbol = raw.lstrip("$").strip(".").upper()
        if symbol in _COMMON_WORDS or symbol.isdigit():
            continue
        if symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= config.MAX_QUOTE_TICKERS:
            break
    return symbols


def tickers_from_memories(memories: list[dict] | None, limit: int | None = None) -> list[str]:
    """Harvest ticker-like symbols out of retrieved memory text.

    Used when the user asks about *their* holdings ("update my portfolio
    prices") without naming symbols. Only pinned, source-free memories in an
    explicit portfolio/holdings/positions/investments category qualify; an
    arbitrary retrieved upload or generated fact is not outbound authorization.
    Portfolio text often concatenates symbols with figures (e.g.
    "SCHD123.456"), so we strip trailing digits off each candidate and drop a
    wide set of non-ticker acronyms.
    """
    if not config.MARKET_QUOTES_ENABLED or not memories:
        return []
    limit = config.MAX_QUOTE_TICKERS if limit is None else limit
    approved = [m for m in memories if _approved_portfolio_memory(m)]
    if not approved:
        return []
    blob = "\n".join((m.get("text") or "") for m in approved)
    symbols: list[str] = []
    for raw in _TICKER.findall(blob):
        symbol = re.sub(r"\d+$", "", raw.lstrip("$").strip(".").upper())
        if len(symbol) < 2 or symbol in _MEMORY_STOPWORDS or symbol.isdigit():
            continue
        if symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


# Ticker-shaped tokens in user-authored text, matched case-insensitively:
# people type "aapl" as often as "AAPL", and a token the user actually typed is
# safe to send outbound regardless of its casing.
_USER_TOKEN = re.compile(r"\$?[A-Za-z][A-Za-z0-9.\-]{0,11}")


def allowed_quote_symbols(user_text: str, memories: list[dict] | None = None) -> set[str]:
    """Symbols a get_stock_quotes call may send to the quote provider.

    Tool arguments are interpolated into the provider URL, so they leave the
    machine just like a web_search query does — a model-composed symbol is a
    covert channel through which a poisoned memory or ingested file can encode
    private text as ticker-shaped chunks. Mirror the web_search derivation
    guard: a symbol is sendable only if the user typed it, or — for the same
    quote+ownership phrasing that unlocks the eager path ("update my portfolio
    prices") — it comes from the approved portfolio-memory harvest above. The
    model can then only *select* among tokens that already had a sanctioned path
    outbound; it cannot mint new ones.
    """
    text = user_text or ""
    allowed: set[str] = set()
    for raw in _USER_TOKEN.findall(text):
        symbol = raw.lstrip("$").strip(".").upper()
        if symbol:
            allowed.add(symbol)
    if _QUOTE_TERMS.search(text) and _OWNERSHIP_TERMS.search(text):
        # Broader cap than one eager request: the allowlist spans a whole tool
        # loop, while each outbound call is still capped at MAX_QUOTE_TICKERS
        # by the get_stock_quotes handler itself.
        allowed.update(tickers_from_memories(memories, limit=4 * config.MAX_QUOTE_TICKERS))
    return allowed


def _fetch_chart_quote(symbol: str) -> dict:
    query_symbol = quote(symbol.replace(".", "-"), safe="")
    request = Request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{query_symbol}?range=1d&interval=1m",
        headers={"User-Agent": "Mozilla/5.0 NeuralMap/1.0"},
    )
    with urlopen(request, timeout=config.MARKET_QUOTES_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(error.get("description") or error.get("code") or "quote provider error")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("not returned by quote provider")
    meta = results[0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    previous = meta.get("chartPreviousClose") or meta.get("previousClose")
    change = price - previous if isinstance(price, (int, float)) and isinstance(previous, (int, float)) else None
    change_percent = change / previous * 100 if change is not None and previous else None
    return {
        "symbol": meta.get("symbol", symbol),
        "name": meta.get("shortName") or meta.get("longName"),
        "price": price,
        "change": change,
        "change_percent": change_percent,
        "currency": meta.get("currency"),
        "market_state": meta.get("marketState"),
        "market_time": meta.get("regularMarketTime") or meta.get("firstTradeDate"),
        "exchange": meta.get("exchangeName"),
        "instrument_type": meta.get("instrumentType"),
    }


def fetch_quotes(symbols: list[str]) -> dict:
    if not symbols:
        return {"as_of": datetime.now(timezone.utc).isoformat(), "quotes": [], "errors": []}
    quotes = []
    errors = []
    for symbol in symbols:
        if not _VALID_SYMBOL.match(symbol):
            errors.append({"symbol": symbol, "error": "invalid ticker symbol"})
            continue
        try:
            quotes.append(_fetch_chart_quote(symbol))
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance chart",
        "quotes": quotes,
        "errors": errors,
    }


def market_context_for(text: str, memories: list[dict] | None = None) -> dict | None:
    symbols = quote_request_tickers(text)
    from_memory = False
    # No tickers typed, but the user is asking about THEIR holdings — pull the
    # symbols from memory (that's what the memory is for) and validate them.
    if not symbols and _QUOTE_TERMS.search(text) and _OWNERSHIP_TERMS.search(text):
        symbols = tickers_from_memories(memories)
        from_memory = True
    if not symbols:
        return None
    try:
        data = fetch_quotes(symbols)
    except Exception as exc:
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "provider": "Yahoo Finance",
            "quotes": [],
            "errors": [{"symbols": symbols, "error": str(exc)}],
        }
    if from_memory:
        # Symbols were guessed from memory, so keep only the ones the provider
        # confirmed and don't surface guess-misses as if the user asked for them.
        data["quotes"] = [q for q in data["quotes"] if q.get("price") is not None]
        data["errors"] = []
        data["source"] = "portfolio tickers resolved from memory"
        if not data["quotes"]:
            return None
    return data
