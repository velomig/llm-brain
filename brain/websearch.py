"""Generic web-search helper for current/external-information questions."""
from __future__ import annotations

from datetime import datetime, timezone
import base64
from html import unescape
from html.parser import HTMLParser
import re
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import config

_SEARCH_TERMS = re.compile(
    r"\b("
    r"search|look\s*up|google|web|internet|online|browse|find\s+out|research|verify\s+online|check\s+online|"
    r"latest|current|today|recent|newest|up[- ]?to[- ]?date|live|"
    r"news|price|prices|quote|quotes|weather|forecast|schedule|score|scores|"
    r"who\s+is|who's|what\s+is\s+the\s+latest|when\s+is|where\s+is"
    r")\b",
    re.IGNORECASE,
)
_CAPABILITY_TERMS = re.compile(
    r"\b("
    r"can\s+you\s+(access|browse|reach|search|use).{0,30}(web|website|websites|internet|online)|"
    r"do\s+you\s+have.{0,30}(web|internet|browser|browsing|search)|"
    r"are\s+you\s+able\s+to.{0,30}(browse|search|access).{0,30}(web|internet|website|websites)"
    r")\b",
    re.IGNORECASE,
)
_NEWS_TERMS = re.compile(r"\b(news|headline|headlines|today|latest|current|recent)\b", re.IGNORECASE)
_FINANCE_TERMS = re.compile(
    r"\b(?:financ(?:e|es|ial)|portfolio|retirement|invest(?:ing|ment|ments)?|"
    r"stocks?|etfs?|funds?|holdings?|positions?|tax(?:es|ation)?|"
    r"fndx|fndy|fndz|acme|exco|coin|bitcoin)\b",
    re.IGNORECASE,
)
_ANALYSIS_TERMS = re.compile(
    r"\b(?:analysis|analy[sz]e|review|assess|evaluate|valuation|allocation|"
    r"rebalance|performance|projection|forecast|scenario|compare|plan(?:ning)?)\b",
    re.IGNORECASE,
)
_QUERY_PREFIXES = (
    re.compile(r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?", re.IGNORECASE),
    re.compile(r"^please\s+", re.IGNORECASE),
    re.compile(r"^i(?:'d|\s+would)\s+like\s+(?:you\s+)?(?:to\s+)?", re.IGNORECASE),
    re.compile(r"^(?:get|give|show|tell|find)\s+(?:me|us)\s+", re.IGNORECASE),
    re.compile(
        r"^(?:search|browse|google|research|look\s*up|find\s+out|check\s+online|verify\s+online)"
        r"\s+(?:(?:the\s+)?(?:web|internet|online)\s+)?(?:for\s+)?",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:what(?:'s|\s+is|\s+are)|who\s+is|where\s+is|when\s+is)\s+(?:the\s+)?", re.IGNORECASE),
)
_RELEVANCE_STOPWORDS = {
    "about", "and", "browse", "can", "check", "current", "find", "for", "from", "get",
    "give", "google", "internet", "latest", "look", "lookup", "me", "news", "online",
    "please", "recent", "research", "search", "show", "tell", "the", "today", "verify",
    "web", "what", "when", "where", "who", "would", "you",
}


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._current: dict | None = None
        self._capture_title = False
        self._capture_snippet = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "a" and "result-link" in classes:
            self._current = {"title": "", "url": _clean_duck_url(attrs_dict.get("href", "")), "snippet": ""}
            self._capture_title = True
            self._buffer = []
        elif self._current is not None and tag in {"td", "div", "span"} and (
            "result-snippet" in classes or "result__snippet" in classes
        ):
            self._capture_snippet = True
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture_title or self._capture_snippet:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture_title and tag == "a" and self._current is not None:
            self._current["title"] = _clean_text(" ".join(self._buffer))
            if self._current["title"] and self._current["url"]:
                self.results.append(self._current)
            self._capture_title = False
            self._buffer = []
        elif self._capture_snippet and tag in {"td", "div", "span"} and self._current is not None:
            self._current["snippet"] = _clean_text(" ".join(self._buffer))
            self._capture_snippet = False
            self._buffer = []


class _BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._current: dict | None = None
        self._li_depth = 0
        self._capture_title = False
        self._capture_snippet = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "li" and "b_algo" in classes:
            self._current = {"title": "", "url": "", "snippet": ""}
            self._li_depth = 1
            self._capture_title = False
            self._capture_snippet = False
            self._buffer = []
            return
        if self._current is None:
            return
        if tag == "li":
            self._li_depth += 1
        if tag == "h2":
            self._capture_title = True
            self._buffer = []
        elif self._capture_title and tag == "a" and not self._current.get("url"):
            self._current["url"] = _clean_bing_url(attrs_dict.get("href", ""))
        elif tag == "p" and not self._current.get("snippet"):
            self._capture_snippet = True
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture_title or self._capture_snippet:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if self._capture_title and tag == "h2":
            self._current["title"] = _clean_text(" ".join(self._buffer))
            self._capture_title = False
            self._buffer = []
        elif self._capture_snippet and tag == "p":
            self._current["snippet"] = _clean_text(" ".join(self._buffer))
            self._capture_snippet = False
            self._buffer = []
        elif tag == "li":
            self._li_depth -= 1
            if self._li_depth <= 0:
                if self._current.get("title") and self._current.get("url"):
                    self.results.append(self._current)
                self._current = None
                self._buffer = []


class _BingNewsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._current: dict | None = None
        self._div_depth = 0
        self._capture_snippet = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "div" and "news-card" in classes and "newsitem" in classes:
            self._current = {
                "title": _clean_text(attrs_dict.get("data-title") or attrs_dict.get("title") or ""),
                "url": attrs_dict.get("data-url") or attrs_dict.get("url") or "",
                "snippet": "",
                "source": _clean_text(attrs_dict.get("data-author") or ""),
            }
            self._div_depth = 1
            self._capture_snippet = False
            self._buffer = []
            return
        if self._current is None:
            return
        if tag == "div":
            self._div_depth += 1
            if "snippet" in classes and not self._current.get("snippet"):
                self._capture_snippet = True
                self._buffer = []
        elif tag == "a" and "title" in classes:
            self._current["url"] = attrs_dict.get("href") or self._current.get("url") or ""

    def handle_data(self, data: str) -> None:
        if self._capture_snippet:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if self._capture_snippet and tag == "div":
            self._current["snippet"] = _clean_text(" ".join(self._buffer))
            self._capture_snippet = False
            self._buffer = []
        if tag == "div":
            self._div_depth -= 1
            if self._div_depth <= 0:
                if self._current.get("title") and self._current.get("url"):
                    self.results.append(self._current)
                self._current = None
                self._buffer = []


class _YahooParser(HTMLParser):
    """Parse Yahoo's ordinary result cards, ignoring its generated summary."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._current: dict | None = None
        self._depth = 0
        self._capture_title = False
        self._capture_snippet = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "div" and "dd" in classes and "algo" in classes:
            self._current = {"title": "", "url": "", "snippet": ""}
            self._depth = 1
            return
        if self._current is None:
            return
        if tag == "div":
            self._depth += 1
        if tag == "a" and attrs_dict.get("data-matarget") == "algo" and not self._current["url"]:
            self._current["url"] = _clean_yahoo_url(attrs_dict.get("href", ""))
        elif tag == "h3" and not self._current["title"]:
            self._capture_title = True
            self._buffer = []
        elif tag == "p" and not self._current["snippet"]:
            self._capture_snippet = True
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture_title or self._capture_snippet:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if self._capture_title and tag == "h3":
            self._current["title"] = _clean_text(" ".join(self._buffer))
            self._capture_title = False
            self._buffer = []
        elif self._capture_snippet and tag == "p":
            self._current["snippet"] = _clean_text(" ".join(self._buffer))
            self._capture_snippet = False
            self._buffer = []
        if tag == "div":
            self._depth -= 1
            if self._depth <= 0:
                if self._current["title"] and self._current["url"]:
                    self.results.append(self._current)
                self._current = None
                self._buffer = []


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(text))).strip()


def _clean_duck_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else url
    return url


def _clean_bing_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(unescape(url))
    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/"):
        target = parse_qs(parsed.query).get("u", [""])[0]
        if target.startswith("a1"):
            encoded = target[2:]
            padding = "=" * (-len(encoded) % 4)
            try:
                return base64.urlsafe_b64decode((encoded + padding).encode("ascii")).decode("utf-8")
            except Exception:
                return url
    return url


def _clean_yahoo_url(url: str) -> str:
    if not url:
        return ""
    match = re.search(r"/RU=([^/]+)/RK=", url)
    return unquote(match.group(1)) if match else url


def search_query_for(text: str) -> str:
    """Turn a polite chat request into the topic string search engines expect."""
    query = " ".join((text or "").split()).strip()
    query = re.sub(r"^/search\s*", "", query, count=1, flags=re.IGNORECASE).strip()
    # Prefixes are intentionally applied repeatedly: "Can you please get me..."
    # contains two layers, both of which should disappear.
    changed = True
    while changed and query:
        changed = False
        for prefix in _QUERY_PREFIXES:
            reduced = prefix.sub("", query, count=1).strip()
            if reduced != query:
                query = reduced
                changed = True
                break
    query = re.sub(r"\bfor\s+today\b", "today", query, flags=re.IGNORECASE)
    return query.strip(" \t\r\n?!.,:;")


def _relevance_terms(query: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[a-z0-9][a-z0-9.+-]*", query, re.IGNORECASE)
        if term.lower() not in _RELEVANCE_STOPWORDS
    }


def _relevant_results(items: list[dict], query: str, max_results: int) -> list[dict]:
    terms = _relevance_terms(query)
    if not terms:
        return _dedupe_results(items, max_results)
    required = 1 if len(terms) <= 2 else 2
    relevant = []
    for item in items:
        haystack = " ".join(str(item.get(k) or "") for k in ("title", "snippet", "source", "url")).lower()
        if sum(1 for term in terms if term in haystack) >= required:
            relevant.append(item)
    return _dedupe_results(relevant, max_results)


def _looks_like_duck_challenge(html: str) -> bool:
    lowered = html.lower()
    return "anomaly.js" in lowered or "challenge-form" in lowered or "botnet" in lowered


def _dedupe_results(items: list[dict], max_results: int) -> list[dict]:
    results = []
    seen = set()
    for item in items:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(item)
        if len(results) >= max_results:
            break
    return results


def _rss_items(xml_data: bytes) -> list[dict]:
    root = ET.fromstring(xml_data)
    items = []
    for node in root.findall(".//item"):
        title = _clean_text(node.findtext("title") or "")
        url = _clean_bing_url(node.findtext("link") or "")
        snippet = _clean_text(node.findtext("description") or "")
        source = _clean_text(node.findtext("source") or "")
        item = {"title": title, "url": url, "snippet": snippet}
        if source:
            item["source"] = source
        if title and url:
            items.append(item)
    return items


def _google_news_results(query: str, max_results: int, errors: list[dict]) -> list[dict]:
    params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    request = Request(
        f"https://news.google.com/rss/search?{params}",
        headers={"User-Agent": "Mozilla/5.0 NeuralMap/1.0"},
    )
    try:
        with urlopen(request, timeout=config.WEB_SEARCH_TIMEOUT) as response:
            results = _relevant_results(_rss_items(response.read()), query, max_results)
        if not results:
            errors.append({"provider": "Google News", "error": "provider returned no relevant results"})
        return results
    except Exception as exc:
        errors.append({"provider": "Google News", "error": str(exc)})
        return []


def _yahoo_results(query: str, max_results: int, errors: list[dict], *, news: bool = False) -> list[dict]:
    params = urlencode({"p": query})
    base = "https://news.search.yahoo.com/search" if news else "https://search.yahoo.com/search"
    provider = "Yahoo News" if news else "Yahoo"
    request = Request(
        f"{base}?{params}",
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"},
    )
    try:
        with urlopen(request, timeout=config.WEB_SEARCH_TIMEOUT) as response:
            html = response.read().decode("utf-8", errors="replace")
        parser = _YahooParser()
        parser.feed(html)
        results = _relevant_results(parser.results, query, max_results)
        if not results:
            errors.append({"provider": provider, "error": "provider returned no relevant results"})
        return results
    except Exception as exc:
        errors.append({"provider": provider, "error": str(exc)})
        return []


def _bing_news_results(query: str, max_results: int, errors: list[dict]) -> list[dict]:
    params = urlencode({"q": query, "setlang": "en-US", "mkt": "en-US", "format": "rss"})
    request = Request(
        f"https://www.bing.com/news/search?{params}",
        headers={"User-Agent": "Mozilla/5.0 NeuralMap/1.0"},
    )
    try:
        with urlopen(request, timeout=config.WEB_SEARCH_TIMEOUT) as response:
            data = response.read()
        try:
            items = _rss_items(data)
        except ET.ParseError:
            items = []
        if not items:
            parser = _BingNewsParser()
            parser.feed(data.decode("utf-8", errors="replace"))
            items = parser.results
        results = _relevant_results(items, query, max_results)
        if not results:
            errors.append({"provider": "Bing News", "error": "provider returned no relevant results"})
        return results
    except Exception as exc:
        errors.append({"provider": "Bing News", "error": str(exc)})
        return []


def _bing_web_results(query: str, max_results: int, errors: list[dict]) -> list[dict]:
    params = urlencode({"q": query, "setlang": "en-US", "mkt": "en-US", "format": "rss"})
    request = Request(
        f"https://www.bing.com/search?{params}",
        headers={"User-Agent": "Mozilla/5.0 NeuralMap/1.0"},
    )
    try:
        with urlopen(request, timeout=config.WEB_SEARCH_TIMEOUT) as response:
            data = response.read()
        try:
            items = _rss_items(data)
        except ET.ParseError:
            items = []
        if not items:
            parser = _BingParser()
            parser.feed(data.decode("utf-8", errors="replace"))
            items = parser.results
        results = _relevant_results(items, query, max_results)
        if not results:
            errors.append({"provider": "Bing", "error": "provider returned no relevant results"})
        return results
    except Exception as exc:
        errors.append({"provider": "Bing", "error": str(exc)})
        return []


def should_search(text: str) -> bool:
    if not config.WEB_SEARCH_ENABLED or config.WEB_SEARCH_MODE == "off":
        return False
    if config.WEB_SEARCH_MODE == "always":
        return bool(text.strip())
    return bool(_SEARCH_TERMS.search(text)) or bool(
        _FINANCE_TERMS.search(text) and _ANALYSIS_TERMS.search(text)
    )


def is_web_capability_question(text: str) -> bool:
    match = _CAPABILITY_TERMS.search(text)
    if not match:
        return False
    # "Can you search the web?" is a capability question; "Can you search the
    # web for X?" is an actual search request. Only harmless meta words may
    # follow the matched capability phrase.
    tail_terms = {
        term.lower()
        for term in re.findall(r"[a-z0-9]+", (text or "")[match.end():], re.IGNORECASE)
    }
    capability_tail = {"access", "available", "capabilities", "capability", "currently", "enabled", "now"}
    return not tail_terms or tail_terms <= capability_tail


def search_web(query: str, max_results: int | None = None) -> dict:
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    requested_query = query
    query = search_query_for(query)
    if not query:
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "provider": "none",
            "query": "",
            "requested_query": requested_query,
            "results": [],
            "errors": [{"error": "search query has no topic"}],
        }
    errors = []

    # Prefer structured news RSS and Yahoo's semantically stronger general-web
    # results. DuckDuckGo and Bing remain fallbacks instead of being allowed to
    # inject unrelated dictionary pages into the answer.
    if _NEWS_TERMS.search(query):
        results = _google_news_results(query, max_results, errors)
        if results:
            return {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "provider": "Google News",
                "query": query,
                "requested_query": requested_query,
                "results": results,
                "errors": errors,
            }
        results = _yahoo_results(query, max_results, errors, news=True)
        if results:
            return {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "provider": "Yahoo News",
                "query": query,
                "requested_query": requested_query,
                "results": results,
                "errors": errors,
            }
    else:
        results = _yahoo_results(query, max_results, errors)
        if results:
            return {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "provider": "Yahoo",
                "query": query,
                "requested_query": requested_query,
                "results": results,
                "errors": errors,
            }

    params = urlencode({"q": query, "kl": "us-en"})
    request = Request(
        f"https://lite.duckduckgo.com/lite/?{params}",
        headers={"User-Agent": "Mozilla/5.0 NeuralMap/1.0"},
    )
    try:
        with urlopen(request, timeout=config.WEB_SEARCH_TIMEOUT) as response:
            html = response.read().decode("utf-8", errors="replace")
        if _looks_like_duck_challenge(html):
            errors.append({"provider": "DuckDuckGo", "error": "provider returned a bot/anomaly challenge"})
        else:
            parser = _DuckDuckGoParser()
            parser.feed(html)
            results = _relevant_results(parser.results, query, max_results)
            if results:
                return {
                    "as_of": datetime.now(timezone.utc).isoformat(),
                    "provider": "DuckDuckGo",
                    "query": query,
                    "requested_query": requested_query,
                    "results": results,
                    "errors": errors,
                }
            errors.append({"provider": "DuckDuckGo", "error": "provider returned no parseable results"})
    except Exception as exc:
        errors.append({"provider": "DuckDuckGo", "error": str(exc)})

    provider = "Bing"
    results = []
    if _NEWS_TERMS.search(query):
        results = _bing_news_results(query, max_results, errors)
        if results:
            provider = "Bing News"
    if not results:
        results = _bing_web_results(query, max_results, errors)
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "query": query,
        "requested_query": requested_query,
        "results": results,
        "errors": errors,
    }


def eager_query_for(text: str) -> str | None:
    """Reduce a user message to a query safe to send eagerly, or None to skip.

    Unlike the web_search tool path, eager injection has no model-derived query
    and no dispatch-time refusal — the raw prompt IS the query. A prompt can be
    thousands of characters of pasted private notes that merely contain a word
    like "latest", so only short, typed-question-shaped messages go out verbatim;
    everything else stays on the box (tool-capable models can still search via
    the guarded web_search tool).
    """
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if len(non_empty_lines) > config.MAX_EAGER_WEB_QUERY_LINES:
        return None   # multi-line block: looks like pasted context, not a query
    query = " ".join(text.split())
    if not query or len(query) > config.MAX_EAGER_WEB_QUERY_CHARS:
        return None
    # /search is an explicit UI shortcut. Send only its payload to the public
    # provider, without the command syntax itself.
    if re.match(r"^/search(?:\s|$)", query, re.IGNORECASE):
        query = re.sub(r"^/search\s*", "", query, count=1, flags=re.IGNORECASE).strip()
        if not query:
            return None
    return query


def web_context_for(text: str) -> dict | None:
    if is_web_capability_question(text):
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "provider": "Neural Map",
            "query": text,
            "capability": {
                "web_search_enabled": config.WEB_SEARCH_ENABLED,
                "web_search_mode": config.WEB_SEARCH_MODE,
                "description": "This app can make live web search requests before the model answers and provide the results as Web Search JSON.",
                "limits": "The model cannot autonomously click around or make extra requests mid-answer; it uses the web/search data the app provides before generation.",
            },
            "results": [],
            "errors": [],
        }
    if not should_search(text):
        return None
    query = eager_query_for(text)
    if query is None:
        return None
    try:
        return search_web(query)
    except Exception as exc:
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "provider": "DuckDuckGo",
            "query": query,
            "results": [],
            "errors": [{"error": str(exc)}],
        }
