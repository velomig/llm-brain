from __future__ import annotations

from brain import websearch


class _Response:
    def __init__(self, html: str) -> None:
        self.html = html

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.html.encode("utf-8")


def test_news_search_cleans_request_and_uses_structured_rss(monkeypatch):
    rss = """<?xml version="1.0"?>
    <rss><channel><item>
      <title>US political headline today</title>
      <link>https://news.example/politics</link>
      <description>Current US political reporting.</description>
      <source>Example News</source>
    </item></channel></rss>"""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        assert "news.google.com/rss/search" in request.full_url
        return _Response(rss)

    monkeypatch.setattr(websearch, "urlopen", fake_urlopen)

    out = websearch.search_web("Can you please get me US political news for today?")

    assert out["provider"] == "Google News"
    assert out["query"] == "US political news today"
    assert out["requested_query"] == "Can you please get me US political news for today?"
    assert out["results"][0]["title"] == "US political headline today"


def test_general_search_uses_relevant_yahoo_results(monkeypatch):
    html = """
    <div class="dd algo algo-sr">
      <div class="compTitle options-toggle">
        <a data-matarget="algo" href="https://example.com/plants">
          <h3><span>Native shade plants for Phoenix</span></h3>
        </a>
      </div>
      <div class="compText"><p>Heat-tolerant native plants for a Phoenix shade garden.</p></div>
    </div>
    """

    def fake_urlopen(request, timeout=None):
        assert "search.yahoo.com/search" in request.full_url
        return _Response(html)

    monkeypatch.setattr(websearch, "urlopen", fake_urlopen)

    out = websearch.search_web("/search native plants for a Phoenix shade garden")

    assert out["provider"] == "Yahoo"
    assert out["query"] == "native plants for a Phoenix shade garden"
    assert out["results"][0]["url"] == "https://example.com/plants"


def test_search_request_is_not_misclassified_as_capability_question(monkeypatch):
    captured = {}

    def fake_search(query, max_results=None):
        captured["query"] = query
        return {"query": query, "results": [{"title": "result"}], "errors": []}

    monkeypatch.setattr(websearch, "search_web", fake_search)

    assert websearch.is_web_capability_question("Can you search the web?") is True
    assert websearch.is_web_capability_question("Do you have internet access?") is True
    assert websearch.is_web_capability_question("Can you search the web for native plants?") is False
    out = websearch.web_context_for("Can you search the web for native plants?")
    assert out["results"] == [{"title": "result"}]
    assert captured["query"] == "Can you search the web for native plants?"


def test_search_web_falls_back_to_bing_news_after_duck_challenge(monkeypatch):
    duck_challenge = '<form id="challenge-form" action="//duckduckgo.com/anomaly.js?cc=botnet"></form>'
    bing_news = """
    <html><body>
      <div class="news-card newsitem cardcommon"
        data-url="https://news.example/current"
        data-title="Current headline about President Trump"
        data-author="Example News">
        <div class="caption">
          <div class="snippet">A current named news item was found.</div>
        </div>
      </div>
    </body></html>
    """

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        if "duckduckgo.com" in url:
            return _Response(duck_challenge)
        if "/news/search" in url:
            return _Response(bing_news)
        raise AssertionError(f"unexpected fallback URL: {url}")

    monkeypatch.setattr(websearch, "urlopen", fake_urlopen)

    out = websearch.search_web("president trump news today")

    assert out["provider"] == "Bing News"
    assert out["results"] == [
        {
            "title": "Current headline about President Trump",
            "url": "https://news.example/current",
            "snippet": "A current named news item was found.",
            "source": "Example News",
        }
    ]
    assert any(e.get("provider") == "DuckDuckGo" and "challenge" in e.get("error", "") for e in out["errors"])


def test_search_web_falls_back_to_bing_web_when_news_has_no_results(monkeypatch):
    duck_challenge = '<form id="challenge-form" action="//duckduckgo.com/anomaly.js?cc=botnet"></form>'
    bing_news_empty = "<html><body>No news cards</body></html>"
    bing_web = """
    <html><body>
      <ol id="b_results">
        <li class="b_algo">
          <h2><a href="https://example.com/page">External fact result</a></h2>
          <div class="b_caption"><p>Example external fact snippet.</p></div>
        </li>
      </ol>
    </body></html>
    """

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        if "duckduckgo.com" in url:
            return _Response(duck_challenge)
        if "/news/search" in url:
            return _Response(bing_news_empty)
        if "/search" in url:
            return _Response(bing_web)
        raise AssertionError(f"unexpected fallback URL: {url}")

    monkeypatch.setattr(websearch, "urlopen", fake_urlopen)

    out = websearch.search_web("latest external fact")

    assert out["provider"] == "Bing"
    assert out["results"][0]["title"] == "External fact result"
    assert out["results"][0]["snippet"] == "Example external fact snippet."


def test_search_query_for_removes_conversational_prefixes():
    assert (
        websearch.search_query_for("Can you please get me US political news for today?")
        == "US political news today"
    )
    assert (
        websearch.search_query_for("I would like news from today in politics in the US")
        == "news from today in politics in the US"
    )
    assert (
        websearch.search_query_for("/search native plants for a Phoenix shade garden")
        == "native plants for a Phoenix shade garden"
    )


def test_irrelevant_results_are_rejected():
    garbage = [
        {"title": "Canva: Visual Suite for Everyone", "url": "https://www.canva.com", "snippet": "Design tool"},
        {"title": "CAN Definition", "url": "https://example.com/can", "snippet": "Dictionary definition"},
    ]

    assert websearch._relevant_results(garbage, "US political news today", 5) == []


def test_general_research_prompt_triggers_search(monkeypatch):
    captured = {}

    def fake_search(query, max_results=None):
        captured["query"] = query
        return {"query": query, "results": [{"title": "Sourdough guide"}], "errors": []}

    monkeypatch.setattr(websearch, "search_web", fake_search)

    out = websearch.web_context_for("Research beginner sourdough hydration ratios")

    assert out["results"][0]["title"] == "Sourdough guide"
    assert captured["query"] == "Research beginner sourdough hydration ratios"


def test_search_command_sends_only_general_topic(monkeypatch):
    captured = {}

    def fake_search(query, max_results=None):
        captured["query"] = query
        return {"query": query, "results": [], "errors": []}

    monkeypatch.setattr(websearch, "search_web", fake_search)

    websearch.web_context_for("/search native plants for a Phoenix shade garden")

    assert captured["query"] == "native plants for a Phoenix shade garden"
