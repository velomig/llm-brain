from brain import websearch


def test_financial_analysis_always_qualifies_for_web_search(monkeypatch):
    monkeypatch.setattr(websearch.config, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(websearch.config, "WEB_SEARCH_MODE", "auto")

    assert websearch.should_search("lets do a portfolio analysis") is True
    assert websearch.should_search("review my retirement allocation") is True
    assert websearch.should_search("remember that my portfolio exists") is False
    assert websearch.should_search("explain memory allocation") is False
