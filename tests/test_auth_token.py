from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config
import server


def test_nonloopback_bind_without_token_refuses_to_start(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    monkeypatch.setattr(config, "SERVER_HOST", "0.0.0.0")
    with pytest.raises(SystemExit):
        server._require_token_beyond_loopback()


def test_nonloopback_bind_with_token_starts(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "secret")
    monkeypatch.setattr(config, "SERVER_HOST", "0.0.0.0")
    server._require_token_beyond_loopback()


def test_loopback_bind_without_token_starts(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    for host in ("127.0.0.1", "::1", "localhost"):
        monkeypatch.setattr(config, "SERVER_HOST", host)
        server._require_token_beyond_loopback()


def test_tokenless_request_from_lan_peer_is_refused(monkeypatch):
    """The import-time guard only sees config.SERVER_HOST; `uvicorn server:app
    --host 0.0.0.0` bypasses it entirely. With no token, a request arriving
    from a non-loopback peer must be refused at request time."""
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    client = TestClient(server.app, base_url="http://127.0.0.1", client=("192.168.1.23", 44321))

    response = client.get("/api/health")

    assert response.status_code == 403
    assert "NEURAL_MAP_TOKEN" in response.json()["detail"]


def test_tokenless_request_from_loopback_peer_is_served(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    for peer in ("127.0.0.1", "::1"):
        client = TestClient(server.app, base_url="http://127.0.0.1", client=(peer, 44321))
        assert client.get("/api/health").status_code == 200


def test_lan_peer_with_token_is_served(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "secret")
    client = TestClient(server.app, base_url="http://127.0.0.1", client=("192.168.1.23", 44321))

    response = client.get("/api/health", headers={"X-Auth-Token": "secret"})

    assert response.status_code == 200


def test_non_ascii_token_probe_is_unauthorized_not_500(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "secret")
    client = TestClient(server.app, base_url="http://127.0.0.1")

    response = client.get("/api/stats?token=s%C3%A9cret")

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_query_token_bootstraps_cookie_and_redirects_clean_url(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "secret")
    client = TestClient(server.app, base_url="http://127.0.0.1")

    response = client.get("/?token=secret&view=chat", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "http://127.0.0.1/?view=chat"
    assert "nm_token=secret" in response.headers["set-cookie"]
    assert "token=secret" not in response.headers["location"]


def test_query_token_is_not_accepted_for_post_api_calls(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "secret")
    client = TestClient(server.app, base_url="http://127.0.0.1")

    response = client.post("/api/chat?token=secret", json={"message": "hello"})

    assert response.status_code == 400
    assert response.json() == {"detail": "query token is only accepted for browser bootstrap"}


def test_oversized_request_body_rejected_before_route_parse(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    monkeypatch.setattr(config, "MAX_REQUEST_BODY_BYTES", 16)
    client = TestClient(server.app, base_url="http://127.0.0.1")

    response = client.post("/api/memory", json={"text": "x" * 100})

    assert response.status_code == 413
    assert "request body exceeds" in response.json()["detail"]


def test_streaming_request_body_without_content_length_is_capped(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    monkeypatch.setattr(config, "MAX_REQUEST_BODY_BYTES", 16)
    client = TestClient(server.app, base_url="http://127.0.0.1")

    def chunks():
        yield b'{"text":"'
        yield b"x" * 100
        yield b'"}'

    response = client.post(
        "/api/memory",
        content=chunks(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert "request body exceeds" in response.json()["detail"]
