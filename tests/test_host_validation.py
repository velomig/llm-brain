"""Host-header allowlist defeats DNS-rebinding against the loopback server.

The server is often unauthenticated (loopback, no NEURAL_MAP_TOKEN). Without a
Host check, a site you visit could rebind its own domain to 127.0.0.1 and reach
the API same-origin. _host_allowed accepts only Hosts that an external DNS zone
can't aim here — IP literals, localhost, *.local, and the configured allowlist —
and rejects registered hostnames (the rebinding vector).
"""
from __future__ import annotations

import config
import server


def test_loopback_and_lan_ip_hosts_allowed():
    assert server._host_allowed("127.0.0.1:8000")
    assert server._host_allowed("127.0.0.1")
    assert server._host_allowed("192.168.1.50:8000")   # iPad on the LAN via IP
    assert server._host_allowed("localhost:8000")
    assert server._host_allowed("[::1]:8000")           # bracketed IPv6 + port


def test_mdns_local_name_allowed():
    assert server._host_allowed("my-mac.local:8000")


def test_registered_hostname_rejected():
    # The DNS-rebinding vector: an attacker-controlled domain pointed at 127.0.0.1.
    assert not server._host_allowed("attacker.com")
    assert not server._host_allowed("evil.example.org:8000")


def test_configured_allowlist_is_honored(monkeypatch):
    monkeypatch.setattr(config, "SERVER_ALLOWED_HOSTS", ["brain.example.com"])
    assert server._host_allowed("brain.example.com:8000")
    assert not server._host_allowed("other.example.com")


def test_missing_host_is_allowed():
    # Browsers (the only rebinding vector) always send Host; its absence is a
    # non-browser client, not an attack, so it must not break the CLI/health.
    assert server._host_allowed(None)


# --- Origin allowlist: CSRF gate for every request -------------------------------
# A malicious page can fire a cross-site "simple request" POST (text/plain body,
# no CORS preflight) at the token-less loopback server — and cross-site GETs,
# whose responses the page can't read, still execute the handler. The Host gate
# doesn't see either, because the browser targets 127.0.0.1 directly. The
# unforgeable Origin header (plus Fetch Metadata for no-cors subresource GETs
# that omit Origin) carries the attacking page's own provenance and is rejected.

def test_ui_origins_allowed():
    # Same-origin as the server it targets: host and port agree with Host.
    assert server._origin_allowed("http://127.0.0.1:8000", "127.0.0.1:8000")
    assert server._origin_allowed("http://localhost:8000", "localhost:8000")
    assert server._origin_allowed("http://192.168.1.50:8000", "192.168.1.50:8000")  # LAN by IP
    assert server._origin_allowed("http://my-mac.local:8000", "my-mac.local:8000")
    assert server._origin_allowed("http://[::1]:8000", "[::1]:8000")               # IPv6 loopback


def test_cross_site_origin_rejected():
    assert not server._origin_allowed("https://evil.example", "127.0.0.1:8000")
    assert not server._origin_allowed("http://attacker.com:8000", "127.0.0.1:8000")
    assert not server._origin_allowed("null", "127.0.0.1:8000")   # sandboxed iframe / file:// page


def test_public_ip_origin_rejected():
    # The reported bypass: a page served from a public IP literal is NOT the
    # loopback server's own origin, even though it is an IP (which the Host gate
    # would accept for anti-rebinding). It must be rejected for mutations.
    assert not server._origin_allowed("https://8.8.8.8", "127.0.0.1:8000")
    assert not server._origin_allowed("http://8.8.8.8:8000", "127.0.0.1:8000")


def test_other_loopback_port_origin_rejected():
    # A different app on another loopback port is a distinct origin.
    assert not server._origin_allowed("http://127.0.0.1:9999", "127.0.0.1:8000")
    assert not server._origin_allowed("http://localhost", "localhost:8000")  # port 80 != 8000


def test_non_http_origin_scheme_rejected():
    # The real UI is only ever served over http(s); ftp://127.0.0.1 is not it,
    # even though the host is a loopback IP.
    assert not server._origin_allowed("ftp://127.0.0.1", "127.0.0.1:8000")
    assert not server._origin_allowed("file:///etc/passwd", "127.0.0.1:8000")


def test_missing_origin_is_allowed():
    # curl, launchd jobs, and non-browser clients send no Origin — they are not
    # confined by the browser sandbox, so rejecting them buys nothing.
    assert server._origin_allowed(None, "127.0.0.1:8000")


def test_configured_allowlist_covers_origin(monkeypatch):
    monkeypatch.setattr(config, "SERVER_ALLOWED_HOSTS", ["brain.example.com"])
    assert server._origin_allowed("https://brain.example.com", "brain.example.com")
    assert not server._origin_allowed("https://other.example.com", "brain.example.com")


def test_state_changing_request_with_foreign_origin_gets_403():
    from fastapi.testclient import TestClient

    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    response = client.post("/api/emotion/reset", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_state_changing_request_with_public_ip_origin_gets_403():
    from fastapi.testclient import TestClient

    # End-to-end version of the reported bypass through the real middleware.
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    response = client.post("/api/emotion/reset", headers={"Origin": "https://8.8.8.8"})
    assert response.status_code == 403


def test_same_origin_and_clientless_requests_unaffected():
    from fastapi.testclient import TestClient

    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    assert client.get("/api/emotion", headers={"Origin": "http://127.0.0.1:8000"}).status_code == 200
    assert client.post("/api/emotion/reset", headers={"Origin": "http://127.0.0.1:8000"}).status_code == 200
    assert client.post("/api/emotion/reset").status_code == 200   # no Origin (curl/scripts)
    assert client.get("/api/emotion").status_code == 200          # no headers at all


def test_cross_origin_get_gets_403():
    from fastapi.testclient import TestClient

    # CORS keeps the page from READING the response, but the request itself
    # still executed server-side — enough to trigger expensive or
    # lazily-mutating work on a token-less loopback server. Enforce the Origin
    # gate on GET too.
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    assert client.get("/api/emotion", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/stats", headers={"Origin": "https://evil.example"}).status_code == 403


def test_cross_site_no_cors_subresource_get_gets_403():
    from fastapi.testclient import TestClient

    # <img src="http://127.0.0.1:8000/api/export"> sends NO Origin header, so
    # only Fetch Metadata can identify it. The browser stamps Sec-Fetch-Site
    # and page script cannot forge it.
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    headers = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"}
    assert client.get("/api/stats", headers=headers).status_code == 403
    # Another loopback port is a different origin: "same-site" is not same-origin.
    headers["Sec-Fetch-Site"] = "same-site"
    assert client.get("/api/stats", headers=headers).status_code == 403


def test_fetch_metadata_same_origin_and_navigations_allowed():
    from fastapi.testclient import TestClient

    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    # The UI's own fetches.
    assert client.get(
        "/api/emotion",
        headers={"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "empty"},
    ).status_code == 200
    # Direct navigation (typed URL / bookmark).
    assert client.get(
        "/",
        headers={"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"},
    ).status_code == 200
    # A cross-site link to the app opens as the user's own visible tab.
    assert client.get(
        "/",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"},
    ).status_code == 200
    # ...but a cross-site POST is never a plain navigation we accept.
    assert client.post(
        "/api/emotion/reset",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"},
    ).status_code == 403
