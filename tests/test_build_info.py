from pathlib import Path

from fastapi.testclient import TestClient

import server
from brain.build_info import source_fingerprint


def test_source_fingerprint_changes_with_python_code(tmp_path: Path):
    (tmp_path / "brain").mkdir()
    (tmp_path / "server.py").write_text("app = 1\n", encoding="utf-8")
    (tmp_path / "config.py").write_text("SETTING = 1\n", encoding="utf-8")
    module = tmp_path / "brain" / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_fingerprint(tmp_path)

    module.write_text("VALUE = 2\n", encoding="utf-8")
    second = source_fingerprint(tmp_path)

    assert first != second
    assert len(first) == 20
    assert len(second) == 20


def test_build_endpoint_reports_loaded_source_fingerprint():
    response = TestClient(server.app, base_url="http://127.0.0.1").get("/api/build")

    assert response.status_code == 200
    assert response.json() == {"source_fingerprint": server.SOURCE_FINGERPRINT}


def test_launcher_restarts_backend_when_fingerprint_is_stale():
    launcher = (
        Path(__file__).resolve().parent.parent
        / "Neural Map.app/Contents/Resources/Scripts/main.applescript"
    ).read_text(encoding="utf-8")

    assert 'set buildURL to theURL & "/api/build"' in launcher
    assert "source_fingerprint" in launcher
    assert "X-Auth-Token" in launcher
    assert "launchctl kickstart -k" in launcher
    assert "launchctl bootstrap" in launcher
    assert "launchctl print" in launcher
    assert "if runningFingerprint is not diskFingerprint" in launcher
