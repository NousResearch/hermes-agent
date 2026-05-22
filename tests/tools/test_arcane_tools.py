import json

from model_tools import get_tool_definitions
from tools import arcane_tools


class _Response:
    def __init__(self, status_code=200, json_data=None, text="", reason=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.reason = reason

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


def test_arcane_tool_schemas_are_discoverable():
    names = {
        tool["function"]["name"]
        for tool in get_tool_definitions(enabled_toolsets=["arcane"], quiet_mode=True)
    }

    assert {
        "arcane_list_files",
        "arcane_read_file",
        "arcane_write_file",
        "arcane_create_snapshot",
        "arcane_get_session",
    }.issubset(names)


def test_arcane_list_files_uses_session_api_and_token(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _Response(
            json_data={
                "files": ["index.html", "styles.css"],
                "artifactFiles": [{"path": "index.html", "size": 12}],
            }
        )

    monkeypatch.setenv("ARCANE_BASE_URL", "http://arcane.test")
    monkeypatch.setenv("ARCANE_SESSION_ID", "sess-1")
    monkeypatch.setenv("ARCANE_ACCESS_TOKEN", "secret-token")
    monkeypatch.setattr(arcane_tools.requests, "request", fake_request)

    result = json.loads(arcane_tools.arcane_list_files())

    assert result["success"] is True
    assert result["files"] == ["index.html", "styles.css"]
    assert calls[0][0] == "GET"
    assert calls[0][1] == "http://arcane.test/api/sessions/sess-1"
    assert calls[0][2]["headers"]["x-arcane-token"] == "secret-token"


def test_arcane_read_file_encodes_artifact_path(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _Response(text="<main>ready</main>")

    monkeypatch.setenv("ARCANE_BASE_URL", "http://arcane.test")
    monkeypatch.setenv("ARCANE_SESSION_ID", "sess-1")
    monkeypatch.delenv("ARCANE_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(arcane_tools.requests, "request", fake_request)

    result = json.loads(arcane_tools.arcane_read_file("nested/file.html"))

    assert result["success"] is True
    assert result["content"] == "<main>ready</main>"
    assert calls[0][0] == "GET"
    assert calls[0][1] == "http://arcane.test/artifact/sess-1/nested%2Ffile.html"


def test_arcane_write_file_encodes_path_and_sends_text(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _Response(
            json_data={
                "file": {"path": "nested/file.html", "size": 17},
                "files": ["nested/file.html"],
                "artifactFiles": [{"path": "nested/file.html", "size": 17}],
            }
        )

    monkeypatch.setenv("ARCANE_BASE_URL", "http://arcane.test")
    monkeypatch.setenv("ARCANE_SESSION_ID", "sess-1")
    monkeypatch.setattr(arcane_tools.requests, "request", fake_request)

    result = json.loads(arcane_tools.arcane_write_file("nested/file.html", "<h1>Arcane</h1>"))

    assert result["success"] is True
    assert result["file"]["path"] == "nested/file.html"
    assert calls[0][0] == "PUT"
    assert calls[0][1] == "http://arcane.test/api/sessions/sess-1/files/nested%2Ffile.html"
    assert calls[0][2]["data"] == b"<h1>Arcane</h1>"
    assert calls[0][2]["headers"]["Content-Type"] == "text/plain; charset=utf-8"


def test_arcane_http_errors_redact_access_token(monkeypatch):
    def fake_request(method, url, **kwargs):
        return _Response(status_code=401, text="bad token secret-token")

    monkeypatch.setenv("ARCANE_BASE_URL", "http://arcane.test")
    monkeypatch.setenv("ARCANE_SESSION_ID", "sess-1")
    monkeypatch.setenv("ARCANE_ACCESS_TOKEN", "secret-token")
    monkeypatch.setattr(arcane_tools.requests, "request", fake_request)

    result = json.loads(arcane_tools.arcane_list_files())

    assert result["success"] is False
    assert "secret-token" not in result["error"]
    assert "[redacted]" in result["error"]
