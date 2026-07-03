from __future__ import annotations

import json
import subprocess

from tools import browser_upload_tool


def test_missing_files_returns_error(monkeypatch):
    monkeypatch.setattr(browser_upload_tool, "_resolve_cdp_endpoint", lambda: "http://127.0.0.1:9222")
    result = json.loads(browser_upload_tool.browser_upload_files(files=[]))
    assert "error" in result
    assert "files" in result["error"]


def test_nonexistent_file_returns_error(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_upload_tool, "_resolve_cdp_endpoint", lambda: "http://127.0.0.1:9222")
    missing = tmp_path / "missing.jpg"
    result = json.loads(browser_upload_tool.browser_upload_files(files=[str(missing)]))
    assert "error" in result
    assert "does not exist" in result["error"]


def test_no_cdp_endpoint_returns_error(monkeypatch, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")
    monkeypatch.setattr(browser_upload_tool, "_resolve_cdp_endpoint", lambda: "")
    result = json.loads(browser_upload_tool.browser_upload_files(files=[str(image)]))
    assert "error" in result
    assert "CDP endpoint" in result["error"]


def test_success_invokes_upload_helper(monkeypatch, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")
    captured = {}

    def fake_run(payload, timeout):
        captured["payload"] = payload
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(
            args=["node"],
            returncode=0,
            stdout=json.dumps(
                {
                    "success": True,
                    "uploadedFiles": 1,
                    "selector": payload["selector"],
                    "targetUrl": "https://www.facebook.com/groups/1",
                    "state": {"fileInputs": [{"files": 1}]},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(browser_upload_tool, "_resolve_cdp_endpoint", lambda: "http://127.0.0.1:9222")
    monkeypatch.setattr(browser_upload_tool, "_run_playwright_upload", fake_run)

    result = json.loads(
        browser_upload_tool.browser_upload_files(
            files=[str(image)],
            selector='input[type="file"][accept*="image"]',
            target_url_contains="facebook.com",
            timeout=12,
            settle_ms=1234,
        )
    )
    assert result["success"] is True
    assert result["uploadedFiles"] == 1
    assert captured["payload"]["files"] == [str(image.resolve())]
    assert captured["payload"]["selector"] == 'input[type="file"][accept*="image"]'
    assert captured["payload"]["targetUrlContains"] == "facebook.com"
    assert captured["payload"]["settleMs"] == 1234
    assert captured["timeout"] == 12


def test_helper_failure_returns_error(monkeypatch, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")

    def fake_run(payload, timeout):
        return subprocess.CompletedProcess(
            args=["node"],
            returncode=1,
            stdout=json.dumps({"success": False, "error": "playwright missing"}),
            stderr="stack trace",
        )

    monkeypatch.setattr(browser_upload_tool, "_resolve_cdp_endpoint", lambda: "http://127.0.0.1:9222")
    monkeypatch.setattr(browser_upload_tool, "_run_playwright_upload", fake_run)

    result = json.loads(browser_upload_tool.browser_upload_files(files=[str(image)]))
    assert "error" in result
    assert "playwright missing" in result["error"]
    assert result["stderr"] == "stack trace"


def test_check_requires_cdp_and_node(monkeypatch):
    monkeypatch.setattr(browser_upload_tool, "_resolve_cdp_endpoint", lambda: "http://127.0.0.1:9222")
    monkeypatch.setattr(browser_upload_tool, "_node_command", lambda: "/usr/bin/node")
    assert browser_upload_tool._browser_upload_files_check() is True

    monkeypatch.setattr(browser_upload_tool, "_resolve_cdp_endpoint", lambda: "")
    assert browser_upload_tool._browser_upload_files_check() is False
