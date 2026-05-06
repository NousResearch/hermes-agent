import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ACQUIRE = REPO_ROOT / "tools" / "web_acquire.py"


def test_web_acquire_does_not_register_tools():
    content = WEB_ACQUIRE.read_text()
    assert "registry.register" not in content


def test_default_scrapling_runtime_python_uses_isolated_runtime():
    from tools.web_acquire import default_scrapling_runtime_python

    runtime_python = default_scrapling_runtime_python()

    assert str(runtime_python).endswith(".hermes/runtimes/scrapling/bin/python")


def test_build_scrapling_command_uses_pilot_runner():
    from tools.web_acquire import build_scrapling_command

    command = build_scrapling_command(
        url="https://example.com",
        selector="h1",
        selector_type="css",
        mode="static",
        fallback_reason="selector_required",
        timeout=20,
        wait_selector=None,
        network_idle=False,
        max_chars=50000,
        runtime_python="/tmp/runtime/bin/python",
    )

    rendered = " ".join(command)
    assert command[0] == "/tmp/runtime/bin/python"
    assert "optional-skills/research/scrapling/scripts/scrapling_extract.py" in rendered
    assert "--url" in command
    assert "https://example.com" in command
    assert "--selector" in command
    assert "h1" in command
    assert "--selector-type" in command
    assert "css" in command
    assert "--mode" in command
    assert "static" in command
    assert "--fallback-reason" in command
    assert "selector_required" in command
    assert "--network-idle" not in command


def test_difficult_web_extract_parses_success_receipt(monkeypatch):
    from tools import web_acquire

    payload = {
        "backend": "scrapling",
        "mode": "static",
        "url": "https://example.com",
        "selector": "h1",
        "selector_type": "css",
        "content": "<h1>Example Domain</h1>",
        "elapsed_ms": 100,
        "fallback_reason": "selector_required",
        "errors": [],
    }

    def fake_run(command, capture_output, text, check, timeout):
        assert command[0] == "/tmp/runtime/bin/python"
        assert capture_output is True
        assert text is True
        assert check is False
        assert timeout == 30
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(web_acquire.subprocess, "run", fake_run)

    receipt = web_acquire.difficult_web_extract(
        "https://example.com",
        selector="h1",
        runtime_python="/tmp/runtime/bin/python",
        timeout=20,
    )

    assert receipt == payload


def test_difficult_web_extract_returns_structured_error_on_subprocess_failure(monkeypatch):
    from tools import web_acquire

    def fake_run(command, capture_output, text, check, timeout):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="python not found")

    monkeypatch.setattr(web_acquire.subprocess, "run", fake_run)

    receipt = web_acquire.difficult_web_extract(
        "https://example.com",
        selector="h1",
        runtime_python="/missing/python",
    )

    assert receipt["backend"] == "scrapling"
    assert receipt["mode"] == "static"
    assert receipt["url"] == "https://example.com"
    assert receipt["selector"] == "h1"
    assert receipt["content"] == ""
    assert receipt["errors"]
    assert receipt["errors"][0]["type"] == "ScraplingAdapterError"
    assert "python not found" in receipt["errors"][0]["message"]


def test_difficult_web_extract_returns_structured_error_for_invalid_json(monkeypatch):
    from tools import web_acquire

    def fake_run(command, capture_output, text, check, timeout):
        return subprocess.CompletedProcess(command, 0, stdout="not json", stderr="fetch log")

    monkeypatch.setattr(web_acquire.subprocess, "run", fake_run)

    receipt = web_acquire.difficult_web_extract(
        "https://example.com",
        selector="h1",
        runtime_python="/tmp/runtime/bin/python",
    )

    assert receipt["backend"] == "scrapling"
    assert receipt["content"] == ""
    assert receipt["errors"][0]["type"] == "InvalidScraplingReceipt"
    assert "not json" in receipt["errors"][0]["stdout"]
    assert "fetch log" in receipt["errors"][0]["stderr"]


def test_difficult_web_extract_rejects_non_http_urls_without_subprocess(monkeypatch):
    from tools import web_acquire

    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess should not be called for invalid URLs")

    monkeypatch.setattr(web_acquire.subprocess, "run", fake_run)

    receipt = web_acquire.difficult_web_extract("file:///etc/passwd", selector="h1")

    assert receipt["backend"] == "scrapling"
    assert receipt["url"] == "file:///etc/passwd"
    assert receipt["content"] == ""
    assert receipt["errors"][0]["type"] == "InvalidURL"
