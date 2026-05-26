import json
from pathlib import Path

import pytest

from tools.document_tools import (
    check_document_ai_requirements,
    check_document_tools_requirements,
    document_ai_extract_tool,
    document_extract_tool,
)


class _Response:
    def __init__(self, payload, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise pytest.importorskip("httpx").HTTPStatusError(
                "boom", request=None, response=self
            )

    def json(self):
        return self._payload


def _set_doc_tools_config(monkeypatch, intake_dir: Path, base_url: str = "http://127.0.0.1:9478"):
    config = {
        "document_tools": {
            "base_url": base_url,
            "intake_dir": str(intake_dir),
            "timeout": 12,
            "cleanup_after_extract": True,
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)


def test_document_extract_stages_local_file_calls_sidecar_and_cleans_up(monkeypatch, tmp_path):
    source_path = tmp_path / "report.txt"
    source_path.write_text("quarterly update", encoding="utf-8")
    intake_dir = tmp_path / "doc-tools" / "intake"
    _set_doc_tools_config(monkeypatch, intake_dir)

    observed = {}

    def fake_post(url, json=None, timeout=None):
        staged_local = intake_dir / Path(json["source"]).name
        observed["url"] = url
        observed["timeout"] = timeout
        observed["staged_local"] = staged_local
        observed["payload"] = json
        assert staged_local.exists()
        return _Response(
            {
                "ok": True,
                "backend_used": "markitdown",
                "source": json["source"],
                "source_kind": "local_path",
                "mime_type": "text/plain",
                "markdown": "quarterly update",
                "structured_data": None,
                "metadata": {"converter": "markitdown"},
                "warnings": [],
                "fallback_chain": ["markitdown"],
                "timings_ms": {"total": 15},
                "error": None,
            }
        )

    monkeypatch.setattr("tools.document_tools.httpx.post", fake_post)

    result = json.loads(document_extract_tool(str(source_path)))

    assert result["ok"] is True
    assert result["backend_used"] == "markitdown"
    assert result["original_source"] == str(source_path.resolve())
    assert result["cleanup_performed"] is True
    assert observed["url"] == "http://127.0.0.1:9478/extract"
    assert observed["timeout"] == 12
    assert observed["staged_local"].exists() is False


def test_document_extract_keeps_user_managed_file_when_already_in_intake(monkeypatch, tmp_path):
    intake_dir = tmp_path / "doc-tools" / "intake"
    intake_dir.mkdir(parents=True)
    source_path = intake_dir / "already-staged.txt"
    source_path.write_text("keep me", encoding="utf-8")
    _set_doc_tools_config(monkeypatch, intake_dir)

    def fake_post(url, json=None, timeout=None):
        assert json["source"] == "/data/intake/already-staged.txt"
        return _Response(
            {
                "ok": True,
                "backend_used": "markitdown",
                "source": json["source"],
                "source_kind": "local_path",
                "mime_type": "text/plain",
                "markdown": "keep me",
                "structured_data": None,
                "metadata": {"converter": "markitdown"},
                "warnings": [],
                "fallback_chain": ["markitdown"],
                "timings_ms": {"total": 8},
                "error": None,
            }
        )

    monkeypatch.setattr("tools.document_tools.httpx.post", fake_post)

    result = json.loads(document_extract_tool(str(source_path)))

    assert result["cleanup_performed"] is False
    assert source_path.exists()


def test_document_extract_delegates_urls_to_web_extract(monkeypatch):
    monkeypatch.setattr(
        "tools.web_tools.web_extract_tool",
        lambda urls, format="markdown": json.dumps(
            {
                "results": [
                    {
                        "url": urls[0],
                        "title": "Example",
                        "content": "hello world",
                        "error": None,
                    }
                ]
            }
        ),
    )

    result = json.loads(document_extract_tool("https://example.com/report.pdf", max_chars=5))

    assert result["backend_used"] == "web_extract"
    assert result["source_kind"] == "url"
    assert result["markdown"] == "hello"
    assert result["metadata"]["truncated"] is True


def test_document_tools_requirements_check_uses_health_endpoint(monkeypatch, tmp_path):
    intake_dir = tmp_path / "doc-tools" / "intake"
    intake_dir.mkdir(parents=True)
    _set_doc_tools_config(monkeypatch, intake_dir, base_url="http://127.0.0.1:9999")
    monkeypatch.setattr(
        "tools.document_tools.httpx.get",
        lambda url, timeout=None: _Response({"ok": True}),
    )
    monkeypatch.setattr("tools.document_tools._health_cache", {})

    assert check_document_tools_requirements() is True


def _set_document_ai_config(monkeypatch, base_url: str = "http://spark-doc-ai:8098", token: str = "test-token"):
    config = {"document_ai": {"base_url": base_url, "token": token, "timeout": 34}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)


def test_document_ai_extract_calls_redacted_spark_endpoint_with_token(monkeypatch, tmp_path):
    source_path = tmp_path / "scan.pdf"
    source_path.write_bytes(b"%PDF-1.4\n")
    _set_document_ai_config(monkeypatch)
    observed = {}

    def fake_post(url, headers=None, json=None, params=None, timeout=None):
        observed["url"] = url
        observed["headers"] = headers
        observed["params"] = params
        observed["timeout"] = timeout
        observed["payload"] = json
        return _Response(
            {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "hello world"}, "backend": "docling"}
                    ]
                },
                "documentAiGateway": {"backend": "docling", "redacted": True},
                "metadata": {},
            }
        )

    monkeypatch.setattr("tools.document_tools.httpx.post", fake_post)

    result = json.loads(document_ai_extract_tool(str(source_path), max_chars=5))

    assert observed["url"] == "http://spark-doc-ai:8098/layout-parsing/redacted"
    assert observed["headers"] == {"X-Document-AI-Token": "test-token"}
    assert observed["params"] == {}
    assert observed["timeout"] == 34
    assert observed["payload"]["filename"] == "scan.pdf"
    assert observed["payload"]["fileType"] == 0
    assert observed["payload"]["file"] == "JVBERi0xLjQK"
    assert result["ok"] is True
    assert result["markdown"] == "hello"
    assert result["markdown_truncated"] is True
    assert result["metadata"]["backend_used"] == "spark_document_ai"
    assert result["metadata"]["endpoint"] == "/layout-parsing/redacted"
    assert result["metadata"]["redacted"] is True


def test_document_ai_extract_can_call_unredacted_endpoint(monkeypatch, tmp_path):
    source_path = tmp_path / "scan.pdf"
    source_path.write_bytes(b"%PDF-1.4\n")
    _set_document_ai_config(monkeypatch)
    observed = {}

    def fake_post(url, headers=None, json=None, params=None, timeout=None):
        observed["url"] = url
        observed["params"] = params
        observed["payload"] = json
        return _Response({"ok": True, "metadata": {}})

    monkeypatch.setattr("tools.document_tools.httpx.post", fake_post)

    result = json.loads(document_ai_extract_tool(str(source_path), redacted=False))

    assert result["ok"] is True
    assert observed["url"] == "http://spark-doc-ai:8098/layout-parsing"
    assert observed["params"] == {"redact": "false"}


def test_document_ai_extract_requires_token(monkeypatch, tmp_path):
    source_path = tmp_path / "scan.pdf"
    source_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"document_ai": {"base_url": "http://spark-doc-ai:8098"}})
    for key in ("HERMES_DOCUMENT_AI_TOKEN", "DOCUMENT_AI_TOKEN", "SPARK_DOCUMENT_AI_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    result = json.loads(document_ai_extract_tool(str(source_path)))

    assert "Spark document-ai token is not configured" in result["error"]


def test_document_ai_requirements_need_token_and_health(monkeypatch):
    _set_document_ai_config(monkeypatch)
    monkeypatch.setattr(
        "tools.document_tools.httpx.get",
        lambda url, timeout=None: _Response({"ok": True}),
    )
    monkeypatch.setattr("tools.document_tools._health_cache", {})

    assert check_document_ai_requirements() is True

