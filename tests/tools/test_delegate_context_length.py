"""Real child-agent context-length resolution tests for delegate_task."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.delegate_tool import _build_child_agent


class _OllamaStub:
    def __init__(self, context_length: int = 196608):
        self.context_length = context_length
        self.show_calls = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": []})
                else:
                    self._send(404, {})

            def do_POST(self):
                if self.path == "/api/show":
                    owner.show_calls += 1
                    self._send(
                        200,
                        {"model_info": {"qwen3.context_length": owner.context_length}},
                    )
                else:
                    self._send(404, {})

            def _send(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _parent(base_url: str):
    return SimpleNamespace(
        base_url=base_url,
        api_key="test-key",
        provider="ollama-cloud",
        api_mode="chat_completions",
        model="parent-model",
        platform="cli",
        enabled_toolsets=[],
        disabled_toolsets=[],
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        openrouter_min_coding_score=None,
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
        reasoning_config=None,
        request_overrides={},
        max_tokens=None,
        _fallback_chain=None,
        acp_command=None,
        acp_args=[],
        session_id=None,
    )


def _write_config(home: Path, base_url: str, *, own_context=None):
    provider_model = {"context_length": own_context} if own_context else {}
    config = {
        "model": {
            "default": "parent-model",
            "provider": "ollama-cloud",
            "base_url": base_url,
            "context_length": 8192,
        },
        "providers": {
            "ollama-cloud": {
                "api": base_url,
                "models": {"qwen3-coder-next:cloud": provider_model},
            }
        },
    }
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (home / "context_length_cache.yaml").write_text(
        yaml.safe_dump({"qwen3-coder-next:cloud@" + base_url: 4096}),
        encoding="utf-8",
    )


def _build(monkeypatch, tmp_path, stub, *, own_context=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path / "hermes", stub.base_url, own_context=own_context)
    child = _build_child_agent(
        task_index=0,
        goal="inspect the repository",
        context=None,
        toolsets=[],
        model="qwen3-coder-next:cloud",
        max_iterations=1,
        task_count=1,
        parent_agent=_parent(stub.base_url),
    )
    return child


def test_child_model_config_wins_without_probe(monkeypatch, tmp_path):
    stub = _OllamaStub()
    try:
        home = tmp_path / "hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        _write_config(home, stub.base_url, own_context=64000)
        cache_before = (home / "context_length_cache.yaml").read_bytes()
        child = _build(monkeypatch, tmp_path, stub, own_context=64000)
        assert child.provider == "ollama-cloud"
        assert child.base_url == stub.base_url
        assert child._config_context_length == 64000
        assert child.context_compressor.context_length == 64000
        assert stub.show_calls == 0
        assert (home / "context_length_cache.yaml").read_bytes() == cache_before
        child.close()
    finally:
        stub.close()


def test_child_model_probe_does_not_inherit_parent_context(monkeypatch, tmp_path):
    stub = _OllamaStub()
    try:
        child = _build(monkeypatch, tmp_path, stub)
        assert child.provider == "ollama-cloud"
        assert child.base_url == stub.base_url
        assert child.context_compressor.context_length == 196608
        assert stub.show_calls >= 1
        assert child.context_compressor.context_length != 8192
        child.close()
    finally:
        stub.close()
