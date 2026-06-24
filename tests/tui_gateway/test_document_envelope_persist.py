"""Document-path prompt.submit persists desktop user envelope for bubble fidelity."""

import threading
import types

import pytest

from tui_gateway import server


def test_run_prompt_submit_passes_desktop_envelope_as_persist_user_message(monkeypatch):
    """Document-path submits must persist wire text + AST, not expanded prose."""
    captured = {}

    class _FakeAgent:
        model = "test-model"
        base_url = ""
        api_key = ""
        provider = "openrouter"
        _config_context_length = None

        def run_conversation(self, user_message, **kwargs):
            captured["user_message"] = user_message
            captured["kwargs"] = kwargs
            return {"messages": [], "final_response": "ok"}

    session = {
        "history_lock": threading.Lock(),
        "history": [],
        "history_version": 0,
        "attached_images": [],
        "inflight_turn": {"user": "wire", "assistant": "", "streaming": True},
        "session_key": "sk-test",
        "agent": _FakeAgent(),
        "cols": 80,
    }

    document = [
        {"type": "text", "value": "hi "},
        {
            "type": "attachment",
            "id": "attachment-1",
            "kind": "file",
            "path": "foo.txt",
            "displayName": "foo",
        },
    ]
    wire_text = "hi @file:`foo.txt`"

    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *a: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a, **k: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *a: "/tmp")
    monkeypatch.setattr(server, "_clear_inflight_turn", lambda *a: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda *a: {})
    monkeypatch.setattr(server, "render_message", lambda *a, **k: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_get_db", lambda: None)

    def _fake_expand_document(*_a, **_k):
        return types.SimpleNamespace(
            message="hi",
            blocked=False,
            warnings=[],
            image_paths=[],
        )

    monkeypatch.setattr("agent.context_references.expand_document", _fake_expand_document)

    def _run_sync(target, daemon):
        target()

        return types.SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(server.threading, "Thread", _run_sync)

    server._run_prompt_submit(
        "rid",
        "sid",
        session,
        wire_text,
        document=document,
        document_version=1,
    )

    assert captured["user_message"] == "hi"
    envelope = captured["kwargs"]["persist_user_message"]
    assert envelope["text"] == wire_text
    assert envelope["document"] == document
    assert envelope["document_version"] == 1
