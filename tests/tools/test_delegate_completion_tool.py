"""Tests for the delegate_completion one-shot completion tool.

call_llm is mocked with the response shape it actually returns: a validated
chat-completions object with .choices[0].message (see
agent/auxiliary_client.py::_validate_llm_response, which coerces every
provider response into that shape before call_llm returns it). Text
extraction therefore goes through extract_content_or_reasoning, which also
covers reasoning models that put output in message.reasoning with
content=None.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import tools.delegate_completion_tool as dct
from tools.delegate_completion_tool import delegate_completion


def _chat_response(content, model="test-model", **message_fields):
    """Build the chat-completions shape call_llm returns for every provider."""
    message = SimpleNamespace(content=content, **message_fields)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model=model,
    )


class TestDispatch:
    def test_single_prompt(self, monkeypatch):
        calls = []

        def fake_call_llm(*args, **kwargs):
            calls.append((args, kwargs))
            return _chat_response("hello world")

        monkeypatch.setattr(dct, "call_llm", fake_call_llm)
        result = json.loads(delegate_completion(prompt="hello"))

        assert result["success"] is True
        assert result["count"] == 1
        assert result["failed"] == 0
        assert result["results"][0]["text"] == "hello world"
        assert result["results"][0]["model"] == "test-model"
        # The auxiliary task name routes config resolution to
        # auxiliary.delegate_completion.
        args, kwargs = calls[0]
        assert args[0] == "delegate_completion"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]

    def test_reasoning_model_content_none(self, monkeypatch):
        # DeepSeek-R1-style backends return content=None with the output in
        # message.reasoning. These are exactly the cheap/local models this
        # tool targets, so extraction must not come back empty.
        monkeypatch.setattr(
            dct, "call_llm",
            lambda *a, **kw: _chat_response(None, reasoning="the answer"),
        )
        result = json.loads(delegate_completion(prompt="x"))
        assert result["success"] is True
        assert result["results"][0]["text"] == "the answer"

    def test_batch_preserves_order(self, monkeypatch):
        def fake_call_llm(task, *, messages, **kwargs):
            return _chat_response(messages[-1]["content"].upper())

        monkeypatch.setattr(dct, "call_llm", fake_call_llm)
        result = json.loads(delegate_completion(batch=["alpha", "beta", "gamma"]))
        assert result["success"] is True
        assert result["count"] == 3
        assert [r["text"] for r in result["results"]] == ["ALPHA", "BETA", "GAMMA"]

    def test_both_prompt_and_batch_rejected(self):
        result = json.loads(delegate_completion(prompt="a", batch=["b"]))
        assert result["success"] is False
        assert "not both" in result["error"]

    def test_neither_prompt_nor_batch_rejected(self):
        result = json.loads(delegate_completion())
        assert result["success"] is False
        assert "required" in result["error"]

    def test_empty_batch_rejected(self):
        result = json.loads(delegate_completion(batch=[]))
        assert result["success"] is False
        assert "non-empty" in result["error"]

    def test_bare_string_batch_treated_as_singleton(self, monkeypatch):
        monkeypatch.setattr(
            dct, "call_llm", lambda *a, **kw: _chat_response("ok"),
        )
        result = json.loads(delegate_completion(batch="just one"))
        assert result["success"] is True
        assert result["count"] == 1

    def test_overrides_forwarded(self, monkeypatch):
        seen = {}

        def fake_call_llm(*args, **kwargs):
            seen.update(kwargs)
            return _chat_response("ok")

        monkeypatch.setattr(dct, "call_llm", fake_call_llm)
        delegate_completion(
            prompt="x",
            provider="openrouter",
            model="some/model",
            timeout=12.5,
            system="be terse",
            temperature=0.2,
            max_tokens=64,
        )
        assert seen["provider"] == "openrouter"
        assert seen["model"] == "some/model"
        assert seen["timeout"] == 12.5
        assert seen["temperature"] == 0.2
        assert seen["max_tokens"] == 64
        assert seen["messages"][0] == {"role": "system", "content": "be terse"}
        assert seen["messages"][1] == {"role": "user", "content": "x"}

    def test_single_prompt_failure(self, monkeypatch):
        def fake_call_llm(*args, **kwargs):
            raise RuntimeError("no provider reachable")

        monkeypatch.setattr(dct, "call_llm", fake_call_llm)
        result = json.loads(delegate_completion(prompt="x"))
        assert result["success"] is False
        assert result["failed"] == 1
        assert "RuntimeError" in result["results"][0]["error"]
        assert "no provider reachable" in result["results"][0]["error"]

    def test_batch_partial_failure_keeps_order(self, monkeypatch):
        def fake_call_llm(task, *, messages, **kwargs):
            text = messages[-1]["content"]
            if text == "boom":
                raise TimeoutError("backend timed out")
            return _chat_response(text)

        monkeypatch.setattr(dct, "call_llm", fake_call_llm)
        result = json.loads(delegate_completion(batch=["a", "boom", "c"]))
        assert result["success"] is False
        assert result["count"] == 3
        assert result["failed"] == 1
        assert result["results"][0]["text"] == "a"
        assert "TimeoutError" in result["results"][1]["error"]
        assert result["results"][2]["text"] == "c"

    def test_explicit_model_reported_when_response_lacks_model(self, monkeypatch):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            model=None,
        )
        monkeypatch.setattr(dct, "call_llm", lambda *a, **kw: response)
        result = json.loads(delegate_completion(prompt="x", model="my/override"))
        assert result["results"][0]["model"] == "my/override"


class TestRegistration:
    def test_registered_in_delegation_toolset(self):
        from tools.registry import registry

        assert registry.get_toolset_for_tool("delegate_completion") == "delegation"

    def test_schema_declares_documented_arguments(self):
        props = dct.DELEGATE_COMPLETION_SCHEMA["parameters"]["properties"]
        assert props["prompt"]["type"] == "string"
        assert props["batch"]["type"] == "array"
        for name in ("system", "provider", "model", "timeout", "temperature", "max_tokens"):
            assert name in props
        assert dct.DELEGATE_COMPLETION_SCHEMA["parameters"]["required"] == []

    def test_advertised_in_core_toolset(self):
        import toolsets

        assert "delegate_completion" in toolsets._HERMES_CORE_TOOLS
