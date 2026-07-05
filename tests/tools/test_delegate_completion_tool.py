"""Tests for the ``delegate_completion`` one-shot completion tool.

The tool is intentionally thin: it forwards each prompt through
``agent.auxiliary_client.call_llm`` and returns the text content. These
tests mock that call to verify:

  * the prompt vs. batch arg shape is normalized correctly
  * the auxiliary config key ``delegate_completion`` is what's used
  * provider / model / timeout overrides are forwarded
  * the response payload envelope matches the documented contract
  * errors from the underlying call_llm surface as ``{"success": False, ...}``
  * None on both shapes is rejected at the tool boundary
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools.delegate_completion_tool import (
    delegate_completion,
)


def _fake_response(text: str):
    """Return a duck-typed auxiliary_client response.

    ``_extract_aux_response_text`` first looks for ``output_text`` on the
    top-level response. Setting that field is the smallest surface that
    exercises the helper's primary branch, which is the branch every
    code path we care about (``call_llm`` with the standard Chat
    Completions response shape) takes in production after the auxiliary
    router normalizes the payload.
    """
    resp = type("Aux", (), {})()
    resp.output_text = text
    return resp


class TestDelegateCompletionDispatch:
    def test_single_prompt_is_normalized(self, monkeypatch):
        sent = []

        def _fake_call_llm(*args, **kwargs):
            sent.append((args, kwargs))
            return _fake_response("hello world")

        # Patch the symbol that ``tools.delegate_completion_tool``
        # imports at call time. Setting the module attribute is enough
        # because the tool does a fresh
        # ``from agent.auxiliary_client import call_llm`` per call.
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm
        )
        raw = delegate_completion(prompt="hello")
        result = json.loads(raw)
        assert result["success"] is True, result
        assert result["count"] == 1
        assert result["results"][0]["text"] == "hello world"
        assert result["results"][0]["input"] == "hello"
        assert len(sent) == 1
        # call_llm must have been called with the named auxiliary task
        # so the auxiliary config block ``delegate_completion`` is read.
        call_args, call_kwargs = sent[0]
        assert call_args and call_args[0] == "delegate_completion"
        messages = call_kwargs["messages"]
        assert messages == [{"role": "user", "content": "hello"}]

    def test_batch_runs_each_prompt(self, monkeypatch):
        responses = iter(
            [
                _fake_response("a-out"),
                _fake_response("b-out"),
                _fake_response("c-out"),
            ]
        )

        def _fake_call_llm(*args, **kwargs):
            return next(responses)

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm
        )
        raw = delegate_completion(batch=["alpha", "beta", "gamma"])
        result = json.loads(raw)
        assert result["success"] is True
        assert result["count"] == 3
        assert [r["text"] for r in result["results"]] == [
            "a-out",
            "b-out",
            "c-out",
        ]
        assert [r["input"] for r in result["results"]] == [
            "alpha",
            "beta",
            "gamma",
        ]

    def test_prompt_takes_precedence_over_batch_with_error(self):
        # Pass both — the function must surface a usage error rather than
        # silently picking one.
        raw = delegate_completion(
            prompt="only-this", batch=["also", "this"],
        )
        result = json.loads(raw)
        assert result["success"] is False
        assert "exactly one" in result["error"].lower()

    def test_neither_prompt_nor_batch_fails(self):
        raw = delegate_completion()
        result = json.loads(raw)
        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_overrides_forwarded_to_call_llm(self, monkeypatch):
        seen = {}

        def _fake_call_llm(*args, **kwargs):
            seen.update(kwargs)
            return _fake_response("ok")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm
        )
        delegate_completion(
            prompt="x",
            provider="openrouter",
            model="xai/grok-mini",
            timeout=12.5,
            system="be terse",
            temperature=0.2,
            max_tokens=64,
        )
        assert seen["provider"] == "openrouter"
        assert seen["model"] == "xai/grok-mini"
        assert seen["timeout"] == 12.5
        assert seen["temperature"] == 0.2
        assert seen["max_tokens"] == 64
        # system message is prepended (not folded into prompt)
        assert seen["messages"][0]["role"] == "system"
        assert seen["messages"][0]["content"] == "be terse"
        assert seen["messages"][1]["content"] == "x"

    def test_auxiliary_call_failure_returns_envelope(self, monkeypatch):
        def _fake_call_llm(*args, **kwargs):
            raise RuntimeError("no provider reachable")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm
        )
        raw = delegate_completion(prompt="x")
        result = json.loads(raw)
        assert result["success"] is False
        assert "RuntimeError" in result["error"]
        assert "no provider reachable" in result["error"]

    def test_empty_response_text_yields_empty_string(self, monkeypatch):
        # Auxiliary backend can legally return all-empty content blocks;
        # the tool's contract is that ``text`` is always a string.
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda *a, **kw: _fake_response(""),
        )
        raw = delegate_completion(prompt="x")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["results"][0]["text"] == ""


class TestDelegateCompletionRegistration:
    """The tool must self-register under the ``delegation`` toolset so
    that ``discover_builtin_tools`` picks it up."""

    def test_tool_module_imports_without_error(self):
        # Smoke test — the auto-register call lives at module bottom
        # and is wrapped in try/except so an unavailable registry is not
        # fatal. Importing the module must succeed at minimum.
        import tools.delegate_completion_tool as m

        assert callable(m.delegate_completion)
        assert "delegate_completion" in m.__all__
        assert "prompt-in" in m.TOOL_DESCRIPTION.lower() or "single" in m.TOOL_DESCRIPTION.lower()

    def test_schema_declares_expected_arguments(self):
        # The schema is what the LLM sees; verify the documented arg
        # set is present so the model knows how to call the tool.
        import tools.delegate_completion_tool as m

        schema = getattr(m, "_DELEGATE_COMPLETION_SCHEMA")
        props = schema["parameters"]["properties"]
        assert "prompt" in props
        assert "batch" in props
        assert "provider" in props
        assert "model" in props
        # Both prompt shapes are the documented contract.
        assert props["prompt"]["type"] == "string"
        assert props["batch"]["type"] == "array"

    def test_registry_picks_up_tool_via_registry_snapshot(self):
        # End-to-end: the registry must contain ``delegate_completion``
        # once the module is imported. We poke the well-known private
        # ``_tools`` mapping (the same mapping the public ``get_definitions``
        # reads internally); this is locked-in workhorse surface rather
        # than a public API contract, but the registry doesn't expose a
        # way to enumerate entries otherwise, and we deliberately avoid
        # the public ``get_definitions`` because (a) it filters by
        # ``check_fn`` and (b) the registry-as-singleton design is
        # stable across every Hermes release.
        import tools.delegate_completion_tool  # noqa: F401
        from tools.registry import registry

        names = {
            name for name in registry._tools  # noqa: SLF001
        }
        assert "delegate_completion" in names
