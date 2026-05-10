"""Unit tests for tools.xai_responses_chat_tool."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from tools import xai_responses_chat_tool
from tools.xai_responses_chat_tool import (
    XAI_RESPONSES_SCHEMA,
    XaiResponsesError,
    _extract_text,
    check_xai_responses_requirements,
    xai_responses_chat,
)


# ---------------------------------------------------------------------------
# Fake httpx
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _FakePost:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url: str, *, headers: Dict[str, str], json: Any, timeout: int):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.response


@pytest.fixture
def fake_post(monkeypatch):
    holder: Dict[str, _FakePost] = {}

    def install(response: _FakeResponse) -> _FakePost:
        fake = _FakePost(response)
        holder["fake"] = fake
        monkeypatch.setattr(xai_responses_chat_tool.httpx, "post", fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def _no_disk_config(monkeypatch):
    monkeypatch.setattr(xai_responses_chat_tool, "_config_section", lambda: {})


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-test-resp")


def _ok(payload: Dict[str, Any]) -> _FakeResponse:
    return _FakeResponse(200, payload)


def _basic_payload(text: str = "hello", response_id: str = "resp-001"):
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "output": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": text}]},
        ],
    }


# ---------------------------------------------------------------------------
# Requirements / schema
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        ok, why = check_xai_responses_requirements()
        assert ok is False
        assert "XAI_API_KEY" in why

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk")
        ok, _ = check_xai_responses_requirements()
        assert ok is True


class TestSchema:
    def test_schema_advertises_distinctive_params(self):
        props = XAI_RESPONSES_SCHEMA["parameters"]["properties"]
        for key in (
            "prompt", "input", "instructions", "model",
            "store", "previous_response_id", "max_turns",
            "parallel_tool_calls", "reasoning_effort",
        ):
            assert key in props

    def test_reasoning_effort_enum(self):
        spec = XAI_RESPONSES_SCHEMA["parameters"]["properties"]["reasoning_effort"]
        assert set(spec["enum"]) == {"none", "low", "medium", "high"}


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

class TestArgValidation:
    def test_missing_prompt_and_input_raises(self, api_key):
        with pytest.raises(ValueError):
            xai_responses_chat()

    def test_empty_prompt_raises(self, api_key):
        with pytest.raises(ValueError):
            xai_responses_chat("")

    def test_prompt_and_input_mutually_exclusive(self, api_key):
        with pytest.raises(ValueError):
            xai_responses_chat("hi", input="hi")

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(XaiResponsesError):
            xai_responses_chat("hi")

    def test_invalid_reasoning_effort_raises(self, api_key):
        with pytest.raises(ValueError):
            xai_responses_chat("hi", reasoning_effort="ULTRA")

    def test_negative_timeout_raises(self, api_key):
        with pytest.raises(ValueError):
            xai_responses_chat("hi", timeout_seconds=-1)


# ---------------------------------------------------------------------------
# Body construction — distinctive Responses API params
# ---------------------------------------------------------------------------

class TestBodyConstruction:
    def test_prompt_becomes_input(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hello there")
        assert fake.calls[0]["json"]["input"] == "hello there"

    def test_explicit_input_passes_through(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        items = [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
        xai_responses_chat(input=items)
        assert fake.calls[0]["json"]["input"] == items

    def test_default_model(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi")
        assert fake.calls[0]["json"]["model"] == "grok-4.3"

    def test_explicit_model(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", model="grok-4.20-multi-agent-0309")
        assert fake.calls[0]["json"]["model"] == "grok-4.20-multi-agent-0309"

    def test_store_true_threaded(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", store=True)
        assert fake.calls[0]["json"]["store"] is True

    def test_store_false_threaded(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", store=False)
        assert fake.calls[0]["json"]["store"] is False

    def test_store_omitted_when_none(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi")
        assert "store" not in fake.calls[0]["json"]

    def test_previous_response_id_threaded(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", previous_response_id="resp-prev-99")
        assert fake.calls[0]["json"]["previous_response_id"] == "resp-prev-99"

    def test_max_turns_threaded(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", max_turns=5)
        assert fake.calls[0]["json"]["max_turns"] == 5

    def test_parallel_tool_calls_threaded(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", parallel_tool_calls=False)
        assert fake.calls[0]["json"]["parallel_tool_calls"] is False

    def test_reasoning_effort_translates_to_object(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", reasoning_effort="high")
        assert fake.calls[0]["json"]["reasoning"] == {"effort": "high"}

    def test_instructions_threaded(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi", instructions="be concise")
        assert fake.calls[0]["json"]["instructions"] == "be concise"

    def test_extra_body_merges_but_not_clobber_input_or_model(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat(
            "hi",
            extra_body={
                "search_parameters": {"mode": "auto"},
                "model": "should-be-ignored",
                "input": "should-be-ignored",
            },
        )
        body = fake.calls[0]["json"]
        assert body["search_parameters"] == {"mode": "auto"}
        assert body["model"] == "grok-4.3"
        assert body["input"] == "hi"

    def test_optional_params_omitted_when_none(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi")
        body = fake.calls[0]["json"]
        for key in (
            "store", "previous_response_id", "max_turns",
            "parallel_tool_calls", "reasoning",
            "max_output_tokens", "temperature", "top_p",
            "tools", "tool_choice", "user", "instructions",
        ):
            assert key not in body


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_returns_response_id(self, api_key, fake_post):
        fake_post(_ok(_basic_payload(response_id="resp-xyz")))
        out = xai_responses_chat("hi")
        assert out["response_id"] == "resp-xyz"

    def test_returns_status(self, api_key, fake_post):
        fake_post(_ok(_basic_payload()))
        out = xai_responses_chat("hi")
        assert out["status"] == "completed"

    def test_returns_output_text(self, api_key, fake_post):
        fake_post(_ok(_basic_payload(text="Bonjour Julien")))
        out = xai_responses_chat("hi")
        assert out["output_text"] == "Bonjour Julien"

    def test_returns_raw_payload(self, api_key, fake_post):
        payload = _basic_payload()
        fake_post(_ok(payload))
        out = xai_responses_chat("hi")
        assert out["raw"] == payload

    def test_extract_text_concatenates_multiple_pieces(self):
        payload = {
            "output": [
                {"content": [
                    {"type": "output_text", "text": "Hello "},
                    {"type": "output_text", "text": "world"},
                ]},
            ]
        }
        assert _extract_text(payload) == "Hello world"

    def test_extract_text_falls_back_to_legacy_field(self):
        payload = {"output_text": "legacy"}
        assert _extract_text(payload) == "legacy"

    def test_extract_text_empty_when_nothing_readable(self):
        assert _extract_text({}) == ""
        assert _extract_text({"output": []}) == ""


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------

class TestHttpErrors:
    def test_4xx_raises(self, api_key, fake_post):
        fake_post(_FakeResponse(401, text='{"error":"bad key"}'))
        with pytest.raises(XaiResponsesError) as exc:
            xai_responses_chat("hi")
        assert "401" in str(exc.value)

    def test_5xx_raises(self, api_key, fake_post):
        fake_post(_FakeResponse(503, text="upstream down"))
        with pytest.raises(XaiResponsesError) as exc:
            xai_responses_chat("hi")
        assert "503" in str(exc.value)

    def test_non_object_body_raises(self, api_key, fake_post):
        fake_post(_FakeResponse(200, payload=["not", "an", "object"]))
        with pytest.raises(XaiResponsesError):
            xai_responses_chat("hi")

    def test_network_error_raises(self, api_key, monkeypatch):
        def _boom(*_a, **_kw):
            raise httpx.ConnectError("boom")
        monkeypatch.setattr(xai_responses_chat_tool.httpx, "post", _boom)
        with pytest.raises(XaiResponsesError) as exc:
            xai_responses_chat("hi")
        assert "POST failed" in str(exc.value)


# ---------------------------------------------------------------------------
# Headers + URL
# ---------------------------------------------------------------------------

class TestHeaders:
    def test_calls_responses_endpoint(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi")
        assert fake.calls[0]["url"].endswith("/responses")

    def test_authorization_bearer(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi")
        assert fake.calls[0]["headers"]["Authorization"] == "Bearer sk-test-resp"

    def test_user_agent(self, api_key, fake_post):
        fake = fake_post(_ok(_basic_payload()))
        xai_responses_chat("hi")
        assert fake.calls[0]["headers"]["User-Agent"].startswith("Hermes-Agent/")
