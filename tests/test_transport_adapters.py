import json

import httpx

from agent.transport_adapters import AnthropicMessagesClient, GoogleGenerateContentClient


def _tool_def(name="web_search"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }


def test_anthropic_messages_client_converts_request_and_response(monkeypatch):
    client = AnthropicMessagesClient(base_url="https://api.example.test/v1", api_key="ant-key")
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "id": "msg_123",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "text", "text": "Found it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "web_search",
                        "input": {"query": "Hermes"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 2},
            },
            request=request,
        )

    monkeypatch.setattr(client._client, "post", fake_post)
    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Search for Hermes"},
        ],
        tools=[_tool_def()],
        max_tokens=512,
        timeout=12.0,
    )

    assert captured["url"] == "https://api.example.test/v1/messages"
    assert captured["headers"]["x-api-key"] == "ant-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["timeout"] == 12.0
    assert captured["json"]["model"] == "claude-sonnet-4-6"
    assert captured["json"]["system"] == [{"type": "text", "text": "Be terse."}]
    assert captured["json"]["tools"][0]["name"] == "web_search"
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.content == "Found it."
    assert response.choices[0].message.tool_calls[0].function.name == "web_search"
    assert json.loads(response.choices[0].message.tool_calls[0].function.arguments) == {"query": "Hermes"}
    assert response.usage.prompt_tokens == 11
    assert response.usage.prompt_tokens_details.cached_tokens == 2


def test_google_generate_content_client_converts_request_and_response(monkeypatch):
    client = GoogleGenerateContentClient(base_url="https://opencode.ai/zen/v1", api_key="goo-key")
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "responseId": "resp_google_1",
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Need a tool."},
                                {"functionCall": {"name": "web_search", "args": {"query": "Hermes"}}},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 13,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 18,
                    "cachedContentTokenCount": 4,
                    "thoughtsTokenCount": 3,
                },
            },
            request=request,
        )

    monkeypatch.setattr(client._client, "post", fake_post)
    response = client.chat.completions.create(
        model="gemini-3-pro",
        messages=[
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Search for Hermes"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query": "Hermes"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"result": "done"}'},
        ],
        tools=[_tool_def()],
        max_tokens=256,
        timeout=9.0,
    )

    assert captured["url"] == "https://opencode.ai/zen/v1/models/gemini-3-pro:generateContent"
    assert captured["headers"]["x-goog-api-key"] == "goo-key"
    assert captured["timeout"] == 9.0
    assert captured["json"]["systemInstruction"] == {"parts": [{"text": "Use tools when needed."}]}
    assert captured["json"]["tools"][0]["functionDeclarations"][0]["name"] == "web_search"
    assert any(part.get("functionCall") for part in captured["json"]["contents"][1]["parts"])
    assert any(part.get("functionResponse") for part in captured["json"]["contents"][2]["parts"])
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.content == "Need a tool."
    assert response.choices[0].message.tool_calls[0].function.name == "web_search"
    assert json.loads(response.choices[0].message.tool_calls[0].function.arguments) == {"query": "Hermes"}
    assert response.usage.prompt_tokens == 13
    assert response.usage.completion_tokens_details.reasoning_tokens == 3
