from __future__ import annotations

import pytest


class FakeCanonClient:
    def __init__(self):
        self.card_requests = []
        self.card_consumes = []
        self.input_requests = []
        self.input_consumes = []
        self.card_interactive = True
        self.closed = False

    async def create_runtime_card_request(self, conversation_id, **kwargs):
        self.card_requests.append((conversation_id, kwargs))
        return {
            "success": True,
            "cardId": kwargs.get("card_id") or "card-1",
            "interactive": self.card_interactive,
        }

    async def consume_runtime_card_response(self, conversation_id, card_id, *, cancel=False):
        self.card_consumes.append((conversation_id, card_id, cancel))
        return {"status": "submitted", "cardId": card_id, "actionId": "approve"}

    async def create_runtime_input_request(self, conversation_id, **kwargs):
        self.input_requests.append((conversation_id, kwargs))
        return {"success": True, "inputId": kwargs["input_id"]}

    async def consume_runtime_input_response(self, conversation_id, input_id, *, cancel=False):
        self.input_consumes.append((conversation_id, input_id, cancel))
        return {
            "status": "submitted",
            "inputId": input_id,
            "answers": {"field": "supplier", "new_value": "4018"},
        }

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_runtime_card_helper_requests_and_consumes(monkeypatch):
    import tools.canon_runtime_tool as runtime_tool

    fake = FakeCanonClient()
    monkeypatch.setattr(runtime_tool, "_new_canon_client", lambda: fake)

    result = await runtime_tool.request_canon_runtime_card(
        target="canon:convo-1",
        card={
            "schema": "canon.card.v1",
            "cardId": "card-1",
            "title": "Review",
            "fallbackText": "Review",
            "blocks": [{"kind": "actions", "actions": [{"id": "approve", "label": "Approve"}]}],
        },
        response_user_id="user-1",
        runtime_id="hermes",
        turn_id="turn-1",
        timeout_seconds=30,
        native={"runtime": "hermes"},
    )

    assert result["status"] == "submitted"
    assert result["actionId"] == "approve"
    assert fake.card_requests == [
        (
            "convo-1",
            {
                "card": {
                    "schema": "canon.card.v1",
                    "cardId": "card-1",
                    "title": "Review",
                    "fallbackText": "Review",
                    "blocks": [{"kind": "actions", "actions": [{"id": "approve", "label": "Approve"}]}],
                },
                "card_id": "card-1",
                "expires_at": fake.card_requests[0][1]["expires_at"],
                "response_user_id": "user-1",
                "runtime_id": "hermes",
                "turn_id": "turn-1",
                "native": {"runtime": "hermes"},
            },
        )
    ]
    assert fake.card_consumes == [("convo-1", "card-1", False)]
    assert fake.closed is True


@pytest.mark.asyncio
async def test_runtime_card_helper_does_not_poll_display_only_cards(monkeypatch):
    import tools.canon_runtime_tool as runtime_tool

    fake = FakeCanonClient()
    fake.card_interactive = False
    monkeypatch.setattr(runtime_tool, "_new_canon_client", lambda: fake)

    result = await runtime_tool.request_canon_runtime_card(
        target="canon:convo-1",
        card={
            "schema": "canon.card.v1",
            "cardId": "card-1",
            "title": "Summary",
            "fallbackText": "Summary",
            "blocks": [{"kind": "summary", "text": "No action needed."}],
        },
        timeout_seconds=30,
    )

    assert result["status"] == "displayed"
    assert result["cardId"] == "card-1"
    assert fake.card_consumes == []
    assert fake.closed is True


@pytest.mark.asyncio
async def test_runtime_input_helper_supports_questions_response_user_and_answers(monkeypatch):
    import tools.canon_runtime_tool as runtime_tool

    fake = FakeCanonClient()
    monkeypatch.setattr(runtime_tool, "_new_canon_client", lambda: fake)

    result = await runtime_tool.request_canon_runtime_input(
        target="canon:convo-2",
        input_id="input-1",
        kind="clarify",
        title="Correction",
        prompt="Pick a field",
        questions=[
            {
                "id": "field",
                "question": "Which field?",
                "choices": [{"label": "Supplier", "value": "supplier"}],
            }
        ],
        response_user_id="user-2",
        turn_id="turn-2",
        sensitive=False,
        timeout_seconds=30,
    )

    assert result["status"] == "submitted"
    assert result["answers"] == {"field": "supplier", "new_value": "4018"}
    assert fake.input_requests == [
        (
            "convo-2",
            {
                "input_id": "input-1",
                "kind": "clarify",
                "expires_at": fake.input_requests[0][1]["expires_at"],
                "title": "Correction",
                "prompt": "Pick a field",
                "choices": None,
                "questions": [
                    {
                        "id": "field",
                        "question": "Which field?",
                        "choices": [{"label": "Supplier", "value": "supplier"}],
                    }
                ],
                "response_user_id": "user-2",
                "turn_id": "turn-2",
                "sensitive": False,
                "native": None,
            },
        )
    ]
    assert fake.input_consumes == [("convo-2", "input-1", False)]
    assert fake.closed is True


def test_runtime_control_is_exposed_on_canon_platform_toolset(monkeypatch, tmp_path):
    import tools.canon_runtime_tool  # noqa: F401 - ensure registration
    from gateway.platform_registry import PlatformEntry, platform_registry
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache
    from toolsets import resolve_toolset

    monkeypatch.setenv("CANON_API_KEY", "dummy")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    previous = platform_registry.get("canon")
    platform_registry.register(
        PlatformEntry(
            name="canon",
            label="Canon",
            adapter_factory=lambda _config: object(),
            check_fn=lambda: True,
        )
    )
    try:
        invalidate_check_fn_cache()
        _clear_tool_defs_cache()
        assert "canon" in _get_platform_tools({}, "canon")
        schemas = get_tool_definitions(
            enabled_toolsets=["hermes-canon"],
            quiet_mode=True,
        )
        names = {schema["function"]["name"] for schema in schemas}
        assert "canon_runtime_control" in names
        assert "canon_runtime_control" in resolve_toolset("canon_runtime")
    finally:
        if previous is None:
            platform_registry.unregister("canon")
        else:
            platform_registry.register(previous)
        invalidate_check_fn_cache()
        _clear_tool_defs_cache()
