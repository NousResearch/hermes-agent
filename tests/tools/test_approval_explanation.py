"""An approval prompt says, in plain English, what the command is about to do.

The line exists because the prompt's only other context is the detector's pattern label
("recursive delete") over a raw command: fine for someone who can read the command, useless for
anyone else. These tests pin the behaviour contract — which producer wins, that a broken model
degrades to the built-in sentence instead of failing the gate, and that the sentence actually
reaches each surface's payload — not the wording of any individual explanation.
"""

from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from tools import approval, approval_context, approval_explain


def _config(monkeypatch, **approvals):
    """Pin the ``approvals`` block the producer reads (``{"explain": ...}`` and friends)."""
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(approvals))


def _model_says(text: str, calls: dict | None = None):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def fake_call_llm(**kwargs):
        if calls is not None:
            calls.update(kwargs)
        return response

    return fake_call_llm


def _model_explodes(**kwargs):
    raise RuntimeError("provider down")


def test_static_explanation_needs_no_model_and_names_the_family(monkeypatch):
    _config(monkeypatch, explain="static")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _model_explodes)

    assert "cannot be undone" in approval_explain.explanation_for(
        "rm -rf ~/build", "recursive delete", "recursive delete")
    # An unknown label gets nothing rather than an invented guess.
    assert approval_explain.explanation_for("echo hi", "mystery pattern", "mystery pattern") is None


def test_off_never_asks_a_model(monkeypatch):
    _config(monkeypatch, explain="off")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _model_explodes)

    assert approval_explain.explanation_for("rm -rf ~", "recursive delete", "recursive delete") is None


def test_aux_explanation_is_used_with_thinking_off(monkeypatch):
    _config(monkeypatch, explain="aux")
    calls: dict = {}
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _model_says("Deletes the build folder.", calls))

    assert approval_explain.explanation_for(
        "rm -rf ~/build", "recursive delete", "recursive delete") == "Deletes the build folder."
    assert calls["task"] == "approval"
    assert calls["reasoning_config"] == {"enabled": False}
    assert calls["max_tokens"] <= 200
    assert "<command>" in calls["messages"][1]["content"]


def test_a_broken_model_falls_back_to_the_built_in_line(monkeypatch):
    _config(monkeypatch, explain="aux")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _model_explodes)

    # Not None and not an exception: the prompt still explains itself, just less specifically.
    assert "cannot be undone" in approval_explain.explanation_for(
        "rm -rf ~/build", "recursive delete", "recursive delete")


def test_plugin_hook_outranks_both_built_in_producers(monkeypatch):
    _config(monkeypatch, explain="aux")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _model_says("Model sentence."))
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook",
                        lambda name, **kwargs: ["Operator sentence."] if name == "approval_explanation" else [])

    assert approval_explain.explanation_for(
        "rm -rf ~/build", "recursive delete", "recursive delete") == "Operator sentence."


def test_an_empty_hook_answer_leaves_the_built_in_line(monkeypatch):
    _config(monkeypatch, explain="static")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook",
                        lambda name, **kwargs: ["", None] if name == "approval_explanation" else [])

    assert "cannot be undone" in approval_explain.explanation_for(
        "rm -rf ~/build", "recursive delete", "recursive delete")


def test_cli_prompt_receives_the_explanation(monkeypatch):
    _config(monkeypatch, explain="static")
    seen: dict = {}

    def callback(command, description, **kwargs):
        seen.update(command=command, description=description, **kwargs)
        return "once"

    result = approval._human_decision(
        approval._COMMAND_GATE, command="rm -rf ~/build", description="recursive delete",
        pattern_key="recursive delete", pattern_keys=["recursive delete"], session_key="explain-cli",
        approval_callback=callback, is_cli=True, is_gateway=False, is_ask=False)

    assert result["approved"] is True
    assert "cannot be undone" in seen["explanation"]
    assert seen["command"] == "rm -rf ~/build"  # display-only: the command itself is untouched


class _CardAdapter(BasePlatformAdapter):
    """Concrete adapter that records the card it was asked to render."""

    MAX_MESSAGE_LENGTH = 1000

    def __init__(self) -> None:
        self.prompts: list = []

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover - unused
        return SendResult(success=True, message_id="m1")

    async def _send_exec_approval_prompt(self, prompt):
        self.prompts.append(prompt)
        return SendResult(success=True, message_id="card")


@pytest.mark.asyncio
async def test_messaging_card_renders_the_explanation_line():
    adapter = _CardAdapter()

    await adapter.send_exec_approval(
        "c1", "rm -rf /tmp/x", "sk", description="recursive delete", request_id="r1",
        explanation="Permanently deletes files and folders; nothing goes to the trash.")

    prompt = adapter.prompts[-1]
    assert "What it does: Permanently deletes files and folders" in prompt.text
    assert prompt.explanation.startswith("Permanently deletes")
    # Still an ordinary approval: the line adds context, it does not change the choices offered.
    assert prompt.choices == ["once", "session", "always", "deny"]


def test_desktop_payload_carries_the_explanation():
    from tui_gateway.server import _approval_request_payload

    payload = _approval_request_payload({
        "command": "rm -rf /tmp/x", "description": "recursive delete",
        "explanation": "Permanently deletes files and folders.", "request_id": "r1",
    })

    assert payload["explanation"] == "Permanently deletes files and folders."
