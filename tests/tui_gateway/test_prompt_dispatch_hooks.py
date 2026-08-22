"""Behavior contract for the TUI pre-prompt plugin dispatch gate."""

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tui_gateway import server


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _RecordingDB:
    def __init__(self):
        self.batches = []

    def append_messages_batch(self, session_id, messages, **_kwargs):
        self.batches.append((session_id, messages))
        return len(messages)


class _FailingDB:
    def append_messages_batch(self, *_args, **_kwargs):
        raise RuntimeError("database unavailable")


def _worker_session(agent, image_path: Path | None = None):
    return {
        "agent": agent,
        "session_key": "stored-ios",
        "source": "ios",
        "required_prompt_handler": "hoppe_ocr_approval",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [str(image_path)] if image_path else [],
        "image_counter": 1 if image_path else 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
    }


@pytest.fixture()
def worker_env(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda name, sid, payload=None: events.append((name, sid, payload)))
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_args: None)
    monkeypatch.setattr(server, "record_turn_start", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_args: None)
    return events


def test_pre_prompt_dispatch_is_a_public_plugin_hook():
    """Removing the public registration point must break its real consumer."""
    from hermes_cli.plugins import VALID_HOOKS

    assert "pre_prompt_dispatch" in VALID_HOOKS


def test_required_image_turn_blocks_without_matching_directive():
    """A removed or unloaded required plugin must not expose the image to the agent."""
    from tui_gateway.prompt_dispatch_hooks import (
        REQUIRED_HANDLER_UNAVAILABLE_TEXT,
        resolve_prompt_dispatch_results,
    )

    decision = resolve_prompt_dispatch_results(
        [],
        required_prompt_handler="hoppe_ocr_approval",
        has_images=True,
    )

    assert decision.action == "block"
    assert decision.handler == "hoppe_ocr_approval"
    assert decision.text == REQUIRED_HANDLER_UNAVAILABLE_TEXT
    assert decision.reason == "required_prompt_handler_unavailable"


def test_required_image_turn_ignores_foreign_and_handlerless_directives():
    """Only the client-required handler may decide its protected image turn."""
    from tui_gateway.prompt_dispatch_hooks import (
        PromptDispatchDecision,
        resolve_prompt_dispatch_results,
    )

    decision = resolve_prompt_dispatch_results(
        [
            {"action": "allow"},
            {"action": "respond", "handler": "other", "text": "wrong"},
            {
                "action": "respond",
                "handler": "hoppe_ocr_approval",
                "text": "Freigabe angelegt",
                "reason": "ocr_approval_created",
            },
        ],
        required_prompt_handler="hoppe_ocr_approval",
        has_images=True,
    )

    assert decision == PromptDispatchDecision(
        action="respond",
        handler="hoppe_ocr_approval",
        text="Freigabe angelegt",
        reason="ocr_approval_created",
    )


def test_required_image_turn_accepts_matching_explicit_allow():
    """A matching handler can deliberately release a protected image turn."""
    from tui_gateway.prompt_dispatch_hooks import resolve_prompt_dispatch_results

    decision = resolve_prompt_dispatch_results(
        [
            {"action": "allow", "handler": "hoppe_ocr_approval"},
            {
                "action": "respond",
                "handler": "hoppe_ocr_approval",
                "text": "too late",
            },
        ],
        required_prompt_handler="hoppe_ocr_approval",
        has_images=True,
    )

    assert decision.action == "allow"
    assert decision.handler == "hoppe_ocr_approval"


def test_optional_desktop_image_uses_first_valid_response():
    """Desktop images retain OCR interception without claiming manual acceptance."""
    from tui_gateway.prompt_dispatch_hooks import resolve_prompt_dispatch_results

    decision = resolve_prompt_dispatch_results(
        [
            None,
            {"action": "allow"},
            {"action": "respond", "handler": "", "text": ""},
            {
                "action": "respond",
                "handler": "hoppe_ocr_approval",
                "text": "Desktop-Inbox",
            },
            {
                "action": "respond",
                "handler": "later",
                "text": "must not win",
            },
        ],
        required_prompt_handler=None,
        has_images=True,
    )

    assert decision.action == "respond"
    assert decision.handler == "hoppe_ocr_approval"
    assert decision.text == "Desktop-Inbox"


def test_plain_text_without_directive_allows_agent():
    """The image-only client policy must not block ordinary iOS text chat."""
    from tui_gateway.prompt_dispatch_hooks import resolve_prompt_dispatch_results

    decision = resolve_prompt_dispatch_results(
        [],
        required_prompt_handler="hoppe_ocr_approval",
        has_images=False,
    )

    assert decision.action == "allow"


def test_hook_invocation_exposes_only_immutable_public_payload(monkeypatch):
    """Plugins receive the stable public values, never mutable session internals."""
    from hermes_cli import lifecycle
    from tui_gateway.prompt_dispatch_hooks import invoke_pre_prompt_dispatch

    captured: dict[str, Any] = {}

    def capture(name: str, **kwargs: Any) -> list[dict[str, str]]:
        captured["name"] = name
        captured.update(kwargs)
        return [
            {
                "action": "respond",
                "handler": "hoppe_ocr_approval",
                "text": "Inbox",
            }
        ]

    monkeypatch.setattr(lifecycle, "invoke_hook", capture)

    decision = invoke_pre_prompt_dispatch(
        session_id="ios-ui",
        session_key="stored-ios",
        source="ios",
        text="Kontakt prüfen",
        attached_images=["/tmp/card.png"],
        required_prompt_handler="hoppe_ocr_approval",
    )

    assert decision.action == "respond"
    assert captured == {
        "name": "pre_prompt_dispatch",
        "session_id": "ios-ui",
        "session_key": "stored-ios",
        "surface": "tui",
        "source": "ios",
        "text": "Kontakt prüfen",
        "attached_images": ("/tmp/card.png",),
        "required_prompt_handler": "hoppe_ocr_approval",
    }


def test_hook_dispatch_failure_blocks_only_required_image_turn(monkeypatch):
    """A host/plugin dispatch failure closes the protected path without breaking text."""
    from hermes_cli import lifecycle
    from tui_gateway.prompt_dispatch_hooks import invoke_pre_prompt_dispatch

    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )

    blocked = invoke_pre_prompt_dispatch(
        session_id="ios-ui",
        session_key="stored-ios",
        source="ios",
        text="Bild",
        attached_images=["/tmp/card.png"],
        required_prompt_handler="hoppe_ocr_approval",
    )
    allowed = invoke_pre_prompt_dispatch(
        session_id="ios-ui",
        session_key="stored-ios",
        source="ios",
        text="Hallo",
        attached_images=[],
        required_prompt_handler="hoppe_ocr_approval",
    )

    assert blocked.action == "block"
    assert allowed.action == "allow"


def test_worker_blocks_required_image_before_agent_and_persists_response(
    monkeypatch,
    tmp_path,
    worker_env,
):
    """A missing required handler must become one durable assistant turn."""
    from tui_gateway.prompt_dispatch_hooks import PromptDispatchDecision

    image_path = tmp_path / "required.png"
    image_path.write_bytes(b"image")
    db = _RecordingDB()
    agent = SimpleNamespace(
        session_id="stored-ios",
        _session_db=db,
        clear_interrupt=lambda: None,
        run_conversation=lambda *_args, **_kwargs: pytest.fail("agent must not run"),
    )
    session = _worker_session(agent, image_path)
    monkeypatch.setattr(
        server,
        "invoke_pre_prompt_dispatch",
        lambda **_kwargs: PromptDispatchDecision(
            action="block",
            handler="hoppe_ocr_approval",
            text="Freigabe-Handler nicht verfügbar",
            reason="required_prompt_handler_unavailable",
        ),
        raising=False,
    )

    assert server._run_prompt_submit("rid", "ios-ui", session, "Bitte prüfen")

    assert session["attached_images"] == []
    assert session["history"] == [
        {
            "role": "user",
            "content": f"Bitte prüfen\n@image:{image_path}",
        },
        {"role": "assistant", "content": "Freigabe-Handler nicht verfügbar"},
    ]
    assert db.batches == [("stored-ios", session["history"])]
    assert [event[0] for event in worker_env] == [
        "message.start",
        "message.delta",
        "message.complete",
    ]
    assert worker_env[-1][2]["status"] == "complete"


def test_worker_uses_matching_hook_response_without_agent(
    monkeypatch,
    tmp_path,
    worker_env,
):
    """A plugin response closes the turn before image routing or model work."""
    from tui_gateway.prompt_dispatch_hooks import PromptDispatchDecision

    image_path = tmp_path / "contact.png"
    image_path.write_bytes(b"image")
    db = _RecordingDB()
    calls = []
    agent = SimpleNamespace(
        session_id="stored-ios",
        _session_db=db,
        clear_interrupt=lambda: None,
        run_conversation=lambda *_args, **_kwargs: calls.append("agent"),
    )
    session = _worker_session(agent, image_path)
    monkeypatch.setattr(
        server,
        "invoke_pre_prompt_dispatch",
        lambda **kwargs: (
            calls.append(("hook", kwargs)),
            PromptDispatchDecision(
                action="respond",
                handler="hoppe_ocr_approval",
                text="Freigabe angelegt",
                reason="ocr_approval_created",
            ),
        )[1],
        raising=False,
    )

    server._run_prompt_submit("rid", "ios-ui", session, "Kontakt prüfen")

    assert [kind for kind, *_rest in calls] == ["hook"]
    assert calls[0][1]["attached_images"] == [str(image_path)]
    assert session["history"][-1] == {
        "role": "assistant",
        "content": "Freigabe angelegt",
    }
    assert len(db.batches) == 1
    assert worker_env[-1][2]["text"] == "Freigabe angelegt"


def test_worker_persistence_failure_never_falls_through_to_agent(
    monkeypatch,
    tmp_path,
    worker_env,
):
    """A durable-write failure must close as an error, not leak the protected image."""
    from tui_gateway.prompt_dispatch_hooks import PromptDispatchDecision

    image_path = tmp_path / "protected.png"
    image_path.write_bytes(b"image")
    calls = []
    agent = SimpleNamespace(
        session_id="stored-ios",
        _session_db=_FailingDB(),
        clear_interrupt=lambda: None,
        run_conversation=lambda *_args, **_kwargs: calls.append("agent"),
    )
    session = _worker_session(agent, image_path)
    monkeypatch.setattr(
        server,
        "invoke_pre_prompt_dispatch",
        lambda **_kwargs: PromptDispatchDecision(
            action="respond",
            handler="hoppe_ocr_approval",
            text="Freigabe angelegt",
        ),
    )

    server._run_prompt_submit("rid", "ios-ui", session, "Kontakt prüfen")

    assert calls == []
    assert session["history"] == []
    assert worker_env[-1][0] == "message.complete"
    assert worker_env[-1][2]["status"] == "error"
    assert worker_env[-1][2]["recoverable"] is True


def test_worker_allows_plain_text_to_reach_agent(monkeypatch, worker_env):
    """The new dispatch point must preserve ordinary text turns."""
    calls = []
    agent = SimpleNamespace(
        session_id="stored-ios",
        clear_interrupt=lambda: None,
        run_conversation=lambda message, **_kwargs: (
            calls.append(message),
            {"final_response": "normal", "messages": []},
        )[1],
    )
    session = _worker_session(agent)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *_args: (threading.Event(), _InlineThread()))

    server._run_prompt_submit("rid", "ios-ui", session, "Hallo")

    assert calls == ["Hallo"]
    assert worker_env[-1][2]["text"] == "normal"
