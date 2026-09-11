"""Native TUI timestamp intake reaches the common agent request boundary."""

from copy import deepcopy
import base64
from datetime import datetime
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest


STAMP = datetime(2026, 8, 20, 12, 0, tzinfo=ZoneInfo("UTC")).timestamp()


def _png_bytes():
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
    )


def test_tui_prompt_turn_stages_clean_persistence_and_timestamped_provider_request(tmp_path, monkeypatch):
    """Desktop/Relay's native TUI path passes its staged text through the real agent loop."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tui_gateway import server

    captured = []
    db = SessionDB(db_path=tmp_path / "state.db")

    def respond(kwargs):
        captured.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop",
            )],
            model="test-model", usage=None,
        )

    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="chat_completions", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, session_db=db, session_id="timestamp-tui",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent._disable_streaming = True
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": True},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr("agent.turn_context.time.time", lambda: STAMP)
        monkeypatch.setattr(agent, "_interruptible_api_call", respond)
        ticker_stop = threading.Event()
        monkeypatch.setattr(
            server, "_start_usage_ticker",
            lambda _sid, _agent: (ticker_stop, SimpleNamespace(join=lambda: None)),
        )
        history = [
            {"role": "user", "content": "earlier question", "timestamp": STAMP},
            {"role": "assistant", "content": "earlier answer"},
        ]
        session = {"session_key": "timestamp-tui", "history_lock": threading.RLock()}
        st = server._TurnRun(agent, None, None, False, history=history)

        server._invoke_agent(
            "ui-session", session, st, "current question", "current question", None, [], None, None,
        )

        assert st.result["completed"] is True
        assert st.run_kwargs["persist_user_message"] == "current question"
        wire_users = [message["content"] for message in captured[0]["messages"] if message["role"] == "user"]
        assert wire_users == [
            "[Thu 2026-08-20 12:00:00 UTC] earlier question",
            "[Thu 2026-08-20 12:00:00 UTC] current question",
        ]
        persisted = db.get_messages_as_conversation("timestamp-tui")
        persisted_user = next(message for message in persisted if message["role"] == "user")
        assert persisted_user["content"] == "current question"
        assert persisted_user["timestamp"] == STAMP
    finally:
        db.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_tui_native_image_caption_is_prepared_before_packaging_and_persists_cleanly(
    tmp_path, monkeypatch, enabled,
):
    """Native caption intake mirrors gateway text preparation before image parts exist."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tui_gateway import server

    image = tmp_path / "caption.png"
    image.write_bytes(_png_bytes())
    captured = []
    db = SessionDB(db_path=tmp_path / "state.db")

    def respond(kwargs):
        captured.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop",
            )],
            model="test-model", usage=None,
        )

    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="chat_completions", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=db, session_id=f"timestamp-native-image-{enabled}",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent._disable_streaming = True
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": enabled}, "agent": {"image_input_mode": "native"},
        })
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "message_timestamps": {"enabled": enabled}, "agent": {"image_input_mode": "native"},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr("tui_gateway.server.time.time", lambda: STAMP)
        monkeypatch.setattr("agent.image_routing.decide_image_input_mode", lambda *args, **kwargs: "native")
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        monkeypatch.setattr(agent, "_interruptible_api_call", respond)

        clean, timestamp = server._prepare_native_image_caption(
            "[2026-08-20T11:59:00+00:00] describe this"
        )
        parts = server._route_turn_images(
            agent, clean, [str(image)],
            timestamp=timestamp,
        )
        session = {"session_key": agent.session_id, "history_lock": threading.RLock()}
        st = server._TurnRun(agent, None, None, False, persist_user_timestamp=timestamp)
        server._invoke_agent("ui-session", session, st, clean, parts, None, [str(image)], None, None)

        expected_caption = "[Thu 2026-08-20 12:00:00 UTC] describe this" if enabled else "describe this"
        text_part = next(part for part in parts if part["type"] == "text")
        assert text_part["text"].startswith(expected_caption)
        assert [part["type"] for part in parts] == ["text", "image_url"]
        outgoing = next(message for message in captured[0]["messages"] if message["role"] == "user")
        assert outgoing["content"] == parts
        persisted = db.get_messages_as_conversation(agent.session_id)
        user = next(message for message in persisted if message["role"] == "user")
        assert user["content"].startswith("describe this")
        assert "[Thu 2026-08-20" not in user["content"]
        assert user["timestamp"] == STAMP
    finally:
        db.close()


def test_tui_native_empty_caption_does_not_invent_a_timestamp_text_part(monkeypatch):
    from tui_gateway import server

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "message_timestamps": {"enabled": True},
    })
    monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
    monkeypatch.setattr("tui_gateway.server.time.time", lambda: STAMP)

    clean, timestamp = server._prepare_native_image_caption("")

    assert clean == ""
    assert server._render_native_image_caption(clean, timestamp) == ""
    assert timestamp == STAMP


@pytest.mark.parametrize(
    ("outcome", "enabled", "caption"),
    [
        ("native", True, "[Thu 2026-08-20 11:59:00 UTC] describe this"),
        ("native", False, ""),
        ("text", True, "[Thu 2026-08-20 11:59:00 UTC] describe this"),
        ("text", False, "[2026-08-20T11:59:00+00:00] describe this"),
        ("skipped", True, ""),
        ("skipped", False, "describe this"),
        ("exception", True, "[2026-08-20T11:59:00+00:00] describe this"),
        ("exception", False, "[Thu 2026-08-20 11:59:00 UTC] describe this"),
    ],
)
def test_tui_image_routing_outcomes_use_one_timestamped_caption_boundary(
    tmp_path, monkeypatch, outcome, enabled, caption,
):
    """Native success and every text fallback preserve one clean admission-time caption."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tui_gateway import server

    image = tmp_path / "caption.png"
    image.write_bytes(_png_bytes())
    captured = []
    db = SessionDB(db_path=tmp_path / "state.db")

    def respond(kwargs):
        captured.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop",
            )],
            model="test-model", usage=None,
        )

    try:
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai-compat",
            model="test-model", api_mode="chat_completions", max_iterations=1,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, session_db=db,
            session_id=f"timestamp-image-{outcome}-{enabled}",
        )
        agent._cached_system_prompt = "Stable synthetic system prompt."
        agent._disable_streaming = True
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
            "message_timestamps": {"enabled": enabled}, "agent": {"image_input_mode": "native"},
        })
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "message_timestamps": {"enabled": enabled}, "agent": {"image_input_mode": "native"},
        })
        monkeypatch.setattr("hermes_time.get_timezone", lambda: ZoneInfo("UTC"))
        monkeypatch.setattr("tui_gateway.server.time.time", lambda: STAMP)
        monkeypatch.setattr("agent.turn_context.time.time", lambda: STAMP)
        monkeypatch.setattr(
            "agent.image_routing.decide_image_input_mode",
            lambda *args, **kwargs: "text" if outcome == "text" else "native",
        )
        if outcome == "skipped":
            monkeypatch.setattr(
                "agent.image_routing.build_native_content_parts",
                lambda prompt, paths: ([{"type": "text", "text": prompt}], paths),
            )
        elif outcome == "exception":
            def _raise_native_packaging(*_args, **_kwargs):
                raise OSError("synthetic native packaging failure")

            monkeypatch.setattr("agent.image_routing.build_native_content_parts", _raise_native_packaging)
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        monkeypatch.setattr(agent, "_interruptible_api_call", respond)
        ticker_stop = threading.Event()
        monkeypatch.setattr(
            server, "_start_usage_ticker",
            lambda _sid, _agent: (ticker_stop, SimpleNamespace(join=lambda: None)),
        )

        clean, timestamp = server._prepare_native_image_caption(caption)
        run_message = server._route_turn_images(
            agent, clean, [str(image)],
            timestamp=timestamp,
        )
        expected_caption = "describe this" if caption else ""
        expected_prefix = "[Thu 2026-08-20 12:00:00 UTC] "
        if outcome == "native":
            assert [part["type"] for part in run_message] == ["text", "image_url"]
            text_parts = [part["text"] for part in run_message if part["type"] == "text"]
            assert len(text_parts) == 1
            assert text_parts[0].count(expected_prefix) == (1 if enabled and expected_caption else 0)
            if expected_caption:
                expected_native_caption = (
                    f"{expected_prefix}{expected_caption}" if enabled else expected_caption)
                assert text_parts[0].startswith(expected_native_caption)
            else:
                assert text_parts[0].startswith("What do you see in this image?")
        else:
            session = {"session_key": agent.session_id, "history_lock": threading.RLock()}
            st = server._TurnRun(agent, None, None, False, persist_user_timestamp=timestamp)
            server._invoke_agent("ui-session", session, st, clean, run_message, None, [str(image)], None, None)
            outgoing = next(message for message in captured[0]["messages"] if message["role"] == "user")
            assert isinstance(outgoing["content"], str)
            assert outgoing["content"].count(expected_prefix) == (1 if enabled else 0)
            assert outgoing["content"].endswith(expected_caption)
            assert f"[The user attached an image: {image.name}]" in outgoing["content"]
            assert f"image_url: {image}" in outgoing["content"]
            assert "[2026-08-20T11:59:00+00:00]" not in outgoing["content"]
            assert "[Thu 2026-08-20 11:59:00 UTC]" not in outgoing["content"]
            persisted = db.get_messages_as_conversation(agent.session_id)
            user = next(message for message in persisted if message["role"] == "user")
            assert user["content"] == (
                f"{expected_caption}\n@image:{image}" if expected_caption else f"@image:{image}")
            assert user["timestamp"] == STAMP
    finally:
        db.close()


def test_tui_codex_app_server_forced_text_route_does_not_receive_native_caption_rendering(
    tmp_path, monkeypatch,
):
    """Codex app-server bypasses common request rendering, so it stays on its clean text route."""
    from tui_gateway import server

    image = tmp_path / "caption.png"
    image.write_bytes(_png_bytes())
    agent = SimpleNamespace(provider="openai-compat", model="test-model", api_mode="codex_app_server")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"agent": {"image_input_mode": "native"}})
    monkeypatch.setattr("agent.image_routing.decide_image_input_mode", lambda *args, **kwargs: "native")

    routed = server._route_turn_images(
        agent, "describe this", [str(image)],
        timestamp=STAMP,
    )

    assert routed == (
        f"[The user attached an image: {image.name}]\n"
        f"[Examine it with the vision_analyze tool using image_url: {image}]\n\n"
        "describe this"
    )
