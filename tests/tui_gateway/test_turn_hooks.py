"""Exercise Desktop/TUI lifecycle hooks through the shared prompt dispatcher."""

import json
import os
import threading
import uuid
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tui_gateway import server


HANDLER = '''
import json
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

async def handle(event, context):
    home = get_hermes_home()
    with SessionDB(home / "state.db") as db:
        messages = db.get_messages(context["session_id"])
    with (home / "hook-events.jsonl").open("a") as output:
        output.write(json.dumps({"event": event, "context": context, "home": str(home),
            "secret": get_secret("HOOK_TEST_TOKEN"), "messages": [m["content"] for m in messages]}) + "\\n")
    if context["message"] == "handler failure":
        raise RuntimeError("hook failed after recording")
'''


@pytest.fixture
def turn_env(monkeypatch, tmp_path):
    def run_inline(target, **_kwargs):
        target()
        return SimpleNamespace()

    monkeypatch.setattr(server, "_start_session_work", run_inline)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a: None)
    monkeypatch.setattr(server, "_sync_agent_compression_with_config", lambda *a: None)
    monkeypatch.setattr(server, "_sync_agent_fallback_with_config", lambda *a: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_start_turn_voice", lambda: (None, False))
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (threading.Event(), SimpleNamespace(join=lambda: None)))
    # The existing config loader initializes process-wide terminal defaults once.
    # Warm that bridge before measuring whether hook dispatch changes the env.
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv()
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)


def _home(tmp_path, name):
    home = tmp_path / name
    hooks = home / "hooks" / "memory-extractor"
    hooks.mkdir(parents=True)
    (home / ".env").write_text(f"HOOK_TEST_TOKEN={name}\n")
    (home / "config.yaml").write_text("memory:\n  enabled: false\n")
    (hooks / "HOOK.yaml").write_text(f"name: extractor-{name}\nevents: [agent:start, agent:end]\n")
    (hooks / "handler.py").write_text(HANDLER)
    return home


def _turn(home, *, source="desktop", outcome="success", message="remember this", rotate=False):
    with SessionDB(home / "state.db") as db:
        key = f"session-{uuid.uuid4().hex}"
        db.create_session(key, source=source)
        agent = SimpleNamespace(session_id=key, model="active-model", provider="custom", user_id="owner")

        def run_conversation(text, **_kwargs):
            if rotate:
                agent.session_id = key + "-continuation"
                db.create_session(agent.session_id, source=source, parent_session_id=key)
            db.append_message(agent.session_id, "user", text)
            db.append_message(agent.session_id, "assistant", "saved response")
            if outcome == "exception":
                raise RuntimeError("turn failed")
            return {"final_response": "saved response", "model": agent.model, "provider": agent.provider,
                    "failed": outcome == "failed"}

        agent.run_conversation = run_conversation
        session = {"agent": agent, "session_key": key, "profile_home": str(home), "source": source,
                   "auth_user_id": "owner", "history": [], "history_lock": threading.Lock(),
                   "history_version": 0, "running": True, "attached_images": [], "cols": 80,
                   "image_counter": 0, "slash_worker": None, "show_reasoning": False}
        assert server._run_prompt_submit("rid", "ui-session", session, message)
        assert session["running"] is False
        return agent.session_id


def _events(home):
    path = home / "hook-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_memory_hooks_follow_profile_and_read_committed_turns(turn_env, tmp_path):
    first, second = _home(tmp_path, "first"), _home(tmp_path, "second")
    ambient_home, ambient_env = get_hermes_home(), dict(os.environ)
    _turn(first)
    latest = _turn(second, source="tui", rotate=True)
    _turn(first, message="handler failure")

    for home, count in ((first, 2), (second, 1)):
        events = _events(home)
        assert [event["event"] for event in events] == ["agent:start", "agent:end"] * count
        assert all(event["home"] == str(home) and event["secret"] == home.name for event in events)
        for event in events[1::2]:
            assert event["messages"][-1] == "saved response"
            assert event["context"]["response"] == "saved response"
            assert event["context"]["model"] == "active-model"
            assert event["context"]["provider"] == "custom"
            assert event["context"]["user_id"] == "owner"
    assert _events(first)[0]["context"]["platform"] == "desktop"
    assert _events(second)[-1]["context"]["platform"] == "tui"
    assert _events(second)[-1]["context"]["session_id"] == latest
    assert get_hermes_home() == ambient_home
    assert dict(os.environ) == ambient_env


@pytest.mark.parametrize("outcome", ["failed", "exception"])
def test_started_turn_has_one_end_hook_on_failure(turn_env, tmp_path, outcome):
    home = _home(tmp_path, "failure")
    _turn(home, outcome=outcome, message="x" * 700)
    events = _events(home)
    assert [event["event"] for event in events] == ["agent:start", "agent:end"]
    assert len(events[0]["context"]["message"]) == 500


def test_rejected_input_emits_no_agent_hooks(turn_env, monkeypatch, tmp_path):
    home = _home(tmp_path, "rejected")
    monkeypatch.setattr(server, "_prepare_turn_input", lambda *a: None)
    _turn(home)
    assert _events(home) == []
