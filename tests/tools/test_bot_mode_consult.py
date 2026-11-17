"""message_agent(mode="consult"): a teammate answers in the tool call, on one private thread per asking conversation.

Only the model is scripted (``run_agent.AIAgent``): Bot Mode gating, the consult roster, profile routing and its
scope binding, child build, SessionDB lineage and the delegate_task continue path are the real ones.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml

from hermes_state import SessionDB
from tools import bot_mode_probe


class _ScriptedModel:
    """Stands in for AIAgent: records the profile home each child was built and ran in, persists each turn."""

    def __init__(self):
        self.builds, self.turns = [], []

    def __call__(self, **kw):
        from hermes_constants import get_hermes_home

        child = MagicMock()
        child.session_id = kw.get("session_id") or f"consult-child-{len(self.builds)}"
        child.model, child.provider = kw.get("model"), kw.get("provider")
        child._parent_session_id = kw.get("parent_session_id")
        child._session_init_model_config = {}  # the builder stamps the lineage markers here, as on AIAgent
        child.get_activity_summary.return_value = {"api_call_count": 1, "current_tool": None, "last_activity_ts": 0}
        db = kw["session_db"]

        def run_conversation(user_message, conversation_history=None, **_):
            self.turns.append({"session_id": child.session_id, "message": user_message,
                               "history": conversation_history, "home": str(get_hermes_home())})
            reply = f"answer #{len(self.turns)}"
            if db.get_session(child.session_id) is None:  # AIAgent's first-turn row, with its init model_config
                db.create_session(child.session_id, "subagent", parent_session_id=kw.get("parent_session_id"),
                                  model_config=dict(child._session_init_model_config))
            db.append_message(child.session_id, "user", user_message)
            db.append_message(child.session_id, "assistant", reply)
            return {"final_response": reply, "completed": True, "api_calls": 1}

        child.run_conversation.side_effect = run_conversation
        self.builds.append({**kw, "home": str(get_hermes_home())})
        return child


def _write_profile(home, *, host, model, key, extra=None):
    home.mkdir(parents=True, exist_ok=True)
    cfg = {"model": {"provider": "custom", "base_url": f"https://{host}/v1", "default": model, "key_env": "PROFILE_KEY"}}
    (home / "config.yaml").write_text(yaml.safe_dump({**cfg, **(extra or {})}), encoding="utf-8")
    (home / ".env").write_text(f"PROFILE_KEY={key}\n", encoding="utf-8")
    (home / "profile.yaml").write_text("ui_meta:\n  hermes-bots: {}\n", encoding="utf-8")


def _asker(db, session_id):
    agent = MagicMock()
    agent.base_url, agent.api_key, agent.provider = "https://launch.invalid/v1", "launch-key", "custom"
    agent.api_mode, agent.model, agent.platform = "chat_completions", "launch-model", "cli"
    agent.providers_allowed = agent.providers_ignored = agent.providers_order = agent.provider_sort = None
    agent.session_id, agent._delegate_depth, agent._interrupt_requested = session_id, 0, False
    agent.enabled_toolsets, agent.disabled_toolsets = ["terminal", "file"], []
    agent._session_db, agent._session_title_hint, agent._bot_mode_protocol = db, "Bot Chat", True
    agent._turn_author = None
    db.create_session(session_id, "cli")
    agent._active_children, agent._active_children_lock = [], threading.Lock()
    agent._print_fn = agent.tool_progress_callback = agent.thinking_callback = None
    return agent


def _install(tmp_path, monkeypatch, roster):
    launch = tmp_path / "hermes-root"
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key",
                   extra={"bot_mode": {"consult_roster": roster}})
    for name in ("coder", "scout"):
        _write_profile(launch / "profiles" / name, host=f"{name}.invalid", model=f"{name}-model", key=f"{name}-key")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("PROFILE_KEY", "launch-key")
    bot_mode_probe._reset_cache_for_tests()
    return launch


def test_consult_returns_the_teammates_answer_on_one_thread_per_asking_conversation(tmp_path, monkeypatch):
    from tools.bot_mode_dm import message_agent_tool, message_agent_tool_schema
    from tools.delegate_tool_registry import _recent_subagents

    launch = _install(tmp_path, monkeypatch, ["coder"])
    assert message_agent_tool_schema()["function"]["parameters"]["properties"]["mode"]["enum"] == ["message", "consult"]
    db = SessionDB(db_path=launch / "state.db")
    first, other = _asker(db, "asker-a"), _asker(db, "asker-b")
    model = _ScriptedModel()
    with patch("run_agent.AIAgent", side_effect=model):
        one = json.loads(message_agent_tool(target="coder", message="Is the parser thread-safe?", mode="consult",
                                            agent=first))
        two = json.loads(message_agent_tool(target="@coder", message="And the lexer?", mode="consult", agent=first))
        elsewhere = json.loads(message_agent_tool(target="coder", message="Unrelated ask", mode="consult",
                                                  agent=other))
        _recent_subagents.clear()  # a gateway restart: only state.db remembers the thread
        later = json.loads(message_agent_tool(target="coder", message="One more", mode="consult", agent=first))

    assert (one["status"], one["answer"], one["resumed"]) == ("answered", "answer #1", False), one
    assert (two["status"], two["answer"], two["resumed"]) == ("answered", "answer #2", True), two
    assert two["subagent_id"] == one["subagent_id"] and two["session_id"] == one["session_id"]
    assert elsewhere["resumed"] is False and elsewhere["session_id"] != one["session_id"]
    assert (later["resumed"], later["session_id"], later["answer"]) == (True, one["session_id"], "answer #4"), later
    # The thread runs as the teammate, inside this process: its profile home, model and key.
    coder = str(launch / "profiles" / "coder")
    assert {b["home"] for b in model.builds} == {coder} and {t["home"] for t in model.turns} == {coder}
    assert {(b["model"], b["api_key"]) for b in model.builds} == {("coder-model", "coder-key")}
    # The follow-up continued the first thread's transcript, seeded with the brief only.
    assert [m["content"] for m in model.turns[1]["history"]][-1] == "answer #1"
    assert "Is the parser thread-safe?" in model.turns[1]["history"][0]["content"]
    row = db.get_session(one["session_id"]) or {}
    assert row["parent_session_id"] == "asker-a" and '"_bot_consult": "coder"' in row["model_config"]


def test_consult_refuses_a_teammate_off_the_roster_and_stays_hidden_without_one(tmp_path, monkeypatch):
    from tools.bot_mode_dm import message_agent_tool, message_agent_tool_schema

    launch = _install(tmp_path, monkeypatch, ["coder"])
    db = SessionDB(db_path=launch / "state.db")
    model = _ScriptedModel()
    with patch("run_agent.AIAgent", side_effect=model):
        out = json.loads(message_agent_tool(target="scout", message="hi", mode="consult", agent=_asker(db, "a")))
    assert "consult roster" in out["error"] and out["teammates"] == ["coder"] and model.builds == []

    _install(tmp_path, monkeypatch, [])
    assert "mode" not in message_agent_tool_schema()["function"]["parameters"]["properties"]
