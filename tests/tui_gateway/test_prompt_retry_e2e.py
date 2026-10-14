"""Regression for #137277: a durable continuation preserves completed tool work."""
from pathlib import Path
import os

import pytest

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall, write_hermes_home


@pytest.mark.parametrize("retry_mode, expected_count", [("retry", 1), ("regenerate", 2)])
def test_provider_429_continuation_reuses_completed_tool_result(monkeypatch, retry_mode, expected_count):
    from run_agent import AIAgent
    from hermes_state import SessionDB
    import model_tools

    count = []

    def counting_tool(*args, **kwargs):
        count.append(1)
        return '{"output":"count=1","exit_code":0}'

    monkeypatch.setattr(model_tools, "handle_function_call", counting_tool)
    home = Path(os.environ["HERMES_HOME"])
    failed_once = []

    def respond(request):
        if not any(row.get("role") == "tool" for row in request["body"]["messages"]):
            return ToolCall("terminal", {"command": "count"}, text="I will count")
        if not failed_once:
            failed_once.append(True)
            return Error(429, "rate limit exceeded")
        return Text("finished using count=1")

    with FakeLLMServer(respond) as provider:
        write_hermes_home(home, provider.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        db = SessionDB(home / "state.db")
        agent = AIAgent(provider="custom", base_url=provider.base_url, api_key="sk-fake", model="fake-model",
                        session_id="retry-continuation", quiet_mode=True, platform="cli", skip_memory=True,
                        enabled_toolsets=["terminal"], max_iterations=5, session_db=db)
        agent._api_max_retries = 1
        agent._auto_recovery_cycles = 0
        try:
            failed = agent.run_conversation("count once then finish", conversation_history=[])
            assert failed.get("failed"), failed
            assert count == [1]
            try:
                history = db.get_messages_as_conversation(agent.session_id, repair_alternation=False, include_row_ids=True)
                from tui_gateway import server
                import threading
                session = {
                    "agent": agent, "session_key": agent.session_id, "history": history,
                    "history_lock": threading.Lock(), "history_version": 0, "running": False,
                    "attached_images": [], "cols": 80, "show_reasoning": False,
                    "tool_progress_mode": "all", "slash_worker": None,
                    "agent_ready": threading.Event(),
                }
                session["agent_ready"].set()
                events = []
                finished = threading.Event()

                def emit(kind, sid, payload=None):
                    events.append((kind, payload))
                    if kind == "message.complete":
                        finished.set()

                monkeypatch.setattr(server, "_get_db", lambda: db)
                monkeypatch.setattr(server, "_emit", emit)
                monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
                monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
                monkeypatch.setattr(server, "_wire_callbacks", lambda *args: None)
                monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *args: None)
                monkeypatch.setattr(server, "_session_cwd", lambda *args: str(home))
                monkeypatch.setattr(server, "_register_session_cwd", lambda *args: None)
                server._start_inflight_turn(session, "count once then finish")
                state = server._TurnRun(agent, None, None, True)
                state.result = failed
                state.prompt_text = "count once then finish"
                payload, _, status = server._complete_turn_payload(session, state, None, 80)
                assert status == "error"
                if retry_mode == "retry":
                    assert payload.get("retry_token"), payload
                    assert server._inflight_snapshot(session)["retry_token"] == payload["retry_token"]
                server._sessions["retry-runtime"] = session
                try:
                    if retry_mode == "retry":
                        method, params = "prompt.retry", {"retry_token": payload["retry_token"]}
                    else:
                        method, params = "prompt.submit", {"text": "count once then finish",
                            "truncate_before_row_id": history[0]["_row_id"],
                            "confirm_truncate": True, "confirm_empty_truncate": True}
                    response = server.handle_request({"id": "retry-1", "method": method, "params": {
                        "session_id": "retry-runtime", **params}})
                    assert response.get("result", {}).get("status") == "streaming", response
                    assert finished.wait(timeout=15), events
                    session["_run_thread"].join(timeout=15)
                    assert not session["_run_thread"].is_alive()
                    terminal = [data for event, data in events if event == "message.complete"]
                    assert terminal[-1]["status"] == "complete", terminal
                    assert terminal[-1]["text"] == "finished using count=1"
                    if retry_mode == "retry":
                        duplicate = server.handle_request({"id": "retry-2", "method": "prompt.retry", "params": {
                            "session_id": "retry-runtime", "retry_token": payload["retry_token"]}})
                        assert duplicate["error"]["code"] == 4034
                finally:
                    server._sessions.pop("retry-runtime", None)
                assert count == [1] * expected_count
                requests = provider.main_requests()
                if retry_mode == "retry":
                    assert requests[-1]["messages"] == requests[-2]["messages"]
                assert requests[-1]["messages"][-1]["role"] == "tool"
                assert sum(row["role"] == "user" for row in db.get_messages(agent.session_id)) == 1
            finally:
                agent._resume_user_row_id = None
        finally:
            agent.close()
            db.close()
