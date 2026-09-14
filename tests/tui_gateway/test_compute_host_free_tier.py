"""Isolated turns preserve admission and queued ownership across the wire."""
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def isolated_turn(tmp_path, monkeypatch):
    from hermes_cli import auth, auth_nous, free_tier_usage as usage
    from hermes_constants import get_hermes_home, set_hermes_home_override
    from run_agent import AIAgent
    from tui_gateway import server
    from tui_gateway.compute_host import ComputeHost

    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    guest = {"auth_method": "anonymous", "anon_token": "anon_isolated_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    identity = usage.current_identity()
    assert identity is not None
    for _ in range(usage.TOOL_CALL_CAP - 1):
        usage.record_completed_tool(identity)
    agent = AIAgent(api_key="fixture", provider="nous", model="nous/welcome",
                    base_url="https://welcome-api.nousresearch.com/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent._cached_system_prompt = "unchanged-prefix"
    monkeypatch.setattr(agent, "_interruptible_api_call", Mock(side_effect=AssertionError("no network")))
    home = str(get_hermes_home())
    history = [{"role": "user", "content": "before"}, {"role": "assistant", "content": "done"}]

    def session(owner):
        return {"agent": owner, "session_key": agent.session_id, "history": list(history),
                "history_version": 1, "history_lock": threading.RLock(), "profile_home": home,
                "running": False, "attached_images": [], "cols": 80, "source": "desktop",
                "cwd": str(tmp_path)}

    parent, child = session(None), session(agent)
    parent.update(_compute_host_active=True, _metadata_mirror={
        "runtime": server._runtime_model_config(agent)})
    submissions, delivered = [], []
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    supervisor = SimpleNamespace(submit_turn=lambda frame, on_complete: submissions.append((frame, on_complete)))
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *a: supervisor)
    write_json = server.write_json
    monkeypatch.setattr(server, "_make_agent", Mock(side_effect=AssertionError("parent must stay agentless")))

    def prepare(sid, sess, st, text, images):
        st.scopes.home = set_hermes_home_override(sess["profile_home"])
        st.history = list(sess["history"])
        st.history_version = sess["history_version"]
        return text, text, 80, None

    monkeypatch.setattr(server, "_prepare_turn_input", prepare)
    for name in ("_ensure_active_session_slot", "_sync_session_key_after_compress",
                 "_publish_session_control_snapshot", "_emit_settled_session_info",
                 "_probe_credentials"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)

    def run_child(frame):
        monkeypatch.setattr(server, "write_json", write_json)
        monkeypatch.setattr(server, "_sessions", {"s": child})
        monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
        host._run_real_turn(json.loads(json.dumps(frame)))
        monkeypatch.delenv("HERMES_COMPUTE_HOST_CHILD")
        monkeypatch.setattr(server, "_sessions", {"s": parent})
        monkeypatch.setattr(server, "write_json", lambda frame: delivered.append(frame) or True)
        frames = [json.loads(line) for line in out.getvalue().splitlines()]
        for item in frames:
            if item["type"] == "rpc":
                server._relay_compute_host_rpc(item["message"])
        return frames[-1]

    monkeypatch.setattr(server, "_sessions", {"s": parent})
    yield SimpleNamespace(server=server, usage=usage, identity=identity, agent=agent,
                          parent=parent, child=child, submissions=submissions,
                          run_child=run_child, delivered=delivered, host=host)
    host.close()


def test_refused_host_restores_exact_queue_envelope_without_history_or_retry(isolated_turn, monkeypatch):
    env = isolated_turn
    server, parent, child = env.server, env.parent, env.child
    transport = SimpleNamespace(write=lambda obj: True)
    queued = {"text": "kept request", "image_paths": ["kept.png"], "transport": transport,
              "turn_author": {"id": "bot:coder", "name": "coder", "is_bot": True}}
    later = {"text": "later", "transport": None}
    parent.update(queued_prompt=queued, queued_prompts=[later])
    parent_history, child_history = parent["history"], child["history"]
    assert server._drain_queued_prompt("r", "s", parent)
    frame, complete = env.submissions[0]
    assert parent["queued_prompt"] is later
    assert frame["turn_author"] == queued["turn_author"]
    # A different admitted turn spends the last tool after the queue head was claimed.
    env.usage.record_completed_tool(env.identity)
    end = env.run_child(frame)
    assert end["type"] == "turn.end"
    assert end.get("code") == env.usage.LIMIT_REASON
    assert end["continuation_required"] and end["retryable"]
    assert end["session_info"]["runtime"]["base_url"] == env.agent.base_url
    assert "api_key" not in end["session_info"]["runtime"]
    complete(end)
    complete(end)  # duplicate completion must not restore the envelope twice
    assert parent["queued_prompt"] is queued and parent["queued_prompts"] == [later]
    assert parent["history"] is parent_history and child["history"] is child_history
    assert parent["history_version"] == child["history_version"] == frame["history_version"]
    assert parent["inflight_turn"]["user"] == queued["text"]
    assert parent["inflight_turn"]["error_surface"]["code"] == env.usage.LIMIT_REASON
    assert not parent["running"] and parent["agent"] is None
    assert len(env.submissions) == 1
    assert server._free_tier_session_refusal(parent)["refusal_reason"] == env.usage.LIMIT_REASON
    assert server._drain_queued_prompt("again", "s", parent)
    assert parent["queued_prompt"] is queued and len(env.submissions) == 1
    # Even reattach/edit admission must use the host mirror before it touches history.
    monkeypatch.setattr(server, "_sess_nowait", lambda *a: (parent, None))
    result = server.handle_request({"id": "edit", "method": "prompt.submit", "params": {
        "session_id": "s", "text": "replacement", "truncate_before_user_ordinal": 0}})
    assert result["error"]["code"] == 4092
    assert parent["history"] is parent_history and parent["queued_prompt"] is queued
    # Cold sessions have no host snapshot: reuse route resolution, not AIAgent.
    cold = {**parent, "_metadata_mirror": {}, "_compute_host_active": False,
            "model_override": {"model": "same-label", "provider": "custom",
                               "base_url": "http://localhost:11434/v1"}}
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-local")
    assert server._free_tier_session_refusal(cold) is None
    server._make_agent.assert_not_called()
    env.agent._interruptible_api_call.assert_not_called()


def test_crossing_host_finishes_active_turn_then_parks_until_route_recovery(isolated_turn, monkeypatch):
    from agent import conversation_loop
    from agent.free_tier import record_tool_completion

    env = isolated_turn
    server, parent, child = env.server, env.parent, env.child
    parent.pop("_metadata_mirror")  # first isolated turn has no mirror yet
    parent["running"] = True
    server._start_inflight_turn(parent, "active")
    active = parent["inflight_turn"]
    queued = {"text": "later", "transport": None, "image_paths": ["later.png"]}
    parent["queued_prompt"] = queued
    parent_history = parent["history"]
    emitted = []
    emit = env.host.emit

    def host_emit(frame):
        emit(frame)
        emitted.append(frame)
        if frame.get("type") == "rpc" and frame["message"].get("method") == "compute_host.runtime":
            with monkeypatch.context() as m:
                m.setattr(server, "_sessions", {"s": parent})
                server._relay_compute_host_rpc(frame["message"])

    monkeypatch.setattr(env.host, "emit", host_emit)
    interrupted = Mock(side_effect=AssertionError("admitted turn cannot be interrupted by cap"))
    monkeypatch.setattr(server, "_interrupt_busy_session", interrupted)
    monkeypatch.setattr(server, "_after_complete_turn", lambda *a: None)

    def turn(agent, text, **kwargs):
        env.host._run_real_turn({**env.submissions[0][0], "request_id": "busy-host"})
        record_tool_completion(agent)
        assert env.usage.status()["capped"]
        assert parent["_metadata_mirror"]["runtime"]["base_url"] == agent.base_url
        assert parent["running"] and parent["inflight_turn"] is active
        with monkeypatch.context() as m:
            m.setattr(server, "_sessions", {"s": parent})
            m.setattr(server, "_sess_nowait", lambda *a: (parent, None))
            m.setattr(server, "_emit", lambda *a: emitted.append(a))
            m.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
            response = server.handle_request({"id": "busy", "method": "prompt.submit", "params": {
                "session_id": "s", "text": "after cap"}})
            assert response["result"]["status"] == "queued"
            assert parent["running"] and parent["inflight_turn"] is active
            assert not any(isinstance(e, tuple) for e in emitted)
        record_tool_completion(agent)  # admitted work still finishes past the threshold
        return {"completed": True, "final_response": "Finished.", "messages": [
            *kwargs["conversation_history"], {"role": "user", "content": text},
            {"role": "assistant", "content": "Finished."}]}

    monkeypatch.setattr(conversation_loop, "_run_conversation_turn", turn)
    assert server._submit_prompt_to_compute_host("r", "s", parent, "active")["result"]["status"] == "streaming"
    frame, complete = env.submissions[0]
    end = env.run_child(frame)
    assert emitted[0]["type"] == "turn.started"
    assert end["continuation_required"] and "code" not in end
    completions = [e["params"]["payload"] for e in env.delivered
                   if e.get("method") == "event" and e["params"]["type"] == "message.complete"]
    assert completions[-1]["text"].endswith(env.usage.LIMIT_NOTICE)
    complete(end)
    assert not parent["running"] and parent["inflight_turn"] is None
    assert parent["queued_prompt"] is queued and len(env.submissions) == 1
    assert child["history"][-1]["content"] == "Finished."
    assert parent["history"] is parent_history  # host is the only authoritative history writer
    assert parent["history_version"] == child["history_version"]
    assert env.agent._cached_system_prompt == "unchanged-prefix"
    assert server._free_tier_session_refusal(parent) is not None
    assert not any(e.get("method") == "compute_host.runtime" for e in env.delivered)
    # The same model label at a local endpoint must release the parked queue.
    env.agent.base_url = "http://localhost:11434/v1"
    env.agent._primary_runtime = None
    ack = env.host._control_ack(server, {"sid": "s", "route_name": "session.history.reload"}, child)
    server._apply_compute_host_metadata_mirror(parent, ack)
    assert server._free_tier_session_refusal(parent) is None
    assert server._drain_queued_prompt("recovered", "s", parent)
    assert len(env.submissions) == 2
    assert env.submissions[-1][0]["text"] == queued["text"]
    assert env.submissions[-1][0]["attached_images"] == queued["image_paths"]
    # Completed shared sign-in also releases a stale welcome mirror without
    # changing this parent's agent/history or spending another tool.
    from hermes_cli import auth_nous
    parent["_metadata_mirror"]["runtime"]["base_url"] = "https://welcome-api.nousresearch.com/v1"
    auth_nous._write_shared_nous_state({"auth_method": "oauth", "access_token": "signed-in-fixture",
                                       "refresh_token": "refresh-fixture"})
    assert server._free_tier_session_refusal(parent) is None
    assert parent["agent"] is None and parent["history"] is parent_history
    interrupted.assert_not_called()
    server._make_agent.assert_not_called()
