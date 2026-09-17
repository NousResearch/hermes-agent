import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from workstation.task_compiler import execute_compiled_work
from agent.context_compressor import _build_operational_reference_envelope


def make_agent():
    definitions = [{"type": "function", "function": {"name": name,
                    "parameters": {"type": "object", "properties": {}}}}
                   for name in ["work_execute", "read_file", "write_file"]]
    with patch("run_agent.get_tool_definitions", return_value=definitions), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.config.load_config_readonly", return_value={}), \
         patch("run_agent.OpenAI"):
        return AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                       quiet_mode=True, skip_memory=True, skip_context_files=True)


def test_real_agent_dispatch_routes_100_items_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent()
    from hermes_state import SessionDB
    agent._session_db = SessionDB(tmp_path / "state.db")
    agent.client = MagicMock()
    req = {"operation_key": "100-reads", "items": [{"path": str(i)} for i in range(100)],
           "steps": [{"tool": "read_file", "args": {"path": "$item.path"}}]}
    calls = []
    def handler(name, args, task_id, **kwargs):
        if name == "work_execute":
            return execute_compiled_work(args, task_id=task_id)
        calls.append(args["path"])
        return json.dumps({"ok": True, "path": args["path"], "text": "row" * 10000})
    call = SimpleNamespace(id="compiled", type="function", function=SimpleNamespace(
        name="work_execute", arguments=json.dumps(req)))
    messages = []
    with patch("run_agent.handle_function_call", side_effect=handler):
        agent._execute_tool_calls(SimpleNamespace(content="", tool_calls=[call]), messages, "task")
    assert calls == [str(i) for i in range(100)]
    assert len(messages) == 1
    result = json.loads(messages[0]["content"])
    assert result["completed"] == 100
    assert not agent.client.chat.completions.create.called
    envelope = _build_operational_reference_envelope(messages)
    assert result["plan_id"] in envelope
    assert result["task_id"] in envelope
    rows = agent._session_db.get_messages(agent.session_id)
    assert not any(str(row.get("tool_call_id", "")).startswith("plan_") for row in rows)


def test_provider_usage_message_builder_unknown_and_known(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent()
    from hermes_state import SessionDB
    agent._session_db = SessionDB(tmp_path / "state.db")
    response = SimpleNamespace(content="done", tool_calls=None)
    agent._current_provider_usage = None
    unknown = agent._build_assistant_message(response, "stop")
    assert unknown.get("_hermes_token_count") is None
    usage = {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 50,
             "reasoning_tokens": 5, "total_tokens": 120}
    agent._current_provider_usage = usage
    known = agent._build_assistant_message(response, "stop")
    assert known["_hermes_token_count"] == 120
    assert known["display_metadata"]["provider_usage"] == usage
    agent._ensure_db_session()
    agent._flush_messages_to_session_db([known])
    rows = agent._session_db.get_messages(agent.session_id)
    assert rows[-1]["token_count"] == 120


def test_quantified_request_cannot_enter_item_mutation_loop(tmp_path, monkeypatch):
    from workstation.task_compiler import batch_intent
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent()
    agent._work_batch_candidate = batch_intent("Crie 100 registros para estes itens")
    call = SimpleNamespace(id="item-write", type="function", function=SimpleNamespace(
        name="write_file", arguments='{"path":"one.json","content":"one"}'))
    messages = []
    with patch("run_agent.handle_function_call") as dispatch:
        agent._execute_tool_calls(SimpleNamespace(tool_calls=[call]), messages, "task")
        assert not dispatch.called
        assert json.loads(messages[0]["content"])["code"] == "durable_compile_required"
        agent._execute_tool_calls(SimpleNamespace(tool_calls=[call]), messages, "task")
        assert agent._tool_guardrail_halt_decision is None
        assert json.loads(messages[-1]["content"])["preflight_status"] == "PREFLIGHT_REQUIRED"
    assert not batch_intent("Compare 100 arquivos e discuta as diferenças")


def test_full_conversation_two_provider_boundaries_for_100_items(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent()
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    req = {"operation_key": "conversation-batch", "items": [{"path": str(i)} for i in range(100)],
           "steps": [{"tool": "read_file", "args": {"path": "$item.path"}}]}
    call = SimpleNamespace(id="compiled", type="function", function=SimpleNamespace(
        name="work_execute", arguments=json.dumps(req)))
    def response(content, tools, finish):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=tools), finish_reason=finish)], model="test/model", usage=None)
    agent.client.chat.completions.create.side_effect = [response("", [call], "tool_calls"),
                                                      response("100 completed", None, "stop")]
    calls = []
    def handler(name, args, task, **kw):
        if name == "work_execute":
            return execute_compiled_work(args, task_id=task)
        calls.append(args["path"])
        return json.dumps({"ok": True, "path": args["path"]})
    with patch("run_agent.handle_function_call", side_effect=handler):
        result = agent.run_conversation("Crie 100 registros para estes itens", task_id="task")
    assert result["final_response"] == "100 completed"
    assert len(calls) == 100
    assert agent.client.chat.completions.create.call_count == 2
