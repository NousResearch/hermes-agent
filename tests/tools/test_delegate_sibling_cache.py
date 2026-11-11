"""Sibling subagents send one shared prompt prefix and one cache scope.

Providers that cache automatically match only inside the scope the client sends, and only up to
the first byte that differs. A child's task context used to sit in its system prompt (ahead of the
tools on OpenAI-family models) under a per-child scope, so siblings shared nothing.
"""

import os
from pathlib import Path


def test_siblings_send_one_system_prompt_and_get_their_context_in_the_user_turn():
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
    from tools.delegate_tool import delegate_task

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer([]) as srv:
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        srv.push(Text("first done"), Text("second done"))
        parent = AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                         session_db=SessionDB(db_path=home / "state.db"), session_id="parent", quiet_mode=True, platform="cli")
        parent._ensure_db_session()
        delegate_task(goal="count the files", context="The parent is auditing numbers.", parent_agent=parent)
        delegate_task(goal="find the flaky test", context="The parent is fixing the gateway.", parent_agent=parent)
        parent.close()
        first, second = srv.main_requests()

    assert first["messages"][0] == second["messages"][0]
    assert first["tools"] == second["tools"]
    assert "auditing numbers" not in str(first["messages"][0])
    task = first["messages"][1]
    assert task["role"] == "user" and task["content"].startswith("count the files")
    assert "The parent is auditing numbers." in task["content"]
    assert "The parent is fixing the gateway." in second["messages"][1]["content"]


def test_schema_correction_turn_carries_the_contract_even_after_a_compaction():
    """The contract lives in the first user turn. The one correction turn continues that conversation
    and pastes the contract again, because a long first turn is compacted before the retry is sent."""
    import json

    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home
    from tools.delegate_tool import delegate_task

    home = Path(os.environ["HERMES_HOME"])
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    (home / "big.txt").write_text("filler line for the transcript\n" * 400)
    # The provider's reported prompt size drives compaction, so the fake reports a real one.
    with FakeLLMServer([], prompt_tokens_fn=lambda body: len(json.dumps(body)) // 4) as srv:
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW
                          + "compression:\n  threshold_tokens: 12000\n")
        srv.push(*[ToolCall("read_file", {"path": str(home / "big.txt"), "offset": 1 + i}) for i in range(10)],
                 Text("there are three"), Text('{"count": 3}'))
        parent = AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                         session_db=SessionDB(db_path=home / "state.db"), session_id="parent", quiet_mode=True, platform="cli")
        parent._ensure_db_session()
        entry = json.loads(delegate_task(goal="count the files", context="Only count .py files.", output_schema=schema,
                                         parent_agent=parent))["results"][0]
        parent.close()
        requests = srv.main_requests()

    correction = requests[-1]
    assert entry["schema_valid"] is True
    assert len(correction["messages"]) < len(requests[-2]["messages"]), "the retry was expected to compact"
    assert "OUTPUT CONTRACT" in correction["messages"][-1]["content"] and '"count"' in correction["messages"][-1]["content"]
    assert len(entry["tool_trace"]) == 10


def test_codex_siblings_share_the_session_header_the_backend_caches_under():
    from agent.chat_completion_helpers import _build_api_kwargs_for_mode
    from run_agent import AIAgent
    from tools.delegate_tool import _build_child_agent

    route = {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "fake",
             "api_mode": "codex_responses", "model": "gpt-5.4"}
    parent = AIAgent(**route, session_id="parent", quiet_mode=True, platform="cli", skip_memory=True)

    def request(agent):
        return _build_api_kwargs_for_mode(agent, [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}])

    children = [
        _build_child_agent(task_index=i, goal="g", context=f"context {i}", toolsets=None, model=None,
                           max_iterations=5, task_count=2, parent_agent=parent)
        for i in range(2)
    ]
    one, two = (request(child) for child in children)
    own = request(parent)
    # A child that lands on a slot-keyed route (xAI) after it was built goes back to its own scope.
    from agent.prompt_cache_scope import resolve_prompt_cache_scope
    children[0].provider, children[0].base_url = "xai", "https://api.x.ai/v1"
    isolated = resolve_prompt_cache_scope(children[0])
    for agent in (*children, parent):
        agent.close()

    assert children[0].session_id != children[1].session_id
    assert one["extra_headers"]["session_id"] == two["extra_headers"]["session_id"]
    assert one["prompt_cache_key"] == two["prompt_cache_key"]
    # The delegation tree gets its own scope; the parent keeps its physical session id.
    assert own["extra_headers"]["session_id"] == "parent" != one["extra_headers"]["session_id"]
    assert own["prompt_cache_key"] != one["prompt_cache_key"]
    assert isolated == children[0].session_id
