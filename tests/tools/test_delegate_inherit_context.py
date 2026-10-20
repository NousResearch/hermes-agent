"""delegate_task tasks[].inherit_context: fold the parent conversation into the child's start.

A verbatim copy of assistant/tool turns tends to be disavowed by the child (it reads them as its
own actions it does not remember), so the history is folded into ONE user-role context message.
"""
import threading
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _fold_conversation_history_to_context,
    delegate_task,
)


def _msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def _parent(history=None):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.prefill_messages = [{"role": "user", "content": "boot seed"}]
    parent._session_messages = history or []
    return parent


class TestFold:
    def test_nothing_to_inherit(self):
        assert _fold_conversation_history_to_context([], 1000) is None
        assert _fold_conversation_history_to_context(None, 1000) is None
        assert _fold_conversation_history_to_context([_msg("system", "sys")], 1000) is None

    def test_single_labeled_user_message(self):
        out = _fold_conversation_history_to_context([_msg("user", "Deploy it"), _msg("assistant", "On it")], 1000)
        assert out["role"] == "user" and isinstance(out["content"], str)
        assert "INHERITED CONTEXT FROM PARENT SESSION" in out["content"]
        assert out["content"].index("Deploy it") < out["content"].index("On it")

    def test_openai_tool_calls_and_tool_results_become_prose(self):
        history = [
            _msg("user", "read the manifest"),
            _msg("assistant", None, tool_calls=[{"id": "c1", "type": "function",
                                                 "function": {"name": "terminal", "arguments": '{"command": "cat x"}'}}]),
            _msg("tool", "region=ap-southeast-2", tool_call_id="c1"),
        ]
        content = _fold_conversation_history_to_context(history, 2000)["content"]
        assert "[ran terminal:" in content and "cat x" in content
        assert "[result: region=ap-southeast-2]" in content

    def test_content_part_lists_and_anthropic_blocks(self):
        history = [
            _msg("user", [{"type": "text", "text": "plain part"}, {"type": "image_url", "image_url": {"url": "x"}}]),
            _msg("assistant", [{"type": "tool_use", "name": "terminal", "input": {"command": "ls"}}]),
            _msg("user", [{"type": "tool_result", "content": [{"type": "text", "text": "a.txt"}]}]),
        ]
        content = _fold_conversation_history_to_context(history, 2000)["content"]
        assert "plain part" in content and "[ran terminal:" in content and "a.txt" in content

    def test_keeps_newest_turns_under_budget(self):
        history = [_msg("user", "OLDEST_MARKER " + "x " * 400)]
        history += [_msg("assistant", "y " * 400) for _ in range(3)]
        history += [_msg("user", "NEWEST_MARKER recent fact")]
        content = _fold_conversation_history_to_context(history, 200)["content"]
        assert "NEWEST_MARKER" in content and "OLDEST_MARKER" not in content


class TestSchema:
    def test_per_task_field_is_advertised(self):
        item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert item["inherit_context"]["type"] == "boolean"


class TestChildConstruction:
    def _build(self, parent, **kw):
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(task_index=0, goal="Continue the work", context=None, toolsets=None, model=None,
                               max_iterations=10, parent_agent=parent, task_count=1, **kw)
        return MockAgent.call_args.kwargs

    def test_default_child_gets_only_the_boot_prefill(self):
        parent = _parent([_msg("user", "SECRET_SHOULD_NOT_LEAK")])
        kwargs = self._build(parent)
        assert kwargs["prefill_messages"] == [{"role": "user", "content": "boot seed"}]

    def test_inherit_context_child_gets_the_folded_history(self):
        parent = _parent([_msg("user", "ESTABLISHED_FACT the region is ap-southeast-2")])
        prefill = self._build(parent, inherit_context=True)["prefill_messages"]
        assert len(prefill) == 1 and prefill[0]["role"] == "user"
        assert "ESTABLISHED_FACT" in prefill[0]["content"]

    def test_inherit_with_empty_history_keeps_the_boot_prefill(self):
        assert self._build(_parent([]), inherit_context=True)["prefill_messages"] == [
            {"role": "user", "content": "boot seed"}]


def test_delegate_task_threads_the_per_task_flag_to_child_construction():
    def _flag_for(task):
        seen = []

        def _capture(**kwargs):
            seen.append(kwargs.get("inherit_context"))
            raise ValueError("stop after capture")

        with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_capture):
            delegate_task(tasks=[task], parent_agent=_parent(), background=False)
        return seen

    assert _flag_for({"goal": "a task with enough detail", "inherit_context": True}) == [True]
    assert _flag_for({"goal": "a task with enough detail", "inherit_context": "true"}) == [True]
    assert _flag_for({"goal": "a task with enough detail"}) == [False]
