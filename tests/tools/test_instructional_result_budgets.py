"""Instruction loads survive result spilling even when the turn exceeds its budget."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tools.budget_config import budget_for_context_window
from tools.tool_result_storage import enforce_turn_budget, maybe_persist_tool_result


@pytest.mark.parametrize("tool_name", ["skill_view", "read_file"])
@pytest.mark.parametrize("storage", ["missing", "fails", "works"])
@pytest.mark.parametrize("context_length", [65_536, 256_000])
def test_instructional_results_survive_both_budgets(tool_name, storage, context_length):
    content = "intro\n" + "instruction\n" * 20_000 + "final instruction"
    env = None if storage == "missing" else MagicMock()
    if env is not None:
        env.execute.return_value = {"returncode": int(storage == "fails")}
    config = replace(budget_for_context_window(context_length), tool_overrides={tool_name: 1})
    output = maybe_persist_tool_result(content, tool_name, "instructions", env, config)
    messages = [{"name": tool_name, "tool_call_id": "instructions", "content": output}]
    enforce_turn_budget(messages, env, config)
    assert messages[0]["content"] == content
    if env is not None:
        env.execute.assert_not_called()


@pytest.mark.parametrize("identity_key", ["name", "tool_name"])
def test_turn_budget_spills_only_eligible_results(identity_key):
    content = "read every instruction\n" * 10_000
    messages = [
        {identity_key: name, "tool_call_id": name, "content": content}
        for name in ["skill_view", "read_file", "custom_instructions", "terminal", ""]
    ]
    # Existing registry escape hatches must be honored by aggregate enforcement too.
    def threshold(name, default):
        return float("inf") if name == "custom_instructions" else default

    with patch("tools.registry.registry.get_max_result_size", side_effect=threshold):
        enforce_turn_budget(messages, config=budget_for_context_window(65_536))
    assert all(message["content"] == content for message in messages[:3])
    assert all(message["content"] != content for message in messages[3:])
