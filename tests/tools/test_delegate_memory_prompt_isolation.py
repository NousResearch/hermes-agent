"""#51797: a delegated child's fresh prompt must inherit its parent's isolation gate."""
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _build_child_agent


@pytest.mark.parametrize("isolated", [False, True])
def test_delegated_child_inherits_memory_prompt_gate(isolated):
    parent = _make_mock_parent()
    parent.skip_memory_prompt = isolated
    with patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value = MagicMock()
        child = _build_child_agent(task_index=0, goal="test isolation", context=None,
                                   toolsets=["file"], model="test-model", max_iterations=1,
                                   parent_agent=parent, task_count=1)
        assert agent_cls.call_args.kwargs["skip_memory_prompt"] is isolated
        assert agent_cls.call_args.kwargs["skip_context_files"] is True
        assert child is not None
