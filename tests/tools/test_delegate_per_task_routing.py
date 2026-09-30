"""Per-task ``model`` / ``provider`` / ``reasoning_effort`` on ``delegate_task`` ``tasks[]`` entries.

Routing travels with the task: one batch can mix a cheap model for mechanical work with the parent's model for
hard work, without mutating the shared ``delegation.*`` config.
"""
import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task


def _parent():
    parent = MagicMock()
    parent.base_url = "https://api.anthropic.com"
    parent.api_key = "parent-key"
    parent.provider = "anthropic"
    parent.api_mode = "anthropic_messages"
    parent.model = "parent-model"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.reasoning_config = {"enabled": True, "effort": "xhigh"}
    parent.request_overrides = {}
    return parent


def _spawn(tasks, cfg=None, resolved=None):
    """Run delegate_task synchronously; return (result, [AIAgent kwargs per child])."""
    cfg = {"max_iterations": 5, **(cfg or {})}
    with patch("tools.delegate_tool._load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=resolved or {}) as resolve, \
         patch("run_agent.AIAgent") as agent_cls:
        child = MagicMock()
        child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1}
        agent_cls.return_value = child
        out = json.loads(delegate_task(tasks=tasks, parent_agent=_parent()))
        return out, [c.kwargs for c in agent_cls.call_args_list], resolve


class TestPerTaskRouting(unittest.TestCase):
    def test_task_routing_applies_to_that_child_and_the_rest_inherit_the_parent(self):
        resolved = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "or-key", "api_mode": "chat_completions"}
        out, calls, resolve = _spawn(
            [
                {"goal": "Reformat the commit list into release notes", "model": "small-model",
                 "reasoning_effort": "low"},
                {"goal": "Sweep the docs folder for stale TODO markers", "provider": "openrouter", "model": "x/y"},
                {"goal": "Find the race condition in the retry worker"},
            ],
            resolved=resolved,
        )
        self.assertNotIn("error", out)
        resolve.assert_called_once_with(requested="openrouter", target_model="x/y")
        route = lambda c: (c["model"], c["provider"], c["base_url"], c["api_key"], c["reasoning_config"])
        parent = _parent()
        xhigh = {"enabled": True, "effort": "xhigh"}
        self.assertEqual(route(calls[0]), ("small-model", parent.provider, parent.base_url, parent.api_key,
                                           {"enabled": True, "effort": "low"}))
        self.assertEqual(route(calls[1]), ("x/y", "openrouter", "https://openrouter.ai/api/v1", "or-key", xhigh))
        self.assertEqual(route(calls[2]), (parent.model, parent.provider, parent.base_url, parent.api_key, xhigh))

    def test_invalid_task_routing_refuses_the_whole_batch_before_spawning(self):
        parent_on_anthropic = {"goal": "Find the race condition in the retry worker"}
        for bad in (
            {"reasoning_effort": "hgih"},
            {"model": ""},
            {"provider": 5},
            {"provider": "openrouter"},                         # provider without a model
            {"provider": "anthropic", "model": "gpt-5.6-sol"},  # model from another vendor's catalog
            {"model": "gpt-5.6-sol"},                           # checked against the parent's provider
        ):
            out, calls, resolve = _spawn([parent_on_anthropic, {"goal": "Summarise the diff for the reviewer", **bad}])
            self.assertIn("error", out, bad)
            self.assertEqual(calls, [], bad)
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
