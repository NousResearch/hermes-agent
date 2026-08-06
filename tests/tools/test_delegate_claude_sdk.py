"""Delegation contract for the Claude Agent SDK runtime.

Three properties, none of which hold by accident:

1. A child inherits the parent's provider mode **without an API key** —
   ``claude-code`` has none by design, and the guards that demand one had to
   learn the difference between "missing" and "none exists".
2. Every child gets its **own** ``ClaudeAgentSession``. ``ClaudeSDKClient`` is
   bound to the event-loop thread that created it, so a shared session across
   parent and children (or across siblings) corrupts the transport.
3. Those sessions are torn down when the child finishes. N children in parallel
   must leave N *fewer* live loop threads than they started, i.e. zero.

No real SDK is involved: the session facade takes a ``client_factory``, which is
the seam these tests use.
"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch

from agent.transports.claude_agent_session import ClaudeAgentSession
from run_agent import AIAgent
from tools.delegate_tool import (
    _build_child_agent,
    _provider_owns_its_own_credentials,
    _resolve_delegation_credentials,
    delegate_task,
)

SESSION_THREAD_NAME = "hermes-claude-agent-sdk"


class _FakeSDKClient:
    """Minimal stand-in for ``ClaudeSDKClient`` (connect/disconnect only)."""

    def __init__(self, options=None):
        self.options = options
        self.connected = False
        self.disconnected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True


def _live_session_threads() -> list[threading.Thread]:
    return [
        t for t in threading.enumerate() if t.name == SESSION_THREAD_NAME and t.is_alive()
    ]


def _make_claude_parent(depth: int = 0):
    parent = MagicMock()
    parent.base_url = "claude-sdk://subscription"
    parent.api_key = ""
    parent.provider = "claude-code"
    parent.api_mode = "claude_agent_sdk"
    parent.model = "claude-sonnet-5"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.client = None
    parent._client_kwargs = {}
    return parent


class TestChildInheritsSubscriptionMode(unittest.TestCase):
    def test_child_gets_the_runtime_without_an_api_key(self):
        parent = _make_claude_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="inherit the subscription runtime",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )
            _, kwargs = MockAgent.call_args

        self.assertEqual(kwargs["provider"], "claude-code")
        self.assertEqual(kwargs["api_mode"], "claude_agent_sdk")
        self.assertEqual(kwargs["base_url"], "claude-sdk://subscription")
        # The point of the provider: there is no credential to inherit.
        self.assertFalse(kwargs["api_key"])

    def test_a_credential_less_runtime_is_not_a_missing_credential(self):
        self.assertTrue(
            _provider_owns_its_own_credentials({"api_mode": "claude_agent_sdk"})
        )
        self.assertTrue(
            _provider_owns_its_own_credentials(
                {"credentials_owner": "claude-agent-sdk"}
            )
        )
        # Every other provider keeps failing loudly on an empty key.
        self.assertFalse(_provider_owns_its_own_credentials({"api_mode": "chat_completions"}))
        self.assertFalse(_provider_owns_its_own_credentials({}))

    def test_delegation_provider_claude_code_does_not_require_a_key(self):
        runtime = {
            "provider": "claude-code",
            "api_mode": "claude_agent_sdk",
            "base_url": "claude-sdk://subscription",
            "api_key": "",
            "credentials_owner": "claude-agent-sdk",
        }
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime
        ):
            creds = _resolve_delegation_credentials(
                {"provider": "claude-code", "model": "claude-sonnet-5"}, None
            )
        self.assertEqual(creds["provider"], "claude-code")
        self.assertEqual(creds["api_mode"], "claude_agent_sdk")
        self.assertEqual(creds["api_key"], "")

    def test_a_genuinely_missing_key_still_raises(self):
        runtime = {
            "provider": "minimax",
            "api_mode": "chat_completions",
            "base_url": "https://api.minimax.io/v1",
            "api_key": "",
        }
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime
        ):
            with self.assertRaises(ValueError):
                _resolve_delegation_credentials({"provider": "minimax"}, None)


class TestChildSessionIsolation(unittest.TestCase):
    """Each child owns one session; none of them survive the delegation."""

    def _run_parallel_children(self, count: int):
        parent = _make_claude_parent()
        created: list[ClaudeAgentSession] = []
        created_lock = threading.Lock()

        def _make_child(**_kwargs):
            child = MagicMock()
            child._claude_session = None
            child.provider = "claude-code"
            child.api_mode = "claude_agent_sdk"
            child.model = "claude-sonnet-5"
            child.session_prompt_tokens = 0
            child.session_completion_tokens = 0
            child.session_estimated_cost_usd = 0.0
            child._delegate_role = "leaf"

            def _run_conversation(user_message, task_id=None, stream_callback=None):
                # Exactly what agent.claude_runtime does on a child's first
                # turn: build this agent's own session and connect it.
                session = ClaudeAgentSession(
                    options_factory=lambda: None,
                    client_factory=_FakeSDKClient,
                )
                session.ensure_started()
                child._claude_session = session
                with created_lock:
                    created.append(session)
                return {
                    "final_response": "done",
                    "completed": True,
                    "api_calls": 1,
                    "messages": [],
                }

            child.run_conversation.side_effect = _run_conversation
            # The real teardown path, run against a duck-typed child.
            child.close.side_effect = lambda: AIAgent._release_claude_agent_sdk_session(
                child
            )
            return child

        before = len(_live_session_threads())
        with patch("run_agent.AIAgent", side_effect=_make_child):
            delegate_task(
                tasks=[{"goal": f"task {i}"} for i in range(count)],
                parent_agent=parent,
            )
        return created, before

    def test_each_child_gets_its_own_session(self):
        # 3 is delegation.max_concurrent_children's default, which is also the
        # hard cap on a single batch — the widest fan-out one call can produce.
        created, _ = self._run_parallel_children(3)
        self.assertEqual(len(created), 3)
        # Identity, not equality: a shared ClaudeSDKClient across sibling
        # threads corrupts the anyio streams it was created with.
        self.assertEqual(len({id(session) for session in created}), 3)

    def test_no_loop_thread_survives_the_delegation(self):
        created, before = self._run_parallel_children(3)
        for session in created:
            self.assertTrue(session.closed)
        # join() inside close() is synchronous, so this is not a race.
        self.assertEqual(len(_live_session_threads()), before)

    def test_children_do_not_share_the_parents_session(self):
        parent = _make_claude_parent()
        parent_session = ClaudeAgentSession(
            options_factory=lambda: None, client_factory=_FakeSDKClient
        )
        parent_session.ensure_started()
        parent._claude_session = parent_session
        try:
            created, _ = self._run_parallel_children(2)
            for session in created:
                self.assertIsNot(session, parent_session)
            # A child's teardown must not close the parent out from under it.
            self.assertFalse(parent_session.closed)
        finally:
            parent_session.close()


if __name__ == "__main__":
    unittest.main()
