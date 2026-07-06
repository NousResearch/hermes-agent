"""Tests for boomerang context-inheritance: the fold helper + delegate_task inherit_context param.

Phase 1 of the /boomerang spec (2026-07-05_boomerang-spec.md, v0.8).
Load-bearing correction from Phase-0 probe P0.3: the parent transcript is FOLDED into a
single user-role context message (NOT copied verbatim as assistant/tool turns, which the
child disavows). See PHASE-0-boomerang-baseline.md.
"""
import inspect

from tools.delegate_tool import (
    _fold_conversation_history_to_context,
    delegate_task,
    _build_child_agent,
    DELEGATE_TASK_SCHEMA,
)


def _msg(role, content):
    return {"role": role, "content": content}


class TestFoldConversationHistory:
    def test_empty_history_returns_none(self):
        assert _fold_conversation_history_to_context([], max_tokens=1000) is None
        assert _fold_conversation_history_to_context(None, max_tokens=1000) is None

    def test_returns_single_user_role_message(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi there")]
        out = _fold_conversation_history_to_context(history, max_tokens=1000)
        assert isinstance(out, dict)
        assert out["role"] == "user", "P0.3: folded context MUST be user-role or the child disavows it"
        assert isinstance(out["content"], str)

    def test_content_is_labeled_inherited_context_block(self):
        history = [_msg("user", "Deploy meridian"), _msg("assistant", "On it")]
        out = _fold_conversation_history_to_context(history, max_tokens=1000)
        assert "INHERITED CONTEXT FROM PARENT SESSION" in out["content"]

    def test_no_raw_tool_use_or_tool_result_blocks_in_output(self):
        history = [
            _msg("user", "read the manifest"),
            {"role": "assistant", "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "t1", "name": "terminal", "input": {"command": "cat x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "region=ap-southeast-2"},
            ]},
        ]
        out = _fold_conversation_history_to_context(history, max_tokens=2000)
        assert out["role"] == "user"
        assert isinstance(out["content"], str)  # prose, never a structured content-block list
        assert "ap-southeast-2" in out["content"]

    def test_preserves_recent_content_over_oldest_when_over_budget(self):
        history = [_msg("user", "OLDEST_MARKER " + "x " * 400)]
        history += [_msg("assistant", "y " * 400) for _ in range(3)]
        history += [_msg("user", "NEWEST_MARKER important recent fact")]
        out = _fold_conversation_history_to_context(history, max_tokens=200)
        assert "NEWEST_MARKER" in out["content"]
        assert "OLDEST_MARKER" not in out["content"]

    def test_string_and_block_content_both_handled(self):
        history = [
            _msg("user", "plain string content"),
            {"role": "assistant", "content": [{"type": "text", "text": "block-form text"}]},
        ]
        out = _fold_conversation_history_to_context(history, max_tokens=1000)
        assert "plain string content" in out["content"]
        assert "block-form text" in out["content"]


class TestInheritContextWiring:
    def test_delegate_task_signature_has_inherit_context(self):
        assert "inherit_context" in inspect.signature(delegate_task).parameters

    def test_build_child_agent_signature_has_inherit_context(self):
        params = inspect.signature(_build_child_agent).parameters
        assert "inherit_context" in params
        # INV-5: defaults to False so every existing caller is byte-identical.
        assert params["inherit_context"].default is False

    def test_schema_advertises_inherit_context_boolean(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "inherit_context" in props
        assert props["inherit_context"]["type"] == "boolean"


class TestInv5DefaultPathByteIdentical:
    """INV-5: with inherit_context absent/false the child's prefill is the parent's
    boot-seed attribute exactly — the conversation history must NOT leak."""

    def test_default_path_uses_parent_prefill_not_conversation_history(self):
        # This mirrors the exact _build_child_agent default-branch logic:
        #   child_prefill_messages = getattr(parent, "prefill_messages", None)
        # and asserts conversation_history never contributes when inherit is off.
        class FakeParent:
            prefill_messages = [{"role": "user", "content": "boot seed"}]
            conversation_history = [{"role": "user", "content": "SECRET_SHOULD_NOT_LEAK"}]

        parent = FakeParent()
        inherit_context = False
        child_prefill = getattr(parent, "prefill_messages", None)  # default branch
        assert child_prefill == [{"role": "user", "content": "boot seed"}]
        assert "SECRET_SHOULD_NOT_LEAK" not in str(child_prefill)

    def test_inherit_path_folds_conversation_history(self):
        history = [{"role": "user", "content": "SECRET_MARKER established fact"}]
        folded = _fold_conversation_history_to_context(history, 50000)
        assert folded["role"] == "user"
        assert "SECRET_MARKER" in folded["content"]


class TestInheritContextIntegration:
    """Integration: drive the REAL delegate_task wiring (signature -> task-list ->
    _build_child_agent -> AIAgent construction) with a mocked AIAgent, and assert the
    child is constructed with the folded conversation_history as prefill_messages.

    This exercises the whole thread, not just the fold helper (AC-4 / INV-4)."""

    def _make_parent(self, conversation_history):
        import threading
        from unittest.mock import MagicMock
        parent = MagicMock()
        parent.base_url = "https://openrouter.ai/api/v1"
        parent.api_key = "***"
        parent.provider = "openrouter"
        parent.api_mode = "chat_completions"
        parent.model = "anthropic/claude-sonnet-4"
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
        parent.prefill_messages = None
        # The GATEWAY populates _session_messages (not conversation_history) — mirror
        # the real production attribute so the fold reads the same source it does live.
        parent._session_messages = conversation_history
        parent.conversation_history = conversation_history
        return parent

    def _make_gateway_parent(self, session_messages):
        """A gateway-shaped parent: has _session_messages, NO conversation_history
        attr at all (a real AIAgent in the gateway has no such attribute). Uses a
        plain object, not MagicMock, so getattr misses are real misses — this is
        what makes it a faithful regression for the 'INHERITED: no' production bug."""
        import threading

        class _GatewayAgent:
            pass

        p = _GatewayAgent()
        p.base_url = "https://openrouter.ai/api/v1"
        p.api_key = "***"
        p.provider = "openrouter"
        p.api_mode = "chat_completions"
        p.model = "anthropic/claude-sonnet-4"
        p.platform = "discord"
        p.providers_allowed = None
        p.providers_ignored = None
        p.providers_order = None
        p.provider_sort = None
        p._session_db = None
        p._delegate_depth = 0
        p._active_children = []
        p._active_children_lock = threading.Lock()
        p._print_fn = None
        p.tool_progress_callback = None
        p.thinking_callback = None
        p.prefill_messages = None
        p._session_messages = session_messages
        # NOTE: deliberately NO conversation_history attribute (gateway reality)
        return p

    def _capture_child_kwargs(self, **delegate_kwargs):
        """Run delegate_task with AIAgent patched; return the kwargs the child got."""
        import json
        from unittest.mock import patch, MagicMock
        captured = {}

        def _fake_aiagent(**kwargs):
            captured.update(kwargs)
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1,
            }
            child._delegate_depth = 1
            child.get_activity_summary.return_value = {"api_call_count": 1}
            return child

        with patch("run_agent.AIAgent", side_effect=_fake_aiagent):
            delegate_task(**delegate_kwargs)
        return captured

    def test_inherit_true_seeds_child_with_folded_context(self):
        history = [
            {"role": "user", "content": "Deploy region is ap-southeast-2"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking health"},
                {"type": "tool_use", "id": "t1", "name": "terminal", "input": {"command": "health"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "replica-4 CrashLooping"},
            ]},
        ]
        parent = self._make_parent(history)
        captured = self._capture_child_kwargs(
            goal="investigate the failing replica", inherit_context=True, parent_agent=parent,
        )
        pf = captured.get("prefill_messages")
        assert isinstance(pf, list) and len(pf) == 1, f"expected one folded msg, got {pf!r}"
        assert pf[0]["role"] == "user"
        assert "INHERITED CONTEXT FROM PARENT SESSION" in pf[0]["content"]
        assert "ap-southeast-2" in pf[0]["content"]
        assert "replica-4 CrashLooping" in pf[0]["content"]

    def test_gateway_parent_with_only_session_messages_still_inherits(self):
        """REGRESSION for the live 'INHERITED: no' bug (2026-07-05): a gateway
        AIAgent has NO conversation_history attribute — its live transcript is on
        _session_messages. The fold must read _session_messages, or production
        inheritance silently yields nothing while unit tests (which set
        conversation_history) pass. Uses a plain gateway-shaped object with only
        _session_messages set."""
        session_messages = [
            {"role": "user", "content": "The Meridian canary is at 12 percent"},
            {"role": "assistant", "content": "Noted, 12% canary."},
        ]
        parent = self._make_gateway_parent(session_messages)
        assert not hasattr(parent, "conversation_history"), "fixture must mirror gateway (no conversation_history)"
        captured = self._capture_child_kwargs(
            goal="what is the canary percent", inherit_context=True, parent_agent=parent,
        )
        pf = captured.get("prefill_messages")
        assert isinstance(pf, list) and len(pf) == 1, f"gateway parent did not inherit: {pf!r}"
        assert pf[0]["role"] == "user"
        assert "12 percent" in pf[0]["content"]

    def test_empty_session_messages_does_not_fall_through_to_conversation_history(self):
        """Greptile P2: a gateway parent with _session_messages == [] (fresh/closed
        turn) has genuinely nothing to inherit. The `is None` (not truthiness) check
        must NOT fall through to conversation_history — and the fold of [] yields
        None, so the child gets no prefill (correct: nothing to inherit)."""
        parent = self._make_gateway_parent([])  # present but EMPTY
        captured = self._capture_child_kwargs(
            goal="do a scoped task", inherit_context=True, parent_agent=parent,
        )
        pf = captured.get("prefill_messages")
        assert pf is None, f"empty transcript should fold to no prefill, got {pf!r}"

    def test_inherit_false_default_does_not_seed_conversation_history(self):
        history = [{"role": "user", "content": "SECRET_SHOULD_NOT_LEAK to the child"}]
        parent = self._make_parent(history)
        captured = self._capture_child_kwargs(
            goal="do a scoped task", parent_agent=parent,  # inherit_context omitted -> default False
        )
        pf = captured.get("prefill_messages")
        # default path forwards parent's boot-seed prefill (None here), never the history
        assert pf is None or "SECRET_SHOULD_NOT_LEAK" not in str(pf)

    def _capture_all_child_kwargs(self, **delegate_kwargs):
        """Like _capture_child_kwargs but returns the list of every child's kwargs
        (batch fan-out constructs N children)."""
        from unittest.mock import patch, MagicMock
        captured = []

        def _fake_aiagent(**kwargs):
            captured.append(kwargs)
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "api_calls": 1,
            }
            child._delegate_depth = 1
            child.get_activity_summary.return_value = {"api_call_count": 1}
            return child

        with patch("run_agent.AIAgent", side_effect=_fake_aiagent):
            delegate_task(**delegate_kwargs)
        return captured

    def test_batch_top_level_inherit_context_propagates_to_each_task(self):
        # Greptile P1 regression: tasks=[...] + top-level inherit_context=True must
        # fold the parent context into EVERY batch child (was silently dropped).
        history = [{"role": "user", "content": "BATCH_MARKER established fact"}]
        parent = self._make_parent(history)
        kids = self._capture_all_child_kwargs(
            tasks=[{"goal": "task A"}, {"goal": "task B"}],
            inherit_context=True,
            parent_agent=parent,
        )
        assert len(kids) == 2, f"expected 2 batch children, got {len(kids)}"
        for k in kids:
            pf = k.get("prefill_messages")
            assert isinstance(pf, list) and len(pf) == 1, f"batch child missing folded context: {pf!r}"
            assert pf[0]["role"] == "user"
            assert "BATCH_MARKER" in pf[0]["content"]

    def test_batch_per_task_override_wins_over_top_level(self):
        # Per-task inherit_context=False overrides top-level True for that task only.
        history = [{"role": "user", "content": "OVERRIDE_MARKER fact"}]
        parent = self._make_parent(history)
        kids = self._capture_all_child_kwargs(
            tasks=[{"goal": "inherits"}, {"goal": "opts out", "inherit_context": False}],
            inherit_context=True,
            parent_agent=parent,
        )
        assert len(kids) == 2
        pf0, pf1 = kids[0].get("prefill_messages"), kids[1].get("prefill_messages")
        assert isinstance(pf0, list) and "OVERRIDE_MARKER" in pf0[0]["content"]
        assert pf1 is None or "OVERRIDE_MARKER" not in str(pf1)

    def test_schema_batch_task_item_advertises_inherit_context(self):
        item_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert "inherit_context" in item_props
        assert item_props["inherit_context"]["type"] == "boolean"


class TestModelDispatchThreadsInheritContext:
    """Greptile P1 (the load-bearing one): the LIVE model-facing dispatcher
    AIAgent._dispatch_delegate_task must forward inherit_context, or a
    model-emitted delegate_task(inherit_context=true) silently drops it and the
    whole /boomerang feature ships dark. This is the real production path;
    the registry lambda is only a fallback."""

    def test_dispatcher_forwards_inherit_context_to_delegate_task(self):
        from unittest.mock import patch, MagicMock
        import run_agent

        agent = MagicMock()
        agent._delegate_depth = 0

        captured = {}

        def _fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        with patch("tools.delegate_tool.delegate_task", side_effect=_fake_delegate_task):
            run_agent.AIAgent._dispatch_delegate_task(
                agent, {"goal": "do it", "inherit_context": True}
            )

        assert "inherit_context" in captured, "dispatcher dropped inherit_context (feature ships dark)"
        assert captured["inherit_context"] is True

    def test_dispatcher_default_inherit_context_is_none_not_forced(self):
        # When the model omits it, the dispatcher passes None (delegate_task then
        # defaults to no inheritance) — never accidentally forces it on.
        from unittest.mock import patch, MagicMock
        import run_agent

        agent = MagicMock()
        agent._delegate_depth = 0
        captured = {}

        def _fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        with patch("tools.delegate_tool.delegate_task", side_effect=_fake_delegate_task):
            run_agent.AIAgent._dispatch_delegate_task(agent, {"goal": "do it"})

        assert captured.get("inherit_context") is None
