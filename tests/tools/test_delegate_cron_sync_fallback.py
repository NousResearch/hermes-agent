"""Regression test for #86632: cron's synchronous delegate_task fallback must
return after the child completes — the automatic background review must not
fire inside the delegated child.

Field signature (issue #86632): a cron job's top-level ``delegate_task`` takes
the #66617 stateless-channel synchronous fallback, the child finishes its turn
normally (``Turn ended: reason=text_response``), and the delegation never
returns to the parent — the heartbeat monitor goes stale and the cron
inactivity watchdog kills the job.  Root cause (traced in the issue thread and
fixed by the ``_delegate_depth`` guard in ``AIAgent._spawn_background_review``):
the child's turn finalization spawned the automatic memory/skill background
review fork.  Delegated children must never spawn that fork — it inherits the
child's (often premium) model, replays the whole conversation, and its spawn
inside the child's finalize path is the wedge site the cron watchdog kills.

This exercises the REAL end-to-end #86632 path: a genuine ``AIAgent`` child
(mocked LLM client) with the skill-review trigger armed, dispatched through
``delegate_task(background=True)`` under a session runtime where async delivery
is unsupported (cron), forcing the synchronous fallback.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as dt


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_real_child():
    """A real AIAgent child, as the cron sync fallback runs one.

    The LLM client is mocked to complete in one call, and the post-turn
    skill-review trigger is armed the way a long child run arms it — so turn
    finalization reaches the background-review gate.
    """
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        child = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="subagent",
        )
    child.client = MagicMock()
    child.client.chat.completions.create.return_value = _mock_response(
        content="child work done", finish_reason="stop"
    )
    child._cached_system_prompt = "You are helpful."
    child._use_prompt_caching = False
    child.compression_enabled = False
    child.save_trajectories = False
    child._fallback_chain = []
    # Delegated-child identity, as _build_child_agent stamps it.
    child._delegate_depth = 1
    child._delegate_role = "leaf"
    child._subagent_id = "subagent-86632"
    child._delegate_saved_tool_names = []
    # Arm the post-turn skill review trigger (finalize_turn's
    # _should_review_skills gate) exactly as a long child run would.
    child._skill_nudge_interval = 1
    child._iters_since_skill = 5
    child.valid_tool_names = {"skill_manage"}
    # Keep the test hermetic: no session persistence.
    child._persist_disabled = True
    child._session_db = None
    child._session_json_enabled = False
    return child


def test_cron_sync_fallback_returns_and_spawns_no_review_fork(monkeypatch):
    """The #86632 path: sync fallback completes AND no review fork spawns.

    Red on the pre-fix code: the delegated child's finalize path constructed a
    background-review fork (``fork_constructed`` fires).  With the
    ``_delegate_depth`` guard the fork is never built, removing the wedge site
    entirely, and ``delegate_task`` returns the child's result promptly.
    """
    fork_spawned = threading.Event()
    release_fork = threading.Event()

    def _recording_review(agent_obj, messages_snapshot, prompt):
        # Stands in for the review fork's replay; wedges like the field report.
        fork_spawned.set()
        release_fork.wait(timeout=60)

    child = _make_real_child()
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "cron_244ee2c8b9da"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._session_db = None

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    # spawn_background_review_thread's target resolves _run_review_in_thread
    # from module globals at call time, so this records (and wedges) the
    # review replay without touching the child's own code paths.
    monkeypatch.setattr(
        "agent.background_review._run_review_in_thread", _recording_review
    )

    done: dict = {}

    def _call_delegate_task():
        # Cron declares the channel stateless (#66617): async delivery is
        # unsupported and there is no bound origin session id to wake, so
        # delegate_task must run the batch synchronously.
        with (
            patch(
                "gateway.session_context.async_delivery_supported",
                return_value=False,
            ),
            patch(
                "tools.async_delegation._current_origin_session_id",
                return_value="",
            ),
        ):
            done["out"] = dt.delegate_task(
                goal="do trivial work and finish",
                context="cron regression #86632",
                background=True,
                parent_agent=parent,
            )

    worker = threading.Thread(target=_call_delegate_task, daemon=True)
    worker.start()
    worker.join(timeout=90)
    try:
        # 1) The synchronous fallback must return to the parent. On the field
        #    failure this never happened; here a non-return means the child's
        #    finalize path (or the parent join) wedged.
        assert not worker.is_alive(), (
            "delegate_task synchronous fallback did not return after the "
            "child completed (#86632 wedge)"
        )
        parsed = json.loads(done["out"])
        results = parsed["results"]
        assert len(results) == 1
        assert results[0]["status"] == "completed"
        assert results[0]["summary"] == "child work done"

        # 2) The wedge site must not exist at all: a delegated child
        #    (_delegate_depth > 0) must never spawn the automatic background
        #    review fork. Give a straggling daemon spawn a moment to show up
        #    so a race cannot mask a regression.
        assert not fork_spawned.wait(timeout=3), (
            "automatic background review fork was constructed inside a "
            "delegation subagent — the #86632 wedge site is back"
        )
    finally:
        # Unblock a stray fork thread if the regression reappears, so it
        # cannot outlive this test and pollute the rest of the session.
        release_fork.set()
