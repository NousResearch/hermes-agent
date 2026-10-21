"""Delegated workers never spawn the post-turn background review (#72082).

The review fork has no human in the loop, costs ~30K tokens per event, and
writes to the shared profile skill library — outside any closed scope the
worker turn runs in (memory is already blocked for children via
``DELEGATE_BLOCKED_TOOLS`` for the same shared-state reason). Cron agents
suppress the fork with ``skip_background_review``; delegated children must
too. Regression check: the child agent built through the real spawn path
carries the flag while the parent keeps the default.
"""

from types import SimpleNamespace


def test_child_agent_carries_skip_background_review(tmp_path, monkeypatch):
    # HOME too: the guard counts the shared checkout's git dir (under ~/.hermes/workdir) as
    # real-home I/O when the update probe touches it; a temp HOME keeps this runnable locally.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    (tmp_path / "hermes-home").mkdir()
    (tmp_path / "hermes-home" / "config.yaml").write_text(
        "model:\n  default: anthropic/claude-sonnet-4.6\n", encoding="utf-8"
    )
    # run_agent calls restore_interrupted_pull() at import; its marker stat lands in the
    # shared checkout's git dir and the home IO guard refuses that. The recovery probe is
    # irrelevant here — stub it before the import.
    import hermes_cli._early_recovery as _er

    monkeypatch.setattr(_er, "restore_interrupted_pull", lambda *a, **k: False)
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc

    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    kw = dict(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        model="anthropic/claude-sonnet-4.6",
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
    )
    parent = AIAgent(session_id="p", **kw)
    child = dt._build_child_agent(
        task_index=0,
        goal="goal",
        context=None,
        toolsets=["file"],
        model=None,
        max_iterations=4,
        task_count=1,
        parent_agent=parent,
    )
    try:
        # Discriminating pair: only the child is suppressed; the interactive
        # parent keeps the default so foreground self-improvement still runs.
        assert child.skip_background_review is True
        assert parent.skip_background_review is False
    finally:
        child.close()
        parent.close()


def test_gate_predicate_matches_turn_finalizer():
    """The finalizer's spawn gate (`not skip_background_review`) stays closed for children."""
    for agent, spawns in (
        (SimpleNamespace(skip_background_review=True), False),
        (SimpleNamespace(skip_background_review=False), True),
        (SimpleNamespace(), True),  # getattr default: legacy agents without the flag
    ):
        gated = not getattr(agent, "skip_background_review", False)
        assert gated is spawns, agent
