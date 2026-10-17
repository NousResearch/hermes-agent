"""A delegated child must not spawn the end-of-turn background review fork (#133830).

Cron already passes ``skip_background_review=True`` (no human in the loop); the subagent
constructor omitted it, so after the child's final answer a review fork kept the child
process alive and its delegation manifest sat in ``running`` until the 450s stall killer
force-finalized it. This pins the constructor wiring through the real ``_build_child_agent``
spawn path — the finalize-turn gating of the flag itself is covered by
``tests/agent/test_skip_background_review.py``.
"""
from __future__ import annotations


def test_child_skips_background_review_but_parent_keeps_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: anthropic/claude-sonnet-4.6\n", encoding="utf-8")
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr(dtc, "_load_config", dict)
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent = AIAgent(session_id="p", **kw)
    assert parent.skip_background_review is False, "fixture: the parent really does review by default"
    child = dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                  max_iterations=4, task_count=1, parent_agent=parent)
    try:
        assert child.skip_background_review is True
    finally:
        child.close()
        parent.close()
