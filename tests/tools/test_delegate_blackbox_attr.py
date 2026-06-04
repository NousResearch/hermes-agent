"""T1b: subagent attribution is set on the child agent at the factory site,
so it survives the ThreadPoolExecutor boundary (contextvars do not)."""
import inspect
import tools.delegate_tool as dt


def test_attribution_set_in_build_child_agent():
    src = inspect.getsource(dt._build_child_agent)
    assert "_blackbox_is_subagent = True" in src
    assert "_blackbox_parent_turn_id" in src
    assert "_blackbox_parent_platform" in src
    assert "_blackbox_parent_chat_id" in src
    # Must be set BEFORE return child (i.e. in the parent thread, pre-submit)
    i_attr = src.index("_blackbox_is_subagent = True")
    i_ret = src.rindex("return child")
    assert i_attr < i_ret, "attribution must be set before return child"


def test_attribution_is_best_effort_guarded():
    """A failure reading session context must not break delegation."""
    src = inspect.getsource(dt._build_child_agent)
    # the attribution block is wrapped in try/except
    blk = src[src.index("_blackbox_is_subagent"):]
    assert "except Exception" in src[:src.index("_blackbox_is_subagent")] or "except Exception" in blk
