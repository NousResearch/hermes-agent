"""Subagents never inherit the 1h prompt-cache tier: with ``prompt_caching.cache_ttl: 1h`` the parent
keeps 1h, the child is built at 5m, and the child's wire markers carry no ``ttl``. A disabled cache
stays disabled (the child is not re-enabled to 5m)."""
from types import SimpleNamespace
from agent.prompt_caching import apply_anthropic_cache_control
from tools.delegate_tool import _apply_child_cache_ttl


def _markers(messages):
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            out += [b["cache_control"] for b in c if isinstance(b, dict) and "cache_control" in b]
        if "cache_control" in m:
            out.append(m["cache_control"])
    return out


def test_child_1h_becomes_5m_and_wire_markers_drop_ttl():
    child = SimpleNamespace(_cache_ttl="1h")
    _apply_child_cache_ttl(child)
    assert child._cache_ttl == "5m"
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    wire = apply_anthropic_cache_control([dict(m) for m in msgs], cache_ttl=child._cache_ttl)
    marks = _markers(wire)
    assert marks and all("ttl" not in m for m in marks), marks
    # and the parent's 1h layout really is different, so this is not a vacuous check
    parent_marks = _markers(apply_anthropic_cache_control([dict(m) for m in msgs], cache_ttl="1h"))
    assert any(m.get("ttl") == "1h" for m in parent_marks)


def test_disabled_and_5m_are_left_alone():
    for ttl in (None, "5m"):
        child = SimpleNamespace(_cache_ttl=ttl)
        _apply_child_cache_ttl(child)
        assert child._cache_ttl == ttl


def test_real_spawn_path_applies_it(tmp_path, monkeypatch):
    """Through ``_build_child_agent`` with a real AIAgent: parent configured 1h, child ends at 5m, and the
    parent is untouched."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "prompt_caching:\n  cache_ttl: 1h\nmodel:\n  default: anthropic/claude-sonnet-4.6\n", encoding="utf-8")
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent = AIAgent(session_id="p", **kw)
    assert parent._cache_ttl == "1h", "fixture: the operator's 1h must actually be in effect"
    child = dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                  max_iterations=4, task_count=1, parent_agent=parent)
    try:
        assert child._cache_ttl == "5m"
        assert parent._cache_ttl == "1h"
    finally:
        child.close()
        parent.close()
