"""Delegated children run on ``delegation.cache_ttl`` (#104168): ``5m`` by default, so neither
``prompt_caching.cache_ttl: 1h`` nor ``auto`` hands a fast fan-out the 2x-write 1h tier; ``1h`` on
request for slow children, whose next call often starts more than five minutes after the previous
request did. A disabled cache stays disabled (the child is not re-enabled to 5m)."""
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.prompt_caching import apply_anthropic_cache_control
from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
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
    _apply_child_cache_ttl(child, {})
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
        _apply_child_cache_ttl(child, {})
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
    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr(dtc, "_load_config", dict)
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


def _spawn(base_url, **parent_kw):
    """A real parent AIAgent and a child built by ``_build_child_agent``, both reading the temp
    HERMES_HOME's config.yaml through the production loaders (no ``_load_config`` patch)."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    parent = AIAgent(base_url=base_url, platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
                     save_trajectories=False, enabled_toolsets=["file"], session_id="parent", **parent_kw)
    child = dt._build_child_agent(task_index=0, goal="Reply with one word.", context=None, toolsets=["file"],
                                  model=None, max_iterations=2, task_count=1, parent_agent=parent)
    return parent, child


def test_delegation_cache_ttl_1h_reaches_the_child_wire():
    """``prompt_caching.cache_ttl: auto`` + ``delegation.cache_ttl: 1h``: the child is built on 1h and every
    cache marker in its requests carries ``ttl: 1h``, on a caching route recorded by a fake provider."""
    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer([Text("done")]) as srv:
        route = (f"custom_providers:\n  - name: cacherelay\n    base_url: {srv.base_url}\n    api_key: sk-fake-e2e\n"
                 "    models:\n      fake-model:\n        prompt_caching: true\n")
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + route
                          + "prompt_caching:\n  cache_ttl: auto\ndelegation:\n  cache_ttl: 1h\n")
        parent, child = _spawn(srv.base_url, provider="custom:cacherelay", api_key="sk-fake-e2e", model="fake-model")
        try:
            assert child._use_prompt_caching, "fixture: the route must actually cache"
            assert child._cache_ttl == "1h"
            child.run_conversation("Reply with one word.", task_id="child")
        finally:
            child.close()
            parent.close()
    requests = srv.main_requests()
    assert requests, "the child never reached the provider"
    marks = [m for r in requests for m in _markers(r["messages"])]
    assert marks and all(m.get("ttl") == "1h" for m in marks), marks


@pytest.mark.parametrize(("prompt_caching", "delegation", "expected", "warns"), [
    ("auto", "", "5m", False),                     # key absent: an inherited human source never gives 1h
    ("off", "delegation:\n  cache_ttl: 1h\n", None, False),  # disabled stays disabled
    ("auto", "delegation:\n  cache_ttl: 2h\n", "5m", True),  # invalid: one warning, then 5m
], ids=["key-absent", "caching-disabled", "invalid-value"])
def test_child_tier_without_a_valid_1h(monkeypatch, caplog, prompt_caching, delegation, expected, warns):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        f"prompt_caching:\n  cache_ttl: {prompt_caching}\nmodel:\n  default: anthropic/claude-sonnet-4.6\n"
        + delegation, encoding="utf-8")
    # The parent's source reaches the child's ``auto`` resolution, which would pick 1h on its own.
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "telegram")
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        parent, child = _spawn("https://openrouter.ai/api/v1", provider="openrouter", api_key="k",
                               api_mode="chat_completions", model="anthropic/claude-sonnet-4.6")
    try:
        assert child._cache_ttl == expected
        assert child._cache_disabled is (expected is None)
        warnings = [r for r in caplog.records if "delegation.cache_ttl" in r.getMessage()]
        assert len(warnings) == (1 if warns else 0), [r.getMessage() for r in warnings]
    finally:
        child.close()
        parent.close()
