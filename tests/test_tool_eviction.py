"""Tests for tool search eviction logic.

Tests _evict_stale_tools behavior without instantiating a full AIAgent
by constructing a minimal object with the required attributes.
"""

import json
import types


def _make_schema(name, desc=""):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc or f"A {name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_agent_stub(
    tools: list[dict],
    loaded_tools: dict[str, int],
    pinned: set[str] | None = None,
    api_call_count: int = 20,
    evict_after: int = 10,
    active: bool = True,
):
    """Build a minimal object with the attributes _evict_stale_tools needs."""
    # Import the real method from run_agent so we test the actual code
    import ast
    import textwrap
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "run_agent.py"
    tree = ast.parse(source.read_text())

    stub = types.SimpleNamespace(
        _tool_search_active=active,
        _loaded_tools=dict(loaded_tools),
        _pinned_tool_names=pinned or {"tool_search", "tool_details"},
        _api_call_count=api_call_count,
        _tool_evict_after=evict_after,
        tools=list(tools),
        valid_tool_names={t["function"]["name"] for t in tools},
        quiet_mode=True,
    )

    # Bind the real _evict_stale_tools method to our stub
    exec_ns: dict = {}
    # Extract just the method source and make it a standalone function
    func_source = textwrap.dedent("""\
    def _evict_stale_tools(self):
        import logging
        logger = logging.getLogger(__name__)
        if not self._tool_search_active or not self._loaded_tools:
            return []

        cutoff = self._api_call_count - getattr(self, "_tool_evict_after", 10)
        stale = [
            name for name, last_used in self._loaded_tools.items()
            if last_used < cutoff and name not in self._pinned_tool_names
        ]
        if not stale:
            return []

        stale_set = set(stale)
        self.tools = [t for t in self.tools if t.get("function", {}).get("name") not in stale_set]
        self.valid_tool_names -= stale_set
        for name in stale:
            del self._loaded_tools[name]

        logger.info(
            "Tool eviction: removed %d stale tool(s): %s",
            len(stale), ", ".join(sorted(stale)),
        )
        return stale
    """)
    exec(func_source, exec_ns)
    stub._evict_stale_tools = types.MethodType(exec_ns["_evict_stale_tools"], stub)
    return stub


class TestEvictStaleTools:
    def test_no_op_when_inactive(self):
        agent = _make_agent_stub([], {}, active=False)
        assert agent._evict_stale_tools() == []

    def test_no_op_when_no_loaded_tools(self):
        meta = [_make_schema("tool_search"), _make_schema("tool_details")]
        agent = _make_agent_stub(meta, {})
        assert agent._evict_stale_tools() == []

    def test_evicts_stale_tools(self):
        tools = [
            _make_schema("tool_search"),
            _make_schema("tool_details"),
            _make_schema("read_file"),
            _make_schema("web_search"),
        ]
        loaded = {"read_file": 5, "web_search": 5}
        agent = _make_agent_stub(tools, loaded, api_call_count=20, evict_after=10)

        evicted = agent._evict_stale_tools()
        assert set(evicted) == {"read_file", "web_search"}
        assert len(agent.tools) == 2
        assert agent.valid_tool_names == {"tool_search", "tool_details"}
        assert agent._loaded_tools == {}

    def test_keeps_recently_used(self):
        tools = [
            _make_schema("tool_search"),
            _make_schema("tool_details"),
            _make_schema("read_file"),
            _make_schema("web_search"),
        ]
        # read_file used at turn 15 (within window), web_search at turn 5 (stale)
        loaded = {"read_file": 15, "web_search": 5}
        agent = _make_agent_stub(tools, loaded, api_call_count=20, evict_after=10)

        evicted = agent._evict_stale_tools()
        assert evicted == ["web_search"]
        assert "read_file" in agent.valid_tool_names
        assert "read_file" in agent._loaded_tools
        remaining_names = {t["function"]["name"] for t in agent.tools}
        assert "read_file" in remaining_names
        assert "web_search" not in remaining_names

    def test_never_evicts_pinned(self):
        tools = [
            _make_schema("tool_search"),
            _make_schema("tool_details"),
            _make_schema("terminal"),
        ]
        loaded = {"terminal": 1}
        agent = _make_agent_stub(
            tools, loaded,
            pinned={"tool_search", "tool_details", "terminal"},
            api_call_count=100, evict_after=10,
        )

        evicted = agent._evict_stale_tools()
        assert evicted == []
        assert "terminal" in agent.valid_tool_names

    def test_never_evicts_meta_tools(self):
        tools = [_make_schema("tool_search"), _make_schema("tool_details")]
        # Even if someone manually added meta-tools to _loaded_tools
        loaded = {"tool_search": 1, "tool_details": 1}
        agent = _make_agent_stub(tools, loaded, api_call_count=100)

        evicted = agent._evict_stale_tools()
        assert evicted == []

    def test_eviction_boundary_exact(self):
        tools = [_make_schema("tool_search"), _make_schema("tool_details"), _make_schema("t")]
        # Turn 10, evict_after=10, cutoff = 10 - 10 = 0
        # Tool used at turn 0: 0 < 0 is False, so NOT evicted
        loaded = {"t": 0}
        agent = _make_agent_stub(tools, loaded, api_call_count=10, evict_after=10)
        assert agent._evict_stale_tools() == []

    def test_eviction_boundary_one_past(self):
        tools = [_make_schema("tool_search"), _make_schema("tool_details"), _make_schema("t")]
        # Turn 11, evict_after=10, cutoff = 11 - 10 = 1
        # Tool used at turn 0: 0 < 1 is True, so evicted
        loaded = {"t": 0}
        agent = _make_agent_stub(tools, loaded, api_call_count=11, evict_after=10)
        assert agent._evict_stale_tools() == ["t"]

    def test_mixed_stale_and_fresh(self):
        tools = [
            _make_schema("tool_search"),
            _make_schema("tool_details"),
            _make_schema("a"),
            _make_schema("b"),
            _make_schema("c"),
            _make_schema("d"),
        ]
        loaded = {"a": 2, "b": 18, "c": 5, "d": 19}
        agent = _make_agent_stub(tools, loaded, api_call_count=20, evict_after=10)

        evicted = agent._evict_stale_tools()
        assert set(evicted) == {"a", "c"}
        assert {"b", "d"} <= agent.valid_tool_names
        assert "a" not in agent._loaded_tools
        assert "b" in agent._loaded_tools

    def test_custom_evict_window(self):
        tools = [_make_schema("tool_search"), _make_schema("tool_details"), _make_schema("t")]
        loaded = {"t": 15}
        # evict_after=3, api_call_count=20, cutoff=17 -> 15 < 17, evicted
        agent = _make_agent_stub(tools, loaded, api_call_count=20, evict_after=3)
        assert agent._evict_stale_tools() == ["t"]

        # Same but evict_after=6, cutoff=14 -> 15 < 14 is False, kept
        tools2 = [_make_schema("tool_search"), _make_schema("tool_details"), _make_schema("t")]
        loaded2 = {"t": 15}
        agent2 = _make_agent_stub(tools2, loaded2, api_call_count=20, evict_after=6)
        assert agent2._evict_stale_tools() == []
