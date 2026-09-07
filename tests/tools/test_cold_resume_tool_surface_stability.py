"""Cold-resume tool-surface byte stability — regression guard for the #103579 族 fix.

Background (observed 2026-09-07, session 20260907_220529_5021de):
  A dashboard/gateway restart cold-resumes a rebuilt agent whose tool surface
  is reassembled by ``assemble_tool_defs``. The ``tool_search`` bridge
  description embeds the **runtime deferrable set** (plugin/MCP tools that pass
  check_fn) as a deferred_count + full catalog listing (name + description),
  so:

    same config, same plugin set — if the deferrable set at *build time*
    differs from the previous process (plugin registration order / check_fn
    pass-through / async platform tool registration), the tool-surface bytes
    differ → the API request's tools[] differs → the cache prefix breaks right
    after the system prompt (hit residue == system-prompt, 14,592 tokens
    measured in this environment).

  The earlier fix (bdc21cb777) froze the tool-surface inheritance for
  **bg-review forks** only; the main-thread cold resume had no snapshot
  mechanism — "rebuild every time" — hence "first turn after a restart breaks,
  next turn recovers".

Test layers:
  * Mechanism (PASS, proves existence): different deferrable sets → different
    bytes; same set twice → identical bytes (no randomness).
  * Contract layer (documentation): byte changes with the deferrable set are
    by design — that is why the snapshot exists. The byte-stability contract
    lives at the agent layer, carried by
    ``tests/agent/test_tool_surface_snapshot.py`` (config fingerprint
    unchanged → cold resume must reuse the session-snapshot bytes regardless
    of registry state).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Mini OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


_CORE = _td("terminal", "Run shell commands.")
_CORE2 = _td("read_file", "Read a file.")

_PLUGIN_A = _td("feishu_doc_read", "Read the full content of a Feishu/Lark document as plain text.")
_PLUGIN_B = _td("feishu_drive_add_comment", "Add a new whole-document comment on a Feishu document.")
_PLUGIN_C = _td("feishu_drive_list_comments", "List all comments in a comment thread on a Feishu document.")
_PLUGIN_D = _td("video_analyze", "Analyze a video from a URL or local file path.")


def _assemble(tool_defs: List[Dict[str, Any]], context_length: int = 1_000_000) -> List[Dict[str, Any]]:
    from tools.tool_search import ToolSearchConfig, assemble_tool_defs

    cfg = ToolSearchConfig.from_raw(None)  # enabled=auto — activates when deferrables exist
    result = assemble_tool_defs(tool_defs, context_length=context_length, config=cfg)
    return result.tool_defs


def _register_mcp_tool(name: str, description: str, toolset: str = "mcp-cold-resume") -> None:
    """Register a plugin/MCP tool in the global registry so classify_tools
    treats it as deferrable (is_deferrable_tool_name queries registry.get_entry)."""
    from tools.registry import registry

    registry.register(
        name=name,
        handler=lambda args, **kw: "{}",
        schema=_td(name, description)["function"],
        toolset=toolset,
    )


def _tool_search_desc(tool_defs: List[Dict[str, Any]]) -> str:
    for td in tool_defs:
        fn = td.get("function") or {}
        if fn.get("name") == "tool_search":
            return fn.get("description", "")
    raise AssertionError("tool_search bridge missing from assembled output")


# Plugin/MCP tool names (unique prefix per test to avoid global-registry pollution)
_P = "coldresume"


class TestMechanism:
    """Mechanism layer: tool-surface bytes are a function of the deferrable set."""

    def test_deferred_catalog_embedded_in_bridge_description(self):
        """The bridge description must embed deferred_count + catalog (current design)."""
        from tools.registry import discover_builtin_tools

        discover_builtin_tools()
        names = [f"{_P}_n{i}" for i in range(4)]
        for i, n in enumerate(names):
            _register_mcp_tool(n, f"Plugin tool {i}.")
        tool_defs = [_CORE, _CORE2] + [_td(n, f"Plugin tool {i}.") for i, n in enumerate(names)]
        result = _assemble(tool_defs)
        desc = _tool_search_desc(result)
        assert "Search 4 additional tools" in desc
        for n in names:
            assert n in desc, f"catalog listing must mention {n}"

    def test_partial_set_produces_different_bytes(self):
        """Same plugin set, different registration/check_fn pass-through (4→2) → different bytes."""
        from tools.registry import discover_builtin_tools

        discover_builtin_tools()
        full_names = [f"{_P}_full_{i}" for i in range(4)]
        for i, n in enumerate(full_names):
            _register_mcp_tool(n, f"Full tool {i}.")
        full = [_CORE, _CORE2] + [_td(n, f"Full tool {i}.") for i, n in enumerate(full_names)]
        partial = [_CORE, _CORE2] + [_td(n, f"Full tool {i}.") for i, n in enumerate(full_names[:2])]
        sa = json.dumps(_assemble(full), ensure_ascii=False, sort_keys=True)
        sb = json.dumps(_assemble(partial), ensure_ascii=False, sort_keys=True)
        assert sa != sb, "different deferred sets must produce different wire bytes (bug carrier)"

    def test_same_set_same_bytes(self):
        """Repeating an identical set must produce byte-identical output (no random sort/race)."""
        from tools.registry import discover_builtin_tools

        discover_builtin_tools()
        names = [f"{_P}_stable_{i}" for i in range(3)]
        for i, n in enumerate(names):
            _register_mcp_tool(n, f"Stable tool {i}.")
        tool_defs = [_CORE, _CORE2] + [_td(n, f"Stable tool {i}.") for i, n in enumerate(names)]
        s1 = json.dumps(_assemble(tool_defs), ensure_ascii=False, sort_keys=True)
        s2 = json.dumps(_assemble(tool_defs), ensure_ascii=False, sort_keys=True)
        assert s1 == s2


class TestByteStabilityContract:
    """Contract layer (documentation): assemble-layer bytes changing with the
    deferrable set is by design.

    That is exactly why the snapshot exists — the byte-stability contract moved
    up to the agent layer and is carried by
    ``tests/agent/test_tool_surface_snapshot.py``:
    config fingerprint (model/toolsets) unchanged → cold resume must reuse the
    session-snapshot bytes regardless of registry/check_fn state
    (test_registry_changes_do_not_invalidate_snapshot). This class no longer
    asserts assemble-layer byte stability (that would be the wrong contract).
    """

    def test_bytes_change_with_deferred_set(self):
        """Bytes change with the set = the necessity of the snapshot (if stable, no fix needed)."""
        from tools.registry import discover_builtin_tools

        discover_builtin_tools()
        names = [f"{_P}_need_{i}" for i in range(4)]
        for i, n in enumerate(names):
            _register_mcp_tool(n, f"Need tool {i}.")
        full = [_CORE, _CORE2] + [_td(n, f"Need tool {i}.") for i, n in enumerate(names)]
        partial = [_CORE, _CORE2] + [_td(n, f"Need tool {i}.") for i, n in enumerate(names[:2])]
        sa = json.dumps(_assemble(full), ensure_ascii=False, sort_keys=True)
        sb = json.dumps(_assemble(partial), ensure_ascii=False, sort_keys=True)
        assert sa != sb

    def test_registration_order_does_not_change_bytes(self):
        """Registration order does not affect bytes (the break driver is the set, not order)."""
        from tools.registry import discover_builtin_tools

        discover_builtin_tools()
        names = [f"{_P}_order2_{i}" for i in range(4)]
        for i, n in enumerate(names):
            _register_mcp_tool(n, f"Order tool {i}.")
        a = [_CORE, _CORE2] + [_td(n, f"Order tool {i}.") for i, n in enumerate(names)]
        b = [_CORE, _CORE2] + [_td(n, f"Order tool {i}.") for i, n in reversed(list(enumerate(names)))]
        sa = json.dumps(_assemble(a), ensure_ascii=False, sort_keys=True)
        sb = json.dumps(_assemble(b), ensure_ascii=False, sort_keys=True)
        assert sa == sb
