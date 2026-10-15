"""Regression: boolean JSON Schema property schemas must not park MCP servers.

See #101669 — mcp 2.0.0's 2025-11-25 InputSchema typed every ``properties``
value as an object, so ``"properties": {"refresh": true}`` (legal JSON Schema
2020-12) failed ``ListToolsResult`` validation and Hermes disabled the whole
server. The same typing applied to OutputSchema.properties; both surfaces are
patched and covered here.

On the post-#102117 tree the compat patch lives in ``tools.mcp_tool_schema``;
the facade ``_ensure_mcp_sdk`` still invokes it on first SDK import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("mcp_types")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Child process proves the facade hook alone; must not import the helper by name.
_FACADE_SEAM_SCRIPT = r"""
from tools.mcp_tool import _ensure_mcp_sdk

assert _ensure_mcp_sdk()

import mcp_types.methods as mcp_methods
from mcp.types import ListToolsResult

raw = {
    "tools": [
        {
            "name": "example_tool",
            "inputSchema": {
                "type": "object",
                "properties": {"refresh": True, "q": {"type": "string"}},
            },
            "outputSchema": {
                "type": "object",
                "properties": {"ok": True, "blocked": False},
            },
        }
    ]
}
mcp_methods.validate_server_result("tools/list", "2025-11-25", raw)
result = ListToolsResult.model_validate(raw)
assert result.tools[0].input_schema["properties"]["refresh"] is True
assert result.tools[0].output_schema["properties"]["ok"] is True
assert result.tools[0].output_schema["properties"]["blocked"] is False
print("facade-seam-ok")
"""


def test_facade_ensure_mcp_sdk_alone_accepts_boolean_property_schemas():
    """Only ``_ensure_mcp_sdk()`` must install wire widening (#101669).

    ``tests/tools/conftest.py`` eagerly calls ``_ensure_mcp_sdk`` before each
    in-process tools test, so a direct ``_patch_...`` call here cannot prove the
    production facade seam. Run an isolated interpreter that never imports the
    helper and never loads that conftest autouse fixture.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(_REPO_ROOT)]
    )
    # Avoid inheriting pytest plugin bootstrap into the child.
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_VERSION", None)

    proc = subprocess.run(
        [sys.executable, "-c", _FACADE_SEAM_SCRIPT],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"facade-seam child failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "facade-seam-ok" in proc.stdout


def test_list_tools_result_accepts_boolean_property_schema():
    from tools.mcp_tool import _ensure_mcp_sdk
    from tools.mcp_tool_schema import _patch_mcp_boolean_property_schemas

    assert _ensure_mcp_sdk()
    # Idempotent if _ensure already patched; also covers calling the helper alone.
    _patch_mcp_boolean_property_schemas()

    import mcp_types.methods as mcp_methods

    raw = {
        "tools": [
            {
                "name": "example_tool",
                "description": "has a boolean property schema",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "refresh": True,
                        "q": {"type": "string"},
                    },
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "ok": True,
                        "blocked": False,
                        "detail": {"type": "string"},
                    },
                },
            },
            {
                "name": "other_tool",
                "description": "unaffected sibling",
                "inputSchema": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                },
            },
            {
                "name": "deny_all_prop",
                "inputSchema": {
                    "type": "object",
                    "properties": {"never": False},
                },
            },
        ]
    }

    # Negotiated pre-2026 sessions validate tools/list against the 2025-11-25
    # surface — this is the path that used to raise dict_type and park the server.
    mcp_methods.validate_server_result("tools/list", "2025-11-25", raw)

    from mcp.types import ListToolsResult

    result = ListToolsResult.model_validate(raw)
    assert len(result.tools) == 3
    assert result.tools[0].input_schema["properties"]["refresh"] is True
    assert result.tools[0].output_schema["properties"]["ok"] is True
    assert result.tools[0].output_schema["properties"]["blocked"] is False
    assert result.tools[0].output_schema["properties"]["detail"]["type"] == "string"
    assert result.tools[2].input_schema["properties"]["never"] is False

    # Wire alias round-trip keeps boolean outputSchema properties.
    dumped = result.model_dump(by_alias=True, mode="json")
    out_props = dumped["tools"][0]["outputSchema"]["properties"]
    assert out_props["ok"] is True
    assert out_props["blocked"] is False
    mcp_methods.validate_server_result("tools/list", "2025-11-25", dumped)


def test_normalize_mcp_input_schema_preserves_boolean_properties():
    from tools.mcp_tool_schema import _normalize_mcp_input_schema

    out = _normalize_mcp_input_schema(
        {
            "type": "object",
            "properties": {
                "refresh": True,
                "q": {"type": "string"},
            },
        }
    )
    assert out["properties"]["refresh"] is True
    assert out["properties"]["q"]["type"] == "string"


def test_boolean_property_schema_compat_patch_is_idempotent():
    from tools.mcp_tool import _ensure_mcp_sdk
    from tools.mcp_tool_schema import _patch_mcp_boolean_property_schemas

    assert _ensure_mcp_sdk()
    _patch_mcp_boolean_property_schemas()
    _patch_mcp_boolean_property_schemas()

    import mcp_types.methods as mcp_methods

    raw = {
        "tools": [
            {
                "name": "t",
                "inputSchema": {
                    "type": "object",
                    "properties": {"refresh": True},
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "ok": True,
                        "blocked": False,
                    },
                },
            }
        ]
    }
    mcp_methods.validate_server_result("tools/list", "2025-11-25", raw)

    from mcp.types import ListToolsResult

    result = ListToolsResult.model_validate(raw)
    assert result.tools[0].input_schema["properties"]["refresh"] is True
    assert result.tools[0].output_schema["properties"]["ok"] is True
    assert result.tools[0].output_schema["properties"]["blocked"] is False
