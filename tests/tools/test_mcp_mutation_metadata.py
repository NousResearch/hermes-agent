from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.mcp_tool import _annotation_mutation_spec


def test_mcp_hermes_mutation_annotation_becomes_registry_metadata() -> None:
    tool = SimpleNamespace(
        annotations={
            "readOnlyHint": False,
            "hermesMutation": {
                "action_type": "network",
                "provider": "Firewall",
                "operation": "apply_rule",
            },
        }
    )
    spec = _annotation_mutation_spec(tool)
    assert spec is not None
    assert spec.action_type == "network"
    assert spec.provider == "Firewall"


def test_malformed_or_read_only_mcp_metadata_fails_closed() -> None:
    malformed = SimpleNamespace(
        annotations={"hermesMutation": {"action_type": "network"}}
    )
    read_only = SimpleNamespace(
        annotations={
            "readOnlyHint": True,
            "hermesMutation": {
                "action_type": "network",
                "provider": "Firewall",
                "operation": "apply_rule",
            },
        }
    )
    assert _annotation_mutation_spec(malformed) is None
    assert _annotation_mutation_spec(read_only) is None
    assert _annotation_mutation_spec(
        SimpleNamespace(
            annotations={
                "hermesMutation": {
                    "action_type": "network",
                    "provider": "Firewall\nsecret",
                    "operation": "apply_rule",
                }
            }
        )
    ) is None


def test_real_mcp_sdk_vendor_meta_survives_and_read_only_is_honored() -> None:
    types = pytest.importorskip("mcp.types")
    mutation = {
        "action_type": "network",
        "provider": "Firewall",
        "operation": "apply_rule",
    }
    tool = types.Tool(
        name="apply_rule",
        inputSchema={"type": "object"},
        annotations=types.ToolAnnotations(readOnlyHint=False),
        _meta={"hermesMutation": mutation},
    )
    spec = _annotation_mutation_spec(tool)
    assert spec is not None
    assert spec.operation == "apply_rule"

    readonly_tool = types.Tool(
        name="read_rule",
        inputSchema={"type": "object"},
        annotations=types.ToolAnnotations(readOnlyHint=True),
        _meta={"hermesMutation": mutation},
    )
    assert _annotation_mutation_spec(readonly_tool) is None
