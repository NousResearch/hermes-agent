from __future__ import annotations

from types import SimpleNamespace

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

