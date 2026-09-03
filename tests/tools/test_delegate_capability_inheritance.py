"""Subagent capability inheritance (follow-up to #94036/#97292).

The trusted-proxy capability map is endpoint-scoped trust: children inherit
it only on the parent's exact route; provider- or endpoint-changing
delegation overrides stay default-deny.
"""

from types import SimpleNamespace

from tools.delegate_tool import _inherit_parent_capabilities


def _parent(capabilities):
    return SimpleNamespace(
        provider="custom:proxy",
        base_url="https://trusted-proxy.corp/v1",
        capabilities=capabilities,
    )


def test_same_route_child_inherits_capability_map():
    parent = _parent({"openai_native_compaction": True})
    assert _inherit_parent_capabilities(parent, None, None) == {
        "openai_native_compaction": True
    }


def test_provider_override_stays_default_deny():
    parent = _parent({"openai_native_compaction": True})
    assert _inherit_parent_capabilities(parent, "openai", None) is None


def test_base_url_override_stays_default_deny():
    parent = _parent({"openai_native_compaction": True})
    assert (
        _inherit_parent_capabilities(parent, None, "https://other.example/v1")
        is None
    )


def test_non_dict_parent_capabilities_yield_none():
    parent = _parent(None)
    assert _inherit_parent_capabilities(parent, None, None) is None


def test_inherited_map_is_sanitized_to_str_bool():
    parent = _parent(
        {"openai_native_compaction": True, "bad": "yes", 3: True, "n": 0}
    )
    assert _inherit_parent_capabilities(parent, None, None) == {
        "openai_native_compaction": True
    }
