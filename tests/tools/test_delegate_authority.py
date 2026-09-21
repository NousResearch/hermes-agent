"""Phase 4 recursion and authority contracts."""
import weakref
from types import SimpleNamespace

import pytest

from tools.delegate_tool_dispatch import _check_delegation_cycle


class ProfileNode:
    __slots__ = ("_delegate_profile_name", "_delegate_parent_ref", "__weakref__")

    def __init__(self, name, parent=None):
        self._delegate_profile_name = name
        self._delegate_parent_ref = weakref.ref(parent) if parent is not None else None


def _node(name, parent=None):
    return ProfileNode(name, parent)


def test_direct_self_profile_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        _check_delegation_cycle(_node("A"), "A")


def test_two_hop_profile_cycle_is_rejected():
    root = _node("A")
    child = _node("B", root)
    with pytest.raises(ValueError, match="cycle"):
        _check_delegation_cycle(child, "A")


def test_longer_profile_cycle_is_rejected():
    root = _node("A")
    b = _node("B", root)
    c = _node("C", b)
    with pytest.raises(ValueError, match="cycle"):
        _check_delegation_cycle(c, "A")


def test_unique_profile_chain_is_allowed():
    root = _node("A")
    child = _node("B", root)
    _check_delegation_cycle(child, "C")


def test_profileless_child_skips_profile_cycle_check():
    _check_delegation_cycle(_node("A"), None)


def test_malformed_ancestry_metadata_fails_closed():
    parent = SimpleNamespace(_delegate_profile_name="A", _delegate_parent_ref="not-a-weakref")
    with pytest.raises(ValueError, match="malformed"):
        _check_delegation_cycle(parent, "B")


def test_ancestry_hop_bound_fails_closed():
    nodes = [_node("root")]
    for index in range(8):
        nodes.append(_node(f"profile-{index}", nodes[-1]))
    with pytest.raises(ValueError, match="8-hop"):
        _check_delegation_cycle(nodes[-1], "new-profile")
