"""delegation.child_extra_toolsets: config-granted toolsets for subagents only.

A lean parent (e.g. a chat orchestrator) can keep always-loaded schemas such as ``browser`` out of its own prompt
while its children still get them. Config-owned, never model-facing; blocked toolsets are still stripped and an
explicit parent disable still wins.
"""

from types import SimpleNamespace

import pytest

from tools import delegate_tool_toolsets as dts

PARENT = ["kanban", "memory", "clarify", "session_search", "web", "delegation"]


def _parent(disabled=None):
    return SimpleNamespace(enabled_toolsets=list(PARENT), disabled_toolsets=disabled)


@pytest.fixture
def delegation_cfg(monkeypatch):
    cfg: dict = {}
    monkeypatch.setattr(dts, "_cfg", lambda: cfg)
    return cfg


def test_absent_key_keeps_inherited_set(delegation_cfg):
    child, _ = dts._resolve_child_toolsets(_parent(), None, "leaf")
    assert "browser" not in child
    assert set(child) == {"session_search", "web"}


def test_extra_toolset_reaches_child_not_parent(delegation_cfg):
    delegation_cfg["child_extra_toolsets"] = ["browser"]
    parent = _parent()
    child, disabled = dts._resolve_child_toolsets(parent, None, "leaf")
    assert "browser" in child
    assert "browser" not in disabled
    assert parent.enabled_toolsets == PARENT  # parent list not mutated


def test_extra_toolset_for_orchestrator_role_and_narrowed_call(delegation_cfg):
    delegation_cfg["child_extra_toolsets"] = ["browser"]
    child, _ = dts._resolve_child_toolsets(_parent(), None, "orchestrator")
    assert {"browser", "delegation"} <= set(child)
    narrowed, _ = dts._resolve_child_toolsets(_parent(), ["web"], "leaf")
    assert {"web", "browser"} <= set(narrowed)


def test_string_form_dedupes_and_drops_unknown_and_blocked(delegation_cfg):
    delegation_cfg["child_extra_toolsets"] = "browser, web, no-such-toolset, kanban, memory, browser"
    child, _ = dts._resolve_child_toolsets(_parent(), None, "leaf")
    assert child.count("browser") == 1
    assert child.count("web") == 1
    assert not {"no-such-toolset", "kanban", "memory"} & set(child)


def test_parent_explicit_disable_still_wins(delegation_cfg):
    delegation_cfg["child_extra_toolsets"] = ["browser"]
    _, disabled = dts._resolve_child_toolsets(_parent(disabled=["browser"]), None, "leaf")
    assert "browser" in disabled


@pytest.mark.parametrize("value", [[], "", None, 0])
def test_empty_values_are_noop(delegation_cfg, value):
    baseline = dts._resolve_child_toolsets(_parent(), None, "leaf")
    delegation_cfg["child_extra_toolsets"] = value
    assert dts._resolve_child_toolsets(_parent(), None, "leaf") == baseline


def test_wildcards_are_rejected(delegation_cfg):
    delegation_cfg["child_extra_toolsets"] = ["all", "*"]
    child, _ = dts._resolve_child_toolsets(_parent(), None, "leaf")
    assert not {"all", "*"} & set(child)


def test_plugin_registered_toolset_is_accepted(delegation_cfg, monkeypatch):
    real = dts.validate_toolset
    monkeypatch.setattr(dts, "validate_toolset", lambda name: name == "google_meet" or real(name))
    delegation_cfg["child_extra_toolsets"] = ["google_meet"]
    child, _ = dts._resolve_child_toolsets(_parent(), None, "leaf")
    assert "google_meet" in child
