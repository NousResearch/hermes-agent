"""Regression: composite hermes-* toolsets must stay in sync with terminal toolset."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _register_tools():
    import model_tools  # noqa: F401 — force tool discovery


@pytest.mark.parametrize("composite", ["hermes-acp", "hermes-api-server"])
def test_composite_includes_full_terminal_toolset(composite: str) -> None:
    from toolsets import resolve_toolset

    terminal_tools = set(resolve_toolset("terminal"))
    composite_tools = set(resolve_toolset(composite))
    assert terminal_tools.issubset(composite_tools), (
        f"{composite} missing terminal members: {sorted(terminal_tools - composite_tools)}"
    )
